// RK3588 Android OpenCL reference full KataGo 10b NN benchmark.
// Executes every operation in the K10NW001 19x19 FP16 packet on a GPU device.
// No CPU or NNAPI fallback. Deliberately generic kernels, not tuned KataGo GPU kernels.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using u32=uint32_t;
using Clock=std::chrono::steady_clock;
using cl_int=int;
using cl_uint=unsigned;
using cl_ulong=unsigned long;
using cl_device_type=unsigned long;
using cl_mem_flags=unsigned long;
using cl_context_properties=intptr_t;
using cl_platform_id=struct _cl_platform_id*;
using cl_device_id=struct _cl_device_id*;
using cl_context=struct _cl_context*;
using cl_command_queue=struct _cl_command_queue*;
using cl_mem=struct _cl_mem*;
using cl_program=struct _cl_program*;
using cl_kernel=struct _cl_kernel*;
constexpr cl_device_type DEVICE_GPU=4;
constexpr cl_mem_flags MEM_READ_WRITE=1,MEM_COPY_HOST_PTR=32;
constexpr cl_uint CL_DEVICE_NAME=0x102B,CL_DEVICE_EXTENSIONS=0x1030,CL_PROGRAM_BUILD_LOG=0x1183;
constexpr int FP16=8,CONST=1,INPUT=2;
struct CL {
    void* lib=nullptr;
    cl_int (*platforms)(cl_uint,cl_platform_id*,cl_uint*)=nullptr;
    cl_int (*devices)(cl_platform_id,cl_device_type,cl_uint,cl_device_id*,cl_uint*)=nullptr;
    cl_int (*deviceInfo)(cl_device_id,cl_uint,size_t,void*,size_t*)=nullptr;
    cl_context (*context)(const cl_context_properties*,cl_uint,const cl_device_id*,void*,void*,cl_int*)=nullptr;
    cl_command_queue (*queue)(cl_context,cl_device_id,cl_ulong,cl_int*)=nullptr;
    cl_program (*program)(cl_context,cl_uint,const char**,const size_t*,cl_int*)=nullptr;
    cl_int (*build)(cl_program,cl_uint,const cl_device_id*,const char*,void*,void*)=nullptr;
    cl_int (*programInfo)(cl_program,cl_device_id,cl_uint,size_t,void*,size_t*)=nullptr;
    cl_kernel (*kernel)(cl_program,const char*,cl_int*)=nullptr;
    cl_mem (*buffer)(cl_context,cl_mem_flags,size_t,void*,cl_int*)=nullptr;
    cl_int (*arg)(cl_kernel,cl_uint,size_t,const void*)=nullptr;
    cl_int (*enqueue)(cl_command_queue,cl_kernel,cl_uint,const size_t*,const size_t*,const size_t*,cl_uint,const void*,void*)=nullptr;
    cl_int (*read)(cl_command_queue,cl_mem,cl_uint,size_t,size_t,void*,cl_uint,const void*,void*)=nullptr;
    cl_int (*finish)(cl_command_queue)=nullptr;
    cl_int (*freeBuffer)(cl_mem)=nullptr;
    cl_int (*freeKernel)(cl_kernel)=nullptr;
    cl_int (*freeProgram)(cl_program)=nullptr;
    cl_int (*freeQueue)(cl_command_queue)=nullptr;
    cl_int (*freeContext)(cl_context)=nullptr;
    template<typename T>void bind(T& dest,const char* name) {
        dest=reinterpret_cast<T>(dlsym(lib,name));
        if(!dest)throw std::runtime_error(std::string("OpenCL symbol missing: ")+name);
    }
    void init() {
        for(const char* name:{"libOpenCL.so","/vendor/lib64/libOpenCL.so","/system/vendor/lib64/libOpenCL.so","libGLES_mali.so"}) {
            lib=dlopen(name,RTLD_NOW|RTLD_LOCAL);
            if(lib)break;
        }
        if(!lib)throw std::runtime_error("OpenCL GPU library not found on RK3588 Android");
#define LOAD(field,sym) bind(field,"cl" sym)
        LOAD(platforms,"GetPlatformIDs");LOAD(devices,"GetDeviceIDs");LOAD(deviceInfo,"GetDeviceInfo");
        LOAD(context,"CreateContext");LOAD(queue,"CreateCommandQueue");LOAD(program,"CreateProgramWithSource");
        LOAD(build,"BuildProgram");LOAD(programInfo,"GetProgramBuildInfo");LOAD(kernel,"CreateKernel");
        LOAD(buffer,"CreateBuffer");LOAD(arg,"SetKernelArg");LOAD(enqueue,"EnqueueNDRangeKernel");
        LOAD(read,"EnqueueReadBuffer");LOAD(finish,"Finish");LOAD(freeBuffer,"ReleaseMemObject");
        LOAD(freeKernel,"ReleaseKernel");LOAD(freeProgram,"ReleaseProgram");LOAD(freeQueue,"ReleaseCommandQueue");
        LOAD(freeContext,"ReleaseContext");
#undef LOAD
    }
};
void ck(cl_int rc,const char* what) {
    if(rc)throw std::runtime_error(std::string(what)+" rc="+std::to_string(rc));
}
void bytes(std::ifstream& f,void* dest,size_t n) {
    if(n)f.read(static_cast<char*>(dest),n);
    if(!f)throw std::runtime_error("Truncated K10NW001 packet");
}
u32 u32read(std::ifstream& f){u32 v=0;bytes(f,&v,4);return v;}
struct Operand {
    u32 type,flags;
    std::vector<u32> dims;
    std::vector<char> raw;
    cl_mem buffer=nullptr;
    size_t count=1;
};
struct Op {
    u32 type;
    std::vector<u32> ins,outs;
};
struct Golden {
    std::string name;
    std::vector<float> values;
};
struct Packet {
    std::vector<Operand> operands;
    std::vector<Op> ops;
    std::vector<u32> inputs,outputs;
    std::vector<Golden> golden;
};
Packet parse(const std::string& path) {
    std::ifstream f(path,std::ios::binary);
    if(!f)throw std::runtime_error("Missing KataGo full model packet: "+path);
    char magic[8];bytes(f,magic,8);
    if(std::memcmp(magic,"K10NW001",8))throw std::runtime_error("Unsupported model packet signature");
    u32 no=u32read(f),nop=u32read(f),ni=u32read(f),nout=u32read(f);
    if(no<100||no>2000||nop<100||nop>1000||ni!=2||nout!=5)
        throw std::runtime_error("Unexpected complete 10b packet structure");
    Packet p;p.operands.reserve(no);
    for(u32 i=0;i<no;i++) {
        Operand a{};a.type=u32read(f);
        u32 rank=u32read(f);a.flags=u32read(f);u32 size=u32read(f);
        u32 d[4];bytes(f,d,16);
        if(rank>4||size>40*1024*1024)throw std::runtime_error("Invalid operand header");
        a.dims.assign(d,d+rank);
        a.count=1;
        for(u32 x:a.dims) {
            if(x==0||a.count>10000000/x)throw std::runtime_error("Invalid operand shape");
            a.count*=x;
        }
        a.raw.resize(size);bytes(f,a.raw.data(),size);
        if((a.flags&(CONST|INPUT))&&(a.type==FP16?size!=a.count*2:
            (a.type==6?size!=a.count:size!=a.count*4)))
            throw std::runtime_error("Bad operand data size at "+std::to_string(i));
        p.operands.push_back(std::move(a));
    }
    for(u32 i=0;i<nop;i++) {
        Op op{};op.type=u32read(f);u32 nin=u32read(f),noo=u32read(f);
        if(nin>16||noo!=1)throw std::runtime_error("Invalid operation");
        op.ins.resize(nin);op.outs.resize(noo);bytes(f,op.ins.data(),nin*4);bytes(f,op.outs.data(),noo*4);
        for(u32 idx:op.ins)if(idx>=no)throw std::runtime_error("Bad operand index");
        for(u32 idx:op.outs)if(idx>=no)throw std::runtime_error("Bad output index");
        p.ops.push_back(std::move(op));
    }
    p.inputs.resize(ni);p.outputs.resize(nout);bytes(f,p.inputs.data(),ni*4);bytes(f,p.outputs.data(),nout*4);
    for(u32 i=0;i<nout;i++) {
        Golden v;u32 len=u32read(f);
        if(len>128)throw std::runtime_error("Bad output label");
        v.name.resize(len);if(len)bytes(f,&v.name[0],len);
        u32 size=u32read(f);if(size%4||size>1000000)throw std::runtime_error("Invalid reference tensor");
        v.values.resize(size/4);bytes(f,v.values.data(),size);
        p.golden.push_back(std::move(v));
    }
    if(p.operands[p.inputs[0]].dims!=std::vector<u32>({1,19,19,22})||
       p.operands[p.inputs[1]].dims!=std::vector<u32>({1,1,1,19}))
        throw std::runtime_error("Only original full-board KataGo 10b FP16 accepted");
    return p;
}
float halfFloat(uint16_t h) {
    u32 sign=(h&0x8000u)<<16, exponent=(h>>10)&31u, frac=h&1023u, bits=0;
    if(exponent==0) {
        if(!frac)bits=sign;
        else {
            int e=-14;
            while((frac&1024u)==0){frac<<=1;--e;}
            frac&=1023u;
            bits=sign|(u32(e+127)<<23)|(frac<<13);
        }
    } else if(exponent==31)bits=sign|0x7F800000u|(frac<<13);
    else bits=sign|((exponent+112u)<<23)|(frac<<13);
    float x;std::memcpy(&x,&bits,4);return x;
}
const char* kernels=R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void ew(__global const half* a,__global const half* b,__global half* o,
int oh,int ow,int oc,int ah,int aw,int ac,int bh,int bw,int bc,int mode) {
    int i=get_global_id(0),c=i%oc,x=(i/oc)%ow,y=i/(oc*ow);
    int ia=((ah==1?0:y)*aw+(aw==1?0:x))*ac+(ac==1?0:c);
    int ib=((bh==1?0:y)*bw+(bw==1?0:x))*bc+(bc==1?0:c);
    o[i]=mode==0?(half)((float)a[ia]+(float)b[ib]):(half)((float)a[ia]*(float)b[ib]);
}
__kernel void relu(__global const half* a,__global half* o,int n){
    int i=get_global_id(0);if(i<n)o[i]=max(a[i],(half)0);
}
__kernel void conv(__global const half* a,__global const half* w,__global const half* bias,__global half* o,
int ih,int iw,int ic,int oh,int ow,int oc,int kh,int kw,int pt,int pl) {
    int index=get_global_id(0), c=index%oc,x=(index/oc)%ow,y=index/(oc*ow);
    float sum=(float)bias[c];
    for(int ky=0;ky<kh;ky++) {
        int iy=y+ky-pt;
        if(iy<0||iy>=ih)continue;
        for(int kx=0;kx<kw;kx++) {
            int ix=x+kx-pl;
            if(ix<0||ix>=iw)continue;
            int ii=(iy*iw+ix)*ic,wi=((ky*kw+kx)*ic)*oc+c;
            for(int ci=0;ci<ic;ci++)sum=fma((float)a[ii+ci],(float)w[wi+ci*oc],sum);
        }
    }
    o[index]=(half)sum;
}
// Blocked implicit-GEMM convolution: M=spatial points, N=output channels,
// K=kernel height*kernel width*input channels. Local-memory reuse avoids
// reloading each input 16 times and each weight across 8 different positions.
// Each workgroup covers 8 output pixels x 16 output channels, K tiled by 32.
#define TILE_M 8
#define TILE_N 16
#define TILE_K 32
__kernel void conv_tiled(__global const half* a,__global const half* weights,
                         __global const half* bias,__global half* out,
                         int ih,int iw,int ic,int oh,int ow,int oc,int kh,int kw,int pt,int pl) {
    const int n=get_group_id(0)*TILE_N+get_local_id(0);
    const int m=get_group_id(1)*TILE_M+get_local_id(1);
    const int lane=get_local_id(1)*TILE_N+get_local_id(0);
    const int totalK=kh*kw*ic;
    __local half A[TILE_M*TILE_K];
    __local half B[TILE_K*TILE_N];
    float sum=0.0f;
    for(int k0=0;k0<totalK;k0+=TILE_K) {
        for(int t=lane;t<TILE_M*TILE_K;t+=TILE_M*TILE_N) {
            int localM=t/TILE_K,localK=t%TILE_K;
            int pixel=get_group_id(1)*TILE_M+localM,k=k0+localK;
            half v=(half)0;
            if(pixel<oh*ow&&k<totalK) {
                int y=pixel/ow,x=pixel%ow;
                int spatial=k/ic,channel=k%ic;
                int iy=y+spatial/kw-pt,ix=x+spatial%kw-pl;
                if(iy>=0&&iy<ih&&ix>=0&&ix<iw)v=a[(iy*iw+ix)*ic+channel];
            }
            A[t]=v;
        }
        for(int t=lane;t<TILE_K*TILE_N;t+=TILE_M*TILE_N) {
            int localK=t/TILE_N,localN=t%TILE_N;
            int k=k0+localK,c=get_group_id(0)*TILE_N+localN;
            B[t]=(k<totalK&&c<oc)?weights[k*oc+c]:(half)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int k=0;k<TILE_K;k++)
            sum=fma((float)A[get_local_id(1)*TILE_K+k],
                    (float)B[k*TILE_N+get_local_id(0)],sum);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(m<oh*ow&&n<oc)out[m*oc+n]=(half)(sum+(float)bias[n]);
}
__kernel void pool(__global const half* a,__global half* o,int h,int w,int c,int ismax) {
    int ci=get_global_id(0);float v=ismax?-3.402823466e+38F:0.0F;
    for(int i=0;i<h*w;i++) {
        float x=(float)a[i*c+ci];v=ismax?fmax(v,x):v+x;
    }
    o[ci]=(half)(ismax?v:v/(float)(h*w));
}
__kernel void cat(__global const half* a,__global half* o,int h,int w,int sc,int dc,int off) {
    int index=get_global_id(0),c=index%sc,pixel=index/sc;
    o[pixel*dc+off+c]=a[index];
}
)CLC";

class Runner {
    CL cl;
    Packet packet;
    cl_device_id gpu=nullptr;
    cl_context ctx=nullptr;
    cl_command_queue q=nullptr;
    cl_program prog=nullptr;
    cl_kernel ew=nullptr,relu=nullptr,conv=nullptr,convTiled=nullptr,pool=nullptr,cat=nullptr;
    bool tiledConv=true;
    size_t tiledCount=0,directCount=0;
    template<typename T> void set(cl_kernel kernel,u32 index,const T& value){ck(cl.arg(kernel,index,sizeof(value),&value),"clSetKernelArg");}
    void bufarg(cl_kernel kernel,u32 index,u32 id){set(kernel,index,packet.operands.at(id).buffer);}
    static int dim(const Operand& o,int axis) {
        if(o.dims.empty())return 1;
        if(o.dims.size()==1)return axis==3?int(o.dims[0]):1;
        if(o.dims.size()==4)return int(o.dims[axis]);
        throw std::runtime_error("Unsupported rank in OpenCL kernel");
    }
    int intvalue(u32 id) {
        const auto& o=packet.operands.at(id);
        if(o.raw.size()!=4)throw std::runtime_error("Expected constant INT32");
        int v;std::memcpy(&v,o.raw.data(),4);return v;
    }
    void launch(cl_kernel k,size_t n) {
        ck(cl.enqueue(q,k,1,nullptr,&n,nullptr,0,nullptr,nullptr),"clEnqueueNDRangeKernel");
    }
    void launchTiled(cl_kernel k,int pixels,int outChannels) {
        const size_t local[]={16,8};
        const size_t global[]={size_t((outChannels+15)/16*16),size_t((pixels+7)/8*8)};
        ck(cl.enqueue(q,k,2,nullptr,global,local,0,nullptr,nullptr),"clEnqueueNDRangeKernel tiled");
    }
    void operation(const Op& op) {
        const auto& out=packet.operands[op.outs[0]];
        u32 src=op.ins.at(0),dst=op.outs[0];
        const auto& a=packet.operands[src];
        switch(op.type){
          case 0:case 18: {
              const auto& b=packet.operands[op.ins[1]];
              if(dim(out,0)!=1)throw std::runtime_error("Only batch=1");
              bufarg(ew,0,src);bufarg(ew,1,op.ins[1]);bufarg(ew,2,dst);
              for(int axis=1;axis<4;axis++)set(ew,3+axis-1,dim(out,axis));
              for(int axis=1;axis<4;axis++)set(ew,6+axis-1,dim(a,axis));
              for(int axis=1;axis<4;axis++)set(ew,9+axis-1,dim(b,axis));
              int mode=op.type==0?0:1;set(ew,12,mode);
              launch(ew,out.count);break;
          }
          case 19: {
              bufarg(relu,0,src);bufarg(relu,1,dst);
              int n=int(out.count);set(relu,2,n);launch(relu,out.count);break;
          }
          case 3: {
              const auto& w=packet.operands[op.ins[1]];
              cl_kernel k=tiledConv?convTiled:conv;
              bufarg(k,0,src);bufarg(k,1,op.ins[1]);bufarg(k,2,op.ins[2]);bufarg(k,3,dst);
              int pars[]={dim(a,1),dim(a,2),dim(a,3),dim(out,1),dim(out,2),dim(out,3),
                          int(w.dims.at(1)),int(w.dims.at(2)),intvalue(op.ins[5]),intvalue(op.ins[3])};
              for(int i=0;i<10;i++)set(k,4+i,pars[i]);
              if(tiledConv)launchTiled(k,dim(out,1)*dim(out,2),dim(out,3));
              else launch(k,out.count);
              break;
          }
          case 31:case 77: {
              bufarg(pool,0,src);bufarg(pool,1,dst);
              int h=dim(a,1),w=dim(a,2),c=dim(a,3),ismax=op.type==77?1:0;
              set(pool,2,h);set(pool,3,w);set(pool,4,c);set(pool,5,ismax);
              launch(pool,c);break;
          }
          case 2: {
              int h=dim(out,1),w=dim(out,2),dc=dim(out,3),off=0;
              for(size_t i=0;i+1<op.ins.size();i++){
                  const auto& part=packet.operands[op.ins[i]];
                  int sc=dim(part,3);
                  bufarg(cat,0,op.ins[i]);bufarg(cat,1,dst);
                  set(cat,2,h);set(cat,3,w);set(cat,4,sc);set(cat,5,dc);set(cat,6,off);
                  launch(cat,part.count);off+=sc;
              }
              if(off!=dc)throw std::runtime_error("Concatenation channel mismatch");
              break;
          }
          default:throw std::runtime_error("Unsupported converted op: "+std::to_string(op.type));
        }
    }
public:
    explicit Runner(const std::string& path):packet(parse(path)) {
        const char* mode=std::getenv("KATAGO_GPU_CONV");
        if(mode&&std::strcmp(mode,"direct")==0)tiledConv=false;
        else if(mode&&std::strcmp(mode,"tiled")!=0)
            throw std::runtime_error("KATAGO_GPU_CONV must be tiled or direct");
        for(const auto& op:packet.ops)if(op.type==3)(tiledConv?tiledCount:directCount)++;
        std::printf("GPU_CONV_KERNEL mode=%s tile=8x16x32 conv_ops=%zu\n",tiledConv?"tiled":"direct",tiledCount+directCount);
        cl.init();
        cl_uint np=0;ck(cl.platforms(0,nullptr,&np),"clGetPlatformIDs");if(!np)throw std::runtime_error("No OpenCL platforms");
        std::vector<cl_platform_id> platforms(np);ck(cl.platforms(np,platforms.data(),nullptr),"clGetPlatformIDs");
        for(auto platform:platforms) {
            cl_uint n=0;
            if(cl.devices(platform,DEVICE_GPU,0,nullptr,&n)!=0||n==0)continue;
            std::vector<cl_device_id> devs(n);
            ck(cl.devices(platform,DEVICE_GPU,n,devs.data(),nullptr),"clGetDeviceIDs");
            gpu=devs[0];break;
        }
        if(!gpu)throw std::runtime_error("No actual OpenCL GPU device found; CPU fallback forbidden");
        char name[256]={},extensions[16384]={};
        ck(cl.deviceInfo(gpu,CL_DEVICE_NAME,sizeof(name),name,nullptr),"clGetDeviceInfo name");
        ck(cl.deviceInfo(gpu,CL_DEVICE_EXTENSIONS,sizeof(extensions),extensions,nullptr),"clGetDeviceInfo extensions");
        if(!std::strstr(extensions,"cl_khr_fp16"))throw std::runtime_error("GPU does not expose cl_khr_fp16");
        std::printf("GPU_DEVICE name=%s backend=OpenCL dtype=FP16 no_cpu_fallback=yes\n",name);
        cl_int rc=0;ctx=cl.context(nullptr,1,&gpu,nullptr,nullptr,&rc);ck(rc,"clCreateContext");
        q=cl.queue(ctx,gpu,0,&rc);ck(rc,"clCreateCommandQueue");
        const size_t length=std::strlen(kernels);
        prog=cl.program(ctx,1,&kernels,&length,&rc);ck(rc,"clCreateProgramWithSource");
        rc=cl.build(prog,1,&gpu,"-cl-std=CL1.2",nullptr,nullptr);
        if(rc) {
            char log[16384]={};cl.programInfo(prog,gpu,CL_PROGRAM_BUILD_LOG,sizeof(log),log,nullptr);
            throw std::runtime_error(std::string("OpenCL kernel build failed: ")+log);
        }
        ew=cl.kernel(prog,"ew",&rc);ck(rc,"clCreateKernel ew");
        relu=cl.kernel(prog,"relu",&rc);ck(rc,"clCreateKernel relu");
        conv=cl.kernel(prog,"conv",&rc);ck(rc,"clCreateKernel conv");
        convTiled=cl.kernel(prog,"conv_tiled",&rc);ck(rc,"clCreateKernel conv_tiled");
        pool=cl.kernel(prog,"pool",&rc);ck(rc,"clCreateKernel pool");
        cat=cl.kernel(prog,"cat",&rc);ck(rc,"clCreateKernel cat");
        // ONNX packet Conv filters are OHWI (out,kh,kw,in). GPU work-items
        // vary output channel fastest, so transpose each filter once to HWIO
        // (kh,kw,in,out) for contiguous coalesced weight loads across GPU lanes.
        std::vector<bool> transposed(packet.operands.size(),false);
        for(const auto& op:packet.ops) {
            if(op.type!=3)continue;
            const u32 id=op.ins.at(1);
            if(transposed[id])continue;
            auto& weight=packet.operands[id];
            if(weight.dims.size()!=4||weight.type!=FP16)
                throw std::runtime_error("Unexpected OpenCL filter tensor");
            size_t oc=weight.dims[0],kh=weight.dims[1],kw=weight.dims[2],ic=weight.dims[3];
            if(weight.raw.size()!=oc*kh*kw*ic*2)
                throw std::runtime_error("Bad GPU filter byte size");
            std::vector<char> hwio(weight.raw.size());
            for(size_t c=0;c<oc;c++)for(size_t y=0;y<kh;y++)
                for(size_t x=0;x<kw;x++)for(size_t i=0;i<ic;i++) {
                    size_t a=(((c*kh+y)*kw+x)*ic+i)*2;
                    size_t b=(((y*kw+x)*ic+i)*oc+c)*2;
                    std::memcpy(hwio.data()+b,weight.raw.data()+a,2);
                }
            weight.raw.swap(hwio);
            transposed[id]=true;
        }
        std::printf("GPU_CONV_LAYOUT format=HWIO coalesced_weights=yes\n");
        for(auto& o:packet.operands) {
            if(o.type!=FP16)continue;
            if((o.flags&(CONST|INPUT))&&o.raw.size()!=o.count*2)
                throw std::runtime_error("Bad FP16 operand size");
            cl_mem_flags flags=MEM_READ_WRITE;
            void* host=nullptr;
            if(!o.raw.empty()) {flags|=MEM_COPY_HOST_PTR;host=o.raw.data();}
            o.buffer=cl.buffer(ctx,flags,o.count*2,host,&rc);ck(rc,"clCreateBuffer");
        }
        std::printf("GPU_GRAPH ops=%zu operands=%zu outputs=%zu\n",packet.ops.size(),packet.operands.size(),packet.outputs.size());
    }
    ~Runner() {
        for(auto& o:packet.operands)if(o.buffer)cl.freeBuffer(o.buffer);
        for(auto k:{ew,relu,conv,convTiled,pool,cat})if(k)cl.freeKernel(k);
        if(prog)cl.freeProgram(prog);
        if(q)cl.freeQueue(q);
        if(ctx)cl.freeContext(ctx);
        if(cl.lib)dlclose(cl.lib);
    }
    std::vector<std::vector<float>> eval() {
        for(const auto& op:packet.ops)operation(op);
        std::vector<std::vector<float>> outputs;
        for(u32 id:packet.outputs){
            const Operand& operand=packet.operands[id];
            std::vector<uint16_t> half(operand.count);
            ck(cl.read(q,operand.buffer,1,0,half.size()*2,half.data(),0,nullptr,nullptr),"clEnqueueReadBuffer");
            std::vector<float> out(half.size());
            for(size_t i=0;i<half.size();i++)out[i]=halfFloat(half[i]);
            outputs.push_back(std::move(out));
        }
        return outputs;
    }
    bool verify(const std::vector<std::vector<float>>& outputs)const {
        bool all=true;
        for(size_t i=0;i<outputs.size();i++){
            const auto& reference=packet.golden.at(i);
            if(outputs[i].size()!=reference.values.size())throw std::runtime_error("Incorrect GPU output dimension");
            size_t wrong=0;double maxerr=0,mean=0;
            for(size_t j=0;j<outputs[i].size();j++){
                float got=outputs[i][j],want=reference.values[j];
                double diff=std::abs(double(got)-double(want));
                if(!std::isfinite(got)||diff>0.05+0.06*std::abs(want))wrong++;
                maxerr=std::max(maxerr,diff);mean+=diff;
            }
            std::printf("GPU_VERIFY output=%s wrong=%zu/%zu max_abs=%.6f mean_abs=%.6f\n",
                        reference.name.c_str(),wrong,outputs[i].size(),maxerr,mean/outputs[i].size());
            if(wrong)all=false;
        }
        return all;
    }
};
double pct(const std::vector<double>& v,double p){
    double pos=(v.size()-1)*p;size_t a=size_t(pos),b=std::min(a+1,v.size()-1);
    return v[a]+(v[b]-v[a])*(pos-a);
}
int main(int argc,char** argv) {
    try{
        if(argc<2)throw std::runtime_error("Usage: gpu_katago_full_android PACKET.bin [warmup=3] [runs=20]");
        int warm=argc>2?std::stoi(argv[2]):3,run=argc>3?std::stoi(argv[3]):20;
        if(warm<0||run<1||run>10000)throw std::runtime_error("Invalid warmup/runs");
        Runner bench(argv[1]);
        for(int i=0;i<warm;i++)bench.eval();
        auto first=bench.eval();
        if(!bench.verify(first))throw std::runtime_error("GPU full network accuracy verification FAILED");
        std::vector<double> elapsed;
        for(int i=0;i<run;i++) {
            auto before=Clock::now();
            bench.eval();
            elapsed.push_back(std::chrono::duration<double,std::milli>(Clock::now()-before).count());
        }
        double mean=std::accumulate(elapsed.begin(),elapsed.end(),0.0)/elapsed.size();
        std::sort(elapsed.begin(),elapsed.end());
        std::printf("GPU_FULL_NN_PASS backend=opencl dtype=fp16 mean=%.3f ms p50=%.3f ms p95=%.3f ms nn_evals_per_sec=%.2f runs=%d\n",
                    mean,pct(elapsed,.5),pct(elapsed,.95),1000.0/mean,run);
    }catch(const std::exception& e){std::fprintf(stderr,"GPU_FULL_NN_FAIL: %s\n",e.what());return 1;}
}
