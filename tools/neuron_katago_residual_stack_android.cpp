// Execute real KataGo 10b rconv1..rconvN (N=1..4) as ONE FP16 Neuron graph.
// Weights and FP32 CPU golden from export_katago_residual_stack.py.
// Full-board InputMask is all ones: multiplication by mask is identity.
// NOT a complete KataGo NN eval or MCTS throughput test.
#include <dlfcn.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

struct NeuronModel;
struct NeuronCompilation;
struct NeuronExecution;
struct NeuronDevice;
struct NeuronOperandType { int32_t type; uint32_t dimensionCount; const uint32_t* dimensions; float scale; int32_t zeroPoint; };
static constexpr int TENSOR_FP16=8, INT32=1, ADD=0, CONV=3, MUL=18;

template<class F> F sym(void* handle,const char* name) {
    void* p=dlsym(handle,name);
    if(!p) throw std::runtime_error(std::string("Missing Neuron symbol: ")+name);
    return reinterpret_cast<F>(p);
}
static void check(int code,const char* name) {
    if(code) throw std::runtime_error(std::string(name)+" rc="+std::to_string(code));
}
static double pct(const std::vector<double>& v,double q) {
    double x=(v.size()-1)*q;
    size_t a=size_t(x),b=std::min(a+1,v.size()-1);
    return v[a]+(v[b]-v[a])*(x-a);
}
struct Block {
    std::vector<uint16_t> scale1,bias1,weight1,scale2,bias2,weight2;
};
struct Packet {
    uint32_t h,w,c,k,n;
    std::vector<uint16_t> input;
    std::vector<Block> blocks;
    std::vector<float> golden;
};
static Packet readPacket(const char* path) {
    std::ifstream f(path,std::ios::binary);
    if(!f)throw std::runtime_error(std::string("Cannot open model packet: ")+path);
    char magic[8];f.read(magic,8);
    if(!f || std::memcmp(magic,"K10RS001",8))throw std::runtime_error("Invalid stacked residual packet magic");
    uint32_t dims[5];f.read(reinterpret_cast<char*>(dims),sizeof(dims));
    if(!f)throw std::runtime_error("Truncated packet header");
    Packet p{};
    p.h=dims[0];p.w=dims[1];p.c=dims[2];p.k=dims[3];p.n=dims[4];
    if(p.h!=19||p.w!=19||p.c!=128||p.k!=3||p.n<1||p.n>4)
        throw std::runtime_error("Expected 19x19/128ch/3x3 and 1..4 residual blocks");
    const size_t volume=size_t(p.h)*p.w*p.c;
    const size_t weights=size_t(p.c)*p.k*p.k*p.c;
    auto read=[&](void* dst,size_t bytes){
        f.read(reinterpret_cast<char*>(dst),bytes);
        if(!f)throw std::runtime_error("Packet data truncated");
    };
    p.input.resize(volume);read(p.input.data(),volume*sizeof(uint16_t));
    p.blocks.resize(p.n);
    for(Block& b:p.blocks) {
        b.scale1.resize(p.c);b.bias1.resize(p.c);b.weight1.resize(weights);
        b.scale2.resize(p.c);b.bias2.resize(p.c);b.weight2.resize(weights);
        read(b.scale1.data(),b.scale1.size()*2);read(b.bias1.data(),b.bias1.size()*2);
        read(b.weight1.data(),b.weight1.size()*2);
        read(b.scale2.data(),b.scale2.size()*2);read(b.bias2.data(),b.bias2.size()*2);
        read(b.weight2.data(),b.weight2.size()*2);
    }
    p.golden.resize(volume);read(p.golden.data(),volume*sizeof(float));
    char extra;
    if(f.read(&extra,1))throw std::runtime_error("Unexpected trailing packet bytes");
    return p;
}

static int run(void* lib,const Packet& p,const std::string& deviceName,int warmup,int runs) {
    auto deviceCount=sym<int (*)(uint32_t*)>(lib,"Neuron_getDeviceCount");
    auto getDevice=sym<int (*)(uint32_t,NeuronDevice**)>(lib,"Neuron_getDevice");
    auto getDeviceName=sym<int (*)(const NeuronDevice*,const char**)>(lib,"NeuronDevice_getName");
    auto createModel=sym<int (*)(NeuronModel**)>(lib,"NeuronModel_create");
    auto freeModel=sym<void (*)(NeuronModel*)>(lib,"NeuronModel_free");
    auto addOperand=sym<int (*)(NeuronModel*,const NeuronOperandType*)>(lib,"NeuronModel_addOperand");
    auto setValue=sym<int (*)(NeuronModel*,int32_t,const void*,size_t)>(lib,"NeuronModel_setOperandValue");
    auto addOp=sym<int (*)(NeuronModel*,int32_t,uint32_t,const uint32_t*,uint32_t,const uint32_t*)>(lib,"NeuronModel_addOperation");
    auto identifyIO=sym<int (*)(NeuronModel*,uint32_t,const uint32_t*,uint32_t,const uint32_t*)>(lib,"NeuronModel_identifyInputsAndOutputs");
    auto finishModel=sym<int (*)(NeuronModel*)>(lib,"NeuronModel_finish");
    auto supported=sym<int (*)(const NeuronModel*,const NeuronDevice* const*,uint32_t,bool*)>(lib,"NeuronModel_getSupportedOperationsForDevices");
    auto createComp=sym<int (*)(NeuronModel*,const NeuronDevice* const*,uint32_t,NeuronCompilation**)>(lib,"NeuronCompilation_createForDevices");
    auto finishComp=sym<int (*)(NeuronCompilation*)>(lib,"NeuronCompilation_finish");
    auto freeComp=sym<void (*)(NeuronCompilation*)>(lib,"NeuronCompilation_free");
    auto createExec=sym<int (*)(NeuronCompilation*,NeuronExecution**)>(lib,"NeuronExecution_create");
    auto setInput=sym<int (*)(NeuronExecution*,int32_t,const NeuronOperandType*,const void*,size_t)>(lib,"NeuronExecution_setInput");
    auto setOutput=sym<int (*)(NeuronExecution*,int32_t,const NeuronOperandType*,void*,size_t)>(lib,"NeuronExecution_setOutput");
    auto compute=sym<int (*)(NeuronExecution*)>(lib,"NeuronExecution_compute");
    auto freeExec=sym<void (*)(NeuronExecution*)>(lib,"NeuronExecution_free");

    uint32_t n=0;check(deviceCount(&n),"Neuron_getDeviceCount");
    if(n>64)throw std::runtime_error("Device count unrealistic");
    NeuronDevice* target=nullptr;
    for(uint32_t i=0;i<n;++i) {
        NeuronDevice* d=nullptr;const char* name=nullptr;
        check(getDevice(i,&d),"Neuron_getDevice");
        check(getDeviceName(d,&name),"NeuronDevice_getName");
        std::printf("Neuron device[%u]: %s\n",i,name?name:"(unknown)");
        if(name&&deviceName==name)target=d;
    }
    if(!target)throw std::runtime_error("Requested Neuron device not found: "+deviceName);
    const NeuronDevice* devices[]={target};

    NeuronModel* model=nullptr;NeuronCompilation* comp=nullptr;NeuronExecution* exec=nullptr;
    try {
        check(createModel(&model),"NeuronModel_create");
        uint32_t fullDims[4]={1,p.h,p.w,p.c};
        uint32_t chanDims[4]={1,1,1,p.c};
        uint32_t weightDims[4]={p.c,p.k,p.k,p.c};
        uint32_t biasDims[1]={p.c};
        NeuronOperandType full={TENSOR_FP16,4,fullDims,0,0};
        NeuronOperandType chan={TENSOR_FP16,4,chanDims,0,0};
        NeuronOperandType filter={TENSOR_FP16,4,weightDims,0,0};
        NeuronOperandType bias={TENSOR_FP16,1,biasDims,0,0};
        NeuronOperandType scalar={INT32,0,nullptr,0,0};

        // Global: 0 input, 1 zero convolution bias, 2 activation NONE,
        // 3 activation RELU, 4..7 padding L/R/T/B, 8..9 strides W/H.
        check(addOperand(model,&full),"input");
        check(addOperand(model,&bias),"shared zero Conv bias");
        for(int i=2;i<10;i++)check(addOperand(model,&scalar),"shared Conv scalar");
        std::vector<uint16_t> zeros(p.c,0);
        check(setValue(model,1,zeros.data(),zeros.size()*2),"Conv zero bias");
        const int32_t flags[8]={0,1,1,1,1,1,1,1};
        for(int i=0;i<8;++i)check(setValue(model,2+i,&flags[i],4),"Conv scalar value");

        // Every block: six original constants followed by seven FP16 output tensors.
        // One graph connects each block's skip input to the previous block's output.
        for(uint32_t b=0;b<p.n;++b) {
            const uint32_t base=10+b*13;
            const Block& params=p.blocks[b];
            for(int k=0;k<6;k++){
                const NeuronOperandType* type=(k==2||k==5)?&filter:&chan;
                check(addOperand(model,type),"block constant");
            }
            for(int k=0;k<7;k++)check(addOperand(model,&full),"block output");
            const std::vector<uint16_t>* tensors[6]={&params.scale1,&params.bias1,&params.weight1,
                                                      &params.scale2,&params.bias2,&params.weight2};
            for(int k=0;k<6;k++)check(setValue(model,base+k,tensors[k]->data(),tensors[k]->size()*2),"block weight/norm");
        }
        auto op=[&](int type,std::initializer_list<uint32_t> in,uint32_t out,const char* name){
            std::vector<uint32_t> ins(in);
            check(addOp(model,type,ins.size(),ins.data(),1,&out),name);
        };
        const char* operationNames[7]={"norm1_mul","norm1_bias_relu","conv_w1",
                                       "norm2_mul","norm2_bias_relu","conv_w2","residual_add"};
        for(uint32_t b=0;b<p.n;++b) {
            uint32_t base=10+b*13;
            uint32_t input=b?10+(b-1)*13+12:0;
            op(MUL,{input,base,2},base+6,"norm1.mul");
            op(ADD,{base+6,base+1,3},base+7,"norm1.add.relu");
            op(CONV,{base+7,base+2,1,4,5,6,7,8,9,2},base+8,"conv.w1");
            op(MUL,{base+8,base+3,2},base+9,"norm2.mul");
            op(ADD,{base+9,base+4,3},base+10,"norm2.add.relu");
            op(CONV,{base+10,base+5,1,4,5,6,7,8,9,2},base+11,"conv.w2");
            op(ADD,{input,base+11,2},base+12,"residual.add");
        }
        const uint32_t inputIdx[]={0};
        const uint32_t outputIdx[]={10+(p.n-1)*13+12};
        check(identifyIO(model,1,inputIdx,1,outputIdx),"NeuronModel_identifyInputsAndOutputs");
        check(finishModel(model),"NeuronModel_finish");

        bool flagsSupport[28]={false};
        check(supported(model,devices,1,flagsSupport),"NeuronModel_getSupportedOperationsForDevices");
        bool all=true;
        for(uint32_t b=0;b<p.n;++b){
            for(int i=0;i<7;++i){
                bool ok=flagsSupport[b*7+i];
                std::printf("KATAGO_STACK_OP_SUPPORT block=%u op=%s supported=%s\n",
                            b+1,operationNames[i],ok?"yes":"no");
                if(!ok)all=false;
            }
        }
        std::fflush(stdout);
        if(!all)throw std::runtime_error("At least one stacked residual operator unsupported on device");
        auto start=std::chrono::steady_clock::now();
        check(createComp(model,devices,1,&comp),"NeuronCompilation_createForDevices");
        check(finishComp(comp),"NeuronCompilation_finish");
        double compileMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        std::printf("KATAGO_STACK_COMPILE device=%s blocks=%u ops=%u compile_ms=%.3f\n",
                    deviceName.c_str(),p.n,p.n*7,compileMs);
        std::fflush(stdout);
        check(createExec(comp,&exec),"NeuronExecution_create");
        std::vector<uint16_t> actual(p.input.size(),0);
        check(setInput(exec,0,nullptr,p.input.data(),p.input.size()*2),"NeuronExecution_setInput");
        check(setOutput(exec,0,nullptr,actual.data(),actual.size()*2),"NeuronExecution_setOutput");
        std::vector<double> latency;
        latency.reserve(runs);
        for(int i=0;i<warmup+runs;++i){
            auto t=std::chrono::steady_clock::now();
            check(compute(exec),"NeuronExecution_compute");
            if(i>=warmup)latency.push_back(std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-t).count());
        }
        size_t bad=0,worst=0;
        double maxError=0,meanError=0;
        float worstResult=0,worstRef=0;
        for(size_t i=0;i<actual.size();++i){
            _Float16 half;
            std::memcpy(&half,&actual[i],sizeof(half));
            float got=float(half),ref=p.golden[i];
            double err=std::fabs(double(got)-ref);
            if(!std::isfinite(got)||!std::isfinite(ref))err=INFINITY;
            if(err>maxError){maxError=err;worst=i;worstResult=got;worstRef=ref;}
            meanError+=err;
            if(err>0.03+0.03*std::fabs(double(ref)))bad++;
        }
        meanError/=actual.size();
        std::printf("KATAGO_STACK_VERIFY device=%s blocks=%u wrong=%zu/%zu max_abs=%.6f mean_abs=%.6f worst_idx=%zu mdla=%.6f onnx=%.6f\n",
                    deviceName.c_str(),p.n,bad,actual.size(),maxError,meanError,worst,worstResult,worstRef);
        if(bad)throw std::runtime_error("Stacked Neuron FP16 result differs from ONNX CPU golden");
        std::sort(latency.begin(),latency.end());
        double mean=std::accumulate(latency.begin(),latency.end(),0.0)/latency.size();
        std::printf("KATAGO_STACK_PASS device=%s blocks=%u ops=%u mean=%.3f ms p50=%.3f ms p95=%.3f ms stacks/s=%.2f\n",
                    deviceName.c_str(),p.n,p.n*7,mean,pct(latency,0.5),pct(latency,0.95),1000.0/mean);
        std::puts("NOTE: 1..4 real residual blocks only. NOT a KataGo 10b NN eval/s or MCTS visits/s.");
    }catch(...){
        if(exec)freeExec(exec);
        if(comp)freeComp(comp);
        if(model)freeModel(model);
        throw;
    }
    freeExec(exec);freeComp(comp);freeModel(model);
    return 0;
}
int main(int argc,char** argv) {
    try{
        if(argc<2||argc>5){
            std::fprintf(stderr,"Usage: %s packet.bin [mtk-mdla] [warmup] [runs]\n",argv[0]);
            return 2;
        }
        Packet p=readPacket(argv[1]);
        std::string device=argc>2?argv[2]:"mtk-mdla";
        int warmup=argc>3?std::stoi(argv[3]):20;
        int runs=argc>4?std::stoi(argv[4]):100;
        if(warmup<0||runs<1||runs>100000)throw std::runtime_error("Invalid iteration count");
        void* lib=dlopen("libneuronusdk_adapter.9.mtk.so",RTLD_NOW|RTLD_LOCAL);
        if(!lib)lib=dlopen("libneuronusdk_adapter.mtk.so",RTLD_NOW|RTLD_LOCAL);
        if(!lib)throw std::runtime_error("Could not load Neuron Adapter");
        std::printf("KATAGO_STACK_GRAPH blocks=%u ops=%u device=%s shape=1x%ux%ux%u\n",
                    p.n,p.n*7,device.c_str(),p.h,p.w,p.c);
        int code=run(lib,p,device,warmup,runs);
        dlclose(lib);
        return code;
    }catch(const std::exception& e){
        std::fprintf(stderr,"KATAGO_STACK_FAIL: %s\n",e.what());
        return 1;
    }
}
