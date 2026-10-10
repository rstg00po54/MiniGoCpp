// True KataGo 10b rconv1 full residual path on one MediaTek Neuron device.
// FP16: normalization Mul/Add+ReLU -> Conv -> norm Mul/Add+ReLU -> Conv -> residual Add.
// Source packet: export_katago_rconv1_block.py, with real 10b weights and ONNX CPU golden.
// Board mask is all ones: two mask-multiplication ops are mathematically identity.
// This is a SINGLE residual block, NOT full KataGo neural network inference.
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
struct NeuronOperandType {int32_t type;uint32_t dimensionCount;const uint32_t* dimensions;float scale;int32_t zeroPoint;};
static constexpr int32_t TENSOR_FP16=8;
static constexpr int32_t INT32=1;
static constexpr int32_t ADD=0;
static constexpr int32_t CONV=3;
static constexpr int32_t MUL=18;

template<class F> F sym(void* lib,const char* n){
    void* p=dlsym(lib,n);
    if(!p)throw std::runtime_error(std::string("Missing Neuron API symbol: ")+n);
    return reinterpret_cast<F>(p);
}
static void check(int rc,const char* what){
    if(rc)throw std::runtime_error(std::string(what)+" rc="+std::to_string(rc));
}
static double pct(const std::vector<double>& v,double q){
    double position=(v.size()-1)*q;
    size_t a=static_cast<size_t>(position),b=std::min(a+1,v.size()-1);
    return v[a]+(v[b]-v[a])*(position-a);
}

struct Packet {
    uint32_t h,w,c,k;
    std::vector<uint16_t> input,scale1,bias1,weight1,scale2,bias2,weight2;
    std::vector<float> golden;
};

static Packet readPacket(const char* path) {
    std::ifstream f(path,std::ios::binary);
    if(!f)throw std::runtime_error(std::string("Cannot open rconv1 packet: ")+path);
    char magic[8]={0};
    f.read(magic,8);
    if(std::memcmp(magic,"K10RB001",8))throw std::runtime_error("Not a KataGo rconv1 block packet");
    std::array<uint32_t,4> dims{};
    f.read(reinterpret_cast<char*>(dims.data()),sizeof(dims));
    if(!f)throw std::runtime_error("Truncated packet dimensions");
    Packet p{};
    p.h=dims[0];p.w=dims[1];p.c=dims[2];p.k=dims[3];
    if(p.h!=19 || p.w!=19 || p.c!=128 || p.k!=3)
        throw std::runtime_error("Expected KataGo 10b 19x19 rconv1 128x128 3x3 packet");
    const size_t volume=size_t(p.h)*p.w*p.c;
    const size_t filter=size_t(p.c)*p.k*p.k*p.c;
    p.input.resize(volume);
    p.scale1.resize(p.c);p.bias1.resize(p.c);p.weight1.resize(filter);
    p.scale2.resize(p.c);p.bias2.resize(p.c);p.weight2.resize(filter);
    p.golden.resize(volume);
    auto take=[&](void* data,size_t size){f.read(reinterpret_cast<char*>(data),size);
        if(!f)throw std::runtime_error("Truncated rconv1 packet");};
    take(p.input.data(),volume*2);
    take(p.scale1.data(),p.c*2);take(p.bias1.data(),p.c*2);
    take(p.weight1.data(),filter*2);
    take(p.scale2.data(),p.c*2);take(p.bias2.data(),p.c*2);
    take(p.weight2.data(),filter*2);
    take(p.golden.data(),volume*4);
    char extra=0;
    if(f.read(&extra,1))throw std::runtime_error("Extra bytes in rconv1 packet");
    return p;
}

static int execute(void* lib,const Packet& p,const std::string& deviceName,int warmup,int runs) {
    auto deviceCount=sym<int (*)(uint32_t*)>(lib,"Neuron_getDeviceCount");
    auto getDevice=sym<int (*)(uint32_t,NeuronDevice**)>(lib,"Neuron_getDevice");
    auto getDeviceName=sym<int (*)(const NeuronDevice*,const char**)>(lib,"NeuronDevice_getName");
    auto modelCreate=sym<int (*)(NeuronModel**)>(lib,"NeuronModel_create");
    auto modelFree=sym<void (*)(NeuronModel*)>(lib,"NeuronModel_free");
    auto modelAdd=sym<int (*)(NeuronModel*,const NeuronOperandType*)>(lib,"NeuronModel_addOperand");
    auto modelValue=sym<int (*)(NeuronModel*,int32_t,const void*,size_t)>(lib,"NeuronModel_setOperandValue");
    auto modelOperation=sym<int (*)(NeuronModel*,int32_t,uint32_t,const uint32_t*,uint32_t,const uint32_t*)>(lib,"NeuronModel_addOperation");
    auto modelIO=sym<int (*)(NeuronModel*,uint32_t,const uint32_t*,uint32_t,const uint32_t*)>(lib,"NeuronModel_identifyInputsAndOutputs");
    auto modelFinish=sym<int (*)(NeuronModel*)>(lib,"NeuronModel_finish");
    auto querySupport=sym<int (*)(const NeuronModel*,const NeuronDevice* const*,uint32_t,bool*)>(lib,"NeuronModel_getSupportedOperationsForDevices");
    auto compCreate=sym<int (*)(NeuronModel*,const NeuronDevice* const*,uint32_t,NeuronCompilation**)>(lib,"NeuronCompilation_createForDevices");
    auto compFinish=sym<int (*)(NeuronCompilation*)>(lib,"NeuronCompilation_finish");
    auto compFree=sym<void (*)(NeuronCompilation*)>(lib,"NeuronCompilation_free");
    auto execCreate=sym<int (*)(NeuronCompilation*,NeuronExecution**)>(lib,"NeuronExecution_create");
    auto execInput=sym<int (*)(NeuronExecution*,int32_t,const NeuronOperandType*,const void*,size_t)>(lib,"NeuronExecution_setInput");
    auto execOutput=sym<int (*)(NeuronExecution*,int32_t,const NeuronOperandType*,void*,size_t)>(lib,"NeuronExecution_setOutput");
    auto execCompute=sym<int (*)(NeuronExecution*)>(lib,"NeuronExecution_compute");
    auto execFree=sym<void (*)(NeuronExecution*)>(lib,"NeuronExecution_free");

    uint32_t n=0;check(deviceCount(&n),"Neuron_getDeviceCount");
    if(n>64)throw std::runtime_error("Unreasonable Neuron device count");
    NeuronDevice* target=nullptr;
    for(uint32_t i=0;i<n;i++){
        NeuronDevice* d=nullptr;const char* name=nullptr;
        check(getDevice(i,&d),"Neuron_getDevice");
        check(getDeviceName(d,&name),"NeuronDevice_getName");
        std::printf("Neuron device[%u]: %s\n",i,name?name:"(unknown)");
        if(name&&deviceName==name)target=d;
    }
    if(!target)throw std::runtime_error("Neuron device not found: "+deviceName);
    const NeuronDevice* devices[]={target};

    NeuronModel* model=nullptr;NeuronCompilation* comp=nullptr;NeuronExecution* exec=nullptr;
    try {
        check(modelCreate(&model),"NeuronModel_create");
        const uint32_t fullDims[4]={1,p.h,p.w,p.c};
        const uint32_t channelDims[4]={1,1,1,p.c};
        const uint32_t filterDims[4]={p.c,p.k,p.k,p.c};
        const uint32_t biasDims[1]={p.c};
        const NeuronOperandType fullType={TENSOR_FP16,4,fullDims,0.0f,0};
        const NeuronOperandType channelType={TENSOR_FP16,4,channelDims,0.0f,0};
        const NeuronOperandType filterType={TENSOR_FP16,4,filterDims,0.0f,0};
        const NeuronOperandType biasType={TENSOR_FP16,1,biasDims,0.0f,0};
        const NeuronOperandType scalarType={INT32,0,nullptr,0.0f,0};

        // 0 input, 1 scale1, 2 bias1, 3 W1, 4 scale2, 5 bias2, 6 W2,
        // 7 shared Conv bias=0, 8 activation NONE, 9 activation RELU,
        // 10..13 pad L/R/T/B, 14/15 strideW/H, 16..22 intermediate/output.
        check(modelAdd(model,&fullType),"add input");
        check(modelAdd(model,&channelType),"add norm1.scale");
        check(modelAdd(model,&channelType),"add norm1.bias");
        check(modelAdd(model,&filterType),"add rconv1/w1");
        check(modelAdd(model,&channelType),"add norm2.scale");
        check(modelAdd(model,&channelType),"add norm2.bias");
        check(modelAdd(model,&filterType),"add rconv1/w2");
        check(modelAdd(model,&biasType),"add zero Conv bias");
        for(int i=8;i<=15;i++)check(modelAdd(model,&scalarType),"add int scalar");
        for(int i=16;i<=22;i++)check(modelAdd(model,&fullType),"add activation tensor");

        check(modelValue(model,1,p.scale1.data(),p.scale1.size()*2),"norm1.scale");
        check(modelValue(model,2,p.bias1.data(),p.bias1.size()*2),"norm1.bias");
        check(modelValue(model,3,p.weight1.data(),p.weight1.size()*2),"rconv1/w1");
        check(modelValue(model,4,p.scale2.data(),p.scale2.size()*2),"norm2.scale");
        check(modelValue(model,5,p.bias2.data(),p.bias2.size()*2),"norm2.bias");
        check(modelValue(model,6,p.weight2.data(),p.weight2.size()*2),"rconv1/w2");
        std::vector<uint16_t> zeroConvBias(p.c,0);
        check(modelValue(model,7,zeroConvBias.data(),zeroConvBias.size()*2),"shared zero Conv bias");
        const int32_t integers[8]={0,1,1,1,1,1,1,1};
        for(int i=0;i<8;i++)check(modelValue(model,8+i,&integers[i],sizeof(int32_t)),"set Conv/ADD scalars");

        auto operation=[&](int type,std::initializer_list<uint32_t> in,uint32_t out,const char* label){
            std::vector<uint32_t> inputs(in);
            check(modelOperation(model,type,inputs.size(),inputs.data(),1,&out),label);
        };
        operation(MUL,{0,1,8},16,"norm1 Mul");
        operation(ADD,{16,2,9},17,"norm1 Add fused ReLU");
        operation(CONV,{17,3,7,10,11,12,13,14,15,8},18,"rconv1/w1 Conv");
        operation(MUL,{18,4,8},19,"norm2 Mul");
        operation(ADD,{19,5,9},20,"norm2 Add fused ReLU");
        operation(CONV,{20,6,7,10,11,12,13,14,15,8},21,"rconv1/w2 Conv");
        operation(ADD,{0,21,8},22,"rconv1 residual Add");
        const uint32_t inputIndex[]={0},outputIndex[]={22};
        check(modelIO(model,1,inputIndex,1,outputIndex),"NeuronModel_identifyInputsAndOutputs");
        check(modelFinish(model),"NeuronModel_finish");

        bool supported[7]={false,false,false,false,false,false,false};
        const char* names[7]={"norm1_mul","norm1_bias_relu","conv_w1","norm2_mul","norm2_bias_relu","conv_w2","residual_add"};
        check(querySupport(model,devices,1,supported),"NeuronModel_getSupportedOperationsForDevices");
        bool all=true;
        for(int i=0;i<7;i++){
            std::printf("KATAGO_RCONV1_OP_SUPPORT device=%s op=%s supported=%s\n",
                        deviceName.c_str(),names[i],supported[i]?"yes":"no");
            if(!supported[i])all=false;
        }
        std::fflush(stdout);
        if(!all)throw std::runtime_error("One or more full-block operations unsupported by requested MDLA device");

        auto begin=std::chrono::steady_clock::now();
        check(compCreate(model,devices,1,&comp),"NeuronCompilation_createForDevices");
        check(compFinish(comp),"NeuronCompilation_finish");
        double compileMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        std::printf("KATAGO_RCONV1_COMPILE device=%s ops=7 fused_relu=yes ms=%.3f\n",
                    deviceName.c_str(),compileMs);
        std::fflush(stdout);
        check(execCreate(comp,&exec),"NeuronExecution_create");
        std::vector<uint16_t> actual(p.input.size(),0);
        check(execInput(exec,0,nullptr,p.input.data(),p.input.size()*2),"NeuronExecution_setInput");
        check(execOutput(exec,0,nullptr,actual.data(),actual.size()*2),"NeuronExecution_setOutput");

        std::vector<double> timings;
        timings.reserve(runs);
        for(int i=0;i<warmup+runs;i++){
            auto t=std::chrono::steady_clock::now();
            check(execCompute(exec),"NeuronExecution_compute");
            if(i>=warmup)timings.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());
        }
        size_t wrong=0,worst=0;
        double maxError=0,sumError=0;
        float worstA=0,worstR=0;
        for(size_t i=0;i<actual.size();i++){
            _Float16 value;
            std::memcpy(&value,&actual[i],sizeof(value));
            const float output=static_cast<float>(value),ref=p.golden[i];
            double error=std::fabs(double(output)-ref);
            if(!std::isfinite(output)||!std::isfinite(ref))error=INFINITY;
            if(error>maxError){maxError=error;worst=i;worstA=output;worstR=ref;}
            sumError+=error;
            if(error>0.03+0.03*std::fabs(double(ref)))wrong++;
        }
        std::printf("KATAGO_RCONV1_VERIFY device=%s wrong=%zu/%zu max_abs=%.6f mean_abs=%.6f worst_idx=%zu mdla=%.6f onnx=%.6f\n",
                    deviceName.c_str(),wrong,actual.size(),maxError,sumError/actual.size(),worst,worstA,worstR);
        if(wrong)throw std::runtime_error("KataGo rconv1 full-block numeric mismatch against ONNX CPU");
        std::sort(timings.begin(),timings.end());
        const double mean=std::accumulate(timings.begin(),timings.end(),0.0)/timings.size();
        std::printf("KATAGO_RCONV1_PASS device=%s ops=7 mean=%.3f ms p50=%.3f ms p95=%.3f ms blocks/s=%.2f\n",
                    deviceName.c_str(),mean,pct(timings,.50),pct(timings,.95),1000.0/mean);
        std::puts("NOTE: KataGo rconv1 block only, NOT a complete NN evaluation or game-search visits/s.");
    }catch(...){
        if(exec)execFree(exec);
        if(comp)compFree(comp);
        if(model)modelFree(model);
        throw;
    }
    execFree(exec);compFree(comp);modelFree(model);
    return 0;
}

int main(int argc,char** argv){
    try{
        if(argc<2||argc>5){
            std::fprintf(stderr,"Usage: %s katago_rconv1.bin [mtk-mdla] [warmup=20] [runs=100]\n",argv[0]);
            return 2;
        }
        const Packet packet=readPacket(argv[1]);
        const std::string device=argc>2?argv[2]:"mtk-mdla";
        const int warmup=argc>3?std::stoi(argv[3]):20;
        const int runs=argc>4?std::stoi(argv[4]):100;
        if(warmup<0||runs<1||runs>100000)throw std::runtime_error("Invalid iteration counts");
        void* lib=dlopen("libneuronusdk_adapter.9.mtk.so",RTLD_NOW|RTLD_LOCAL);
        if(!lib)lib=dlopen("libneuronusdk_adapter.mtk.so",RTLD_NOW|RTLD_LOCAL);
        if(!lib)throw std::runtime_error("Failed to load MediaTek Neuron Adapter 9");
        std::printf("KATAGO_RCONV1_GRAPH: board=%ux%u channels=%u kernel=%ux%u device=%s\n",
                    packet.h,packet.w,packet.c,packet.k,packet.k,device.c_str());
        int result=execute(lib,packet,device,warmup,runs);
        dlclose(lib);
        return result;
    }catch(const std::exception& e){
        std::fprintf(stderr,"KATAGO_RCONV1_FAIL: %s\n",e.what());
        return 1;
    }
}
