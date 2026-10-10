// Real KataGo 10b rconv5 global pooling and 1x1 convolution on MTK Neuron.
// ONNX CPU golden and true FP16 feature input/weights exported from 10b model.
// FP16: MEAN -> MUL(mask scale); REDUCE_MAX; CONCAT 3 tensors; CONV 1x1.
// InputMask is all ones (19x19 full board). NOT full KataGo inference.
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

struct NeuronDevice; struct NeuronModel; struct NeuronCompilation; struct NeuronExecution;
struct NeuronOperandType {
    int32_t type; uint32_t dimensionCount; const uint32_t* dimensions;
    float scale; int32_t zeroPoint;
};
static constexpr int32_t FP16=8, INT32=1, BOOL=6, TENSOR_INT32=4;
static constexpr int32_t OP_MEAN=31, OP_MUL=18, OP_MAX=77, OP_CONCAT=2, OP_CONV=3;

template<class F> F sym(void* lib,const char* s) {
    void* p=dlsym(lib,s);
    if(!p)throw std::runtime_error(std::string("Missing Neuron symbol: ")+s);
    return reinterpret_cast<F>(p);
}
static void check(int rc,const char* name) {
    if(rc)throw std::runtime_error(std::string(name)+" rc="+std::to_string(rc));
}
struct Packet {
    uint32_t h,w,ci,co;
    uint16_t scale;
    std::vector<uint16_t> input,weight;
    std::vector<float> golden;
};
static Packet readPacket(const char* path) {
    std::ifstream f(path,std::ios::binary);
    if(!f)throw std::runtime_error(std::string("Cannot open gpool packet: ")+path);
    char magic[8]={};f.read(magic,8);
    if(!f || std::memcmp(magic,"K10GP001",8))throw std::runtime_error("Expected K10GP001 gpool packet");
    std::array<uint32_t,4> dims{};
    f.read(reinterpret_cast<char*>(dims.data()),sizeof(dims));
    Packet p{};p.h=dims[0];p.w=dims[1];p.ci=dims[2];p.co=dims[3];
    if(!f||p.h!=19||p.w!=19||p.ci!=32||p.co!=96)
        throw std::runtime_error("Incorrect rconv5 global pooling dimensions");
    f.read(reinterpret_cast<char*>(&p.scale),2);
    p.input.resize(size_t(p.h)*p.w*p.ci);
    p.weight.resize(size_t(p.co)*p.co);
    p.golden.resize(p.co);
    auto read=[&](void* data,size_t bytes){
        f.read(reinterpret_cast<char*>(data),bytes);
        if(!f)throw std::runtime_error("Truncated global pool data");
    };
    read(p.input.data(),p.input.size()*2);
    read(p.weight.data(),p.weight.size()*2);
    read(p.golden.data(),p.golden.size()*4);
    char extra;
    if(f.read(&extra,1))throw std::runtime_error("Unexpected bytes after gpool packet");
    return p;
}
static double pct(const std::vector<double>& values,double q) {
    double pos=(values.size()-1)*q;
    size_t a=size_t(pos),b=std::min(a+1,values.size()-1);
    return values[a]+(values[b]-values[a])*(pos-a);
}
static int run(void* lib,const Packet& p,const std::string& targetName,int warmup,int reps) {
    auto getCount=sym<int (*)(uint32_t*)>(lib,"Neuron_getDeviceCount");
    auto getDevice=sym<int (*)(uint32_t,NeuronDevice**)>(lib,"Neuron_getDevice");
    auto getName=sym<int (*)(const NeuronDevice*,const char**)>(lib,"NeuronDevice_getName");
    auto modelCreate=sym<int (*)(NeuronModel**)>(lib,"NeuronModel_create");
    auto modelFree=sym<void (*)(NeuronModel*)>(lib,"NeuronModel_free");
    auto modelOperand=sym<int (*)(NeuronModel*,const NeuronOperandType*)>(lib,"NeuronModel_addOperand");
    auto modelValue=sym<int (*)(NeuronModel*,int32_t,const void*,size_t)>(lib,"NeuronModel_setOperandValue");
    auto modelOperation=sym<int (*)(NeuronModel*,int32_t,uint32_t,const uint32_t*,uint32_t,const uint32_t*)>(
        lib,"NeuronModel_addOperation");
    auto modelIO=sym<int (*)(NeuronModel*,uint32_t,const uint32_t*,uint32_t,const uint32_t*)>(
        lib,"NeuronModel_identifyInputsAndOutputs");
    auto modelFinish=sym<int (*)(NeuronModel*)>(lib,"NeuronModel_finish");
    auto opSupport=sym<int (*)(const NeuronModel*,const NeuronDevice* const*,uint32_t,bool*)>(
        lib,"NeuronModel_getSupportedOperationsForDevices");
    auto compCreate=sym<int (*)(NeuronModel*,const NeuronDevice* const*,uint32_t,NeuronCompilation**)>(
        lib,"NeuronCompilation_createForDevices");
    auto compFinish=sym<int (*)(NeuronCompilation*)>(lib,"NeuronCompilation_finish");
    auto compFree=sym<void (*)(NeuronCompilation*)>(lib,"NeuronCompilation_free");
    auto execCreate=sym<int (*)(NeuronCompilation*,NeuronExecution**)>(lib,"NeuronExecution_create");
    auto execInput=sym<int (*)(NeuronExecution*,int32_t,const NeuronOperandType*,const void*,size_t)>(
        lib,"NeuronExecution_setInput");
    auto execOutput=sym<int (*)(NeuronExecution*,int32_t,const NeuronOperandType*,void*,size_t)>(
        lib,"NeuronExecution_setOutput");
    auto execCompute=sym<int (*)(NeuronExecution*)>(lib,"NeuronExecution_compute");
    auto execFree=sym<void (*)(NeuronExecution*)>(lib,"NeuronExecution_free");

    NeuronDevice* device=nullptr;
    uint32_t count=0;check(getCount(&count),"Neuron_getDeviceCount");
    if(count>64)throw std::runtime_error("Invalid Neuron device count");
    for(uint32_t i=0;i<count;i++) {
        NeuronDevice* d=nullptr;const char* name=nullptr;
        check(getDevice(i,&d),"Neuron_getDevice");
        check(getName(d,&name),"NeuronDevice_getName");
        std::printf("Neuron device[%u]: %s\n",i,name?name:"(null)");
        if(name && targetName==name)device=d;
    }
    if(!device)throw std::runtime_error("Neuron device missing: "+targetName);
    const NeuronDevice* devices[]={device};

    NeuronModel* model=nullptr;NeuronCompilation* compilation=nullptr;NeuronExecution* execution=nullptr;
    try {
        check(modelCreate(&model),"NeuronModel_create");
        const uint32_t inDims[4]={1,p.h,p.w,p.ci};
        const uint32_t midDims[4]={1,1,1,p.ci};
        const uint32_t concatDims[4]={1,1,1,p.co};
        const uint32_t scalarDims[4]={1,1,1,1};
        const uint32_t axesDims[1]={2};
        const uint32_t filterDims[4]={p.co,1,1,p.co};
        const uint32_t biasDims[1]={p.co};
        const NeuronOperandType inputType={FP16,4,inDims,0,0};
        const NeuronOperandType pooledType={FP16,4,midDims,0,0};
        const NeuronOperandType concatType={FP16,4,concatDims,0,0};
        const NeuronOperandType floatScaleType={FP16,4,scalarDims,0,0};
        const NeuronOperandType axesType={TENSOR_INT32,1,axesDims,0,0};
        const NeuronOperandType boolType={BOOL,0,nullptr,0,0};
        const NeuronOperandType intType={INT32,0,nullptr,0,0};
        const NeuronOperandType filterType={FP16,4,filterDims,0,0};
        const NeuronOperandType biasType={FP16,1,biasDims,0,0};
        // Operand numbering: 0 input, 1 axes, 2 keep int, 3 keep bool,
        // 4 mean result, 5 GP scale, 6 activation NONE, 7 scaled mean,
        // 8 max, 9 concat axis, 10 concat result, 11 conv weight,
        // 12 zero bias, 13 implicit padding VALID, 14 strideW, 15 strideH,
        // 16 conv output.
        const NeuronOperandType* types[]={&inputType,&axesType,&intType,&boolType,
            &pooledType,&floatScaleType,&intType,&pooledType,
            &pooledType,&intType,&concatType,&filterType,&biasType,
            &intType,&intType,&intType,&concatType};
        for(const auto* type:types)check(modelOperand(model,type),"add global pool operand");
        int32_t axes[2]={1,2},one=1,zero=0,axis=3,valid=2;
        uint8_t keepTrue=1;
        check(modelValue(model,1,axes,sizeof(axes)),"set spatial axes");
        check(modelValue(model,2,&one,sizeof(one)),"set MEAN keepDims");
        check(modelValue(model,3,&keepTrue,sizeof(keepTrue)),"set MAX keepDims");
        check(modelValue(model,5,&p.scale,sizeof(p.scale)),"set actual KataGo mean scale");
        check(modelValue(model,6,&zero,sizeof(zero)),"set activation NONE");
        check(modelValue(model,9,&axis,sizeof(axis)),"set concat channel axis");
        check(modelValue(model,11,p.weight.data(),p.weight.size()*2),"set original w1r weight");
        std::vector<uint16_t> zeros(p.co,0);
        check(modelValue(model,12,zeros.data(),zeros.size()*2),"set convolution zero bias");
        check(modelValue(model,13,&valid,sizeof(valid)),"set convolution VALID padding");
        check(modelValue(model,14,&one,sizeof(one)),"set width stride=1");
        check(modelValue(model,15,&one,sizeof(one)),"set height stride=1");
        auto op=[&](int type,std::initializer_list<uint32_t> input,uint32_t output,const char* name){
            std::vector<uint32_t> args(input);
            check(modelOperation(model,type,args.size(),args.data(),1,&output),name);
        };
        op(OP_MEAN,{0,1,2},4,"MEAN spatial");
        op(OP_MUL,{4,5,6},7,"MUL scaled mean");
        op(OP_MAX,{0,1,3},8,"REDUCE_MAX spatial");
        op(OP_CONCAT,{4,7,8,9},10,"CONCAT 3-way mean/scaled-mean/max");
        op(OP_CONV,{10,11,12,13,14,15,6},16,"1x1 original KataGo w1r Conv");
        const uint32_t inputIndex[]={0},outputIndex[]={16};
        check(modelIO(model,1,inputIndex,1,outputIndex),"identify gpool IO");
        check(modelFinish(model),"NeuronModel_finish");
        bool supported[5]={false,false,false,false,false};
        const char* names[5]={"mean","scaled_mean_mul","reduce_max","concat","w1r_conv1x1"};
        check(opSupport(model,devices,1,supported),"NeuronModel_getSupportedOperationsForDevices");
        bool complete=true;
        for(int i=0;i<5;i++) {
            std::printf("KATAGO_GPOOL_OP_SUPPORT device=%s op=%s supported=%s\n",
                        targetName.c_str(),names[i],supported[i]?"yes":"no");
            if(!supported[i])complete=false;
        }
        std::fflush(stdout);
        if(!complete)throw std::runtime_error("MDLA does not support entire rconv5 gpool branch");
        auto start=std::chrono::steady_clock::now();
        check(compCreate(model,devices,1,&compilation),"NeuronCompilation_createForDevices");
        check(compFinish(compilation),"NeuronCompilation_finish");
        double compileMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        std::printf("KATAGO_GPOOL_COMPILE device=%s ops=5 compile_ms=%.3f\n",targetName.c_str(),compileMs);
        check(execCreate(compilation,&execution),"NeuronExecution_create");
        check(execInput(execution,0,nullptr,p.input.data(),p.input.size()*2),"NeuronExecution_setInput");
        std::vector<uint16_t> actual(p.co,0);
        check(execOutput(execution,0,nullptr,actual.data(),actual.size()*2),"NeuronExecution_setOutput");
        for(int i=0;i<warmup;i++)check(execCompute(execution),"warmup compute");
        std::vector<double> latencies;latencies.reserve(reps);
        for(int i=0;i<reps;i++) {
            auto before=std::chrono::steady_clock::now();
            check(execCompute(execution),"NeuronExecution_compute");
            latencies.push_back(std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-before).count());
        }
        size_t wrong=0;
        double maxAbs=0,sumAbs=0;
        for(size_t i=0;i<actual.size();i++) {
            _Float16 half;std::memcpy(&half,&actual[i],sizeof(half));
            float got=float(half),ref=p.golden[i];
            double error=std::fabs(double(got)-ref);
            if(!std::isfinite(got)||!std::isfinite(ref))error=INFINITY;
            maxAbs=std::max(maxAbs,error);
            sumAbs+=error;
            if(error>0.02+0.03*std::fabs(double(ref)))wrong++;
        }
        std::printf("KATAGO_GPOOL_VERIFY device=%s wrong=%zu/%zu max_abs=%.6f mean_abs=%.6f\n",
                    targetName.c_str(),wrong,actual.size(),maxAbs,sumAbs/actual.size());
        if(wrong)throw std::runtime_error("KataGo GPool fp16 output differs from ONNX CPU reference");
        std::sort(latencies.begin(),latencies.end());
        const double mean=std::accumulate(latencies.begin(),latencies.end(),0.0)/latencies.size();
        std::printf("KATAGO_GPOOL_PASS device=%s mean=%.3f ms p50=%.3f ms p95=%.3f ms\n",
                    targetName.c_str(),mean,pct(latencies,0.5),pct(latencies,0.95));
        std::puts("NOTE: only the actual KataGo rconv5 GPool->w1r branch, NOT full network.");
    }catch(...) {
        if(execution)execFree(execution);
        if(compilation)compFree(compilation);
        if(model)modelFree(model);
        throw;
    }
    execFree(execution);compFree(compilation);modelFree(model);
    return 0;
}
int main(int argc,char** argv) {
    try{
        if(argc<2||argc>5)throw std::runtime_error("Usage: gpool_android data.bin [mtk-mdla] [warmup] [runs]");
        Packet p=readPacket(argv[1]);
        std::string device=argc>2?argv[2]:"mtk-mdla";
        int warmup=argc>3?std::stoi(argv[3]):20;
        int reps=argc>4?std::stoi(argv[4]):100;
        if(warmup<0||reps<1||reps>100000)throw std::runtime_error("Invalid iteration count");
        void* lib=dlopen("libneuronusdk_adapter.9.mtk.so",RTLD_NOW|RTLD_LOCAL);
        if(!lib)lib=dlopen("libneuronusdk_adapter.mtk.so",RTLD_NOW|RTLD_LOCAL);
        if(!lib)throw std::runtime_error("Could not load MediaTek Neuron Adapter");
        std::printf("KATAGO_GPOOL_GRAPH device=%s input=1x%ux%ux%u output=1x1x1x%u\n",
                    device.c_str(),p.h,p.w,p.ci,p.co);
        int rc=run(lib,p,device,warmup,reps);
        dlclose(lib);
        return rc;
    }catch(const std::exception& e) {
        std::fprintf(stderr,"KATAGO_GPOOL_FAIL: %s\n",e.what());
        return 1;
    }
}
