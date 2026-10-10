// Probe whether MediaTek MDLA supports the full 19x19x128 FP16 residual ADD.
// Synthetic tensors only; does NOT prove full KataGo residual block works.
#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

struct NeuronModel;
struct NeuronCompilation;
struct NeuronExecution;
struct NeuronDevice;
struct NeuronOperandType {
    int32_t type; uint32_t dimensionCount; const uint32_t* dimensions;
    float scale; int32_t zeroPoint;
};
template<class F> F sym(void* library, const char* name) {
    void* p=dlsym(library,name);
    if (!p) throw std::runtime_error(std::string("Missing Neuron symbol: ")+name);
    return reinterpret_cast<F>(p);
}
static void check(int rc, const char* operation) {
    if (rc) throw std::runtime_error(std::string(operation)+" rc="+std::to_string(rc));
}
static double pct(const std::vector<double>& v, double q) {
    const double f=(v.size()-1)*q;
    const size_t a=size_t(f),b=std::min(a+1,v.size()-1);
    return v[a]+(v[b]-v[a])*(f-a);
}
static int run(void* lib, const std::string& deviceName,int warmup,int runs) {
    auto count=sym<int (*)(uint32_t*)>(lib,"Neuron_getDeviceCount");
    auto device=sym<int (*)(uint32_t,NeuronDevice**)>(lib,"Neuron_getDevice");
    auto deviceNameApi=sym<int (*)(const NeuronDevice*,const char**)>(lib,"NeuronDevice_getName");
    auto createModel=sym<int (*)(NeuronModel**)>(lib,"NeuronModel_create");
    auto freeModel=sym<void (*)(NeuronModel*)>(lib,"NeuronModel_free");
    auto addOperand=sym<int (*)(NeuronModel*,const NeuronOperandType*)>(lib,"NeuronModel_addOperand");
    auto setValue=sym<int (*)(NeuronModel*,int32_t,const void*,size_t)>(lib,"NeuronModel_setOperandValue");
    auto addOperation=sym<int (*)(NeuronModel*,int32_t,uint32_t,const uint32_t*,uint32_t,const uint32_t*)>(lib,"NeuronModel_addOperation");
    auto setIO=sym<int (*)(NeuronModel*,uint32_t,const uint32_t*,uint32_t,const uint32_t*)>(lib,"NeuronModel_identifyInputsAndOutputs");
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

    uint32_t n=0;check(count(&n),"Neuron_getDeviceCount");
    if(n>64)throw std::runtime_error("Invalid device count");
    NeuronDevice* selected=nullptr;
    for(uint32_t i=0;i<n;i++){
        NeuronDevice* ptr=nullptr;const char* name=nullptr;
        check(device(i,&ptr),"Neuron_getDevice");
        check(deviceNameApi(ptr,&name),"NeuronDevice_getName");
        std::printf("Neuron device[%u]: %s\n",i,name?name:"(null)");
        if(name&&deviceName==name)selected=ptr;
    }
    if(!selected)throw std::runtime_error("Device not found: "+deviceName);
    const NeuronDevice* chosen[1]={selected};

    NeuronModel* model=nullptr;NeuronCompilation* comp=nullptr;NeuronExecution* exec=nullptr;
    try {
        check(createModel(&model),"NeuronModel_create");
        const uint32_t dims[4]={1,19,19,128};
        const NeuronOperandType tensor={8,4,dims,0.0f,0};
        const NeuronOperandType activation={1,0,nullptr,0.0f,0};
        check(addOperand(model,&tensor),"add tensor input A");
        check(addOperand(model,&tensor),"add tensor input B");
        check(addOperand(model,&activation),"add activation");
        check(addOperand(model,&tensor),"add tensor output");
        const int32_t none=0;
        check(setValue(model,2,&none,sizeof(none)),"activation NONE");
        const uint32_t inputs[3]={0,1,2},outputs[1]={3};
        check(addOperation(model,0,3,inputs,1,outputs),"ADD operation");
        const uint32_t externalInputs[2]={0,1};
        check(setIO(model,2,externalInputs,1,outputs),"NeuronModel_identifyInputsAndOutputs");
        check(finishModel(model),"NeuronModel_finish");
        bool isSupported=false;
        check(supported(model,chosen,1,&isSupported),"NeuronModel_getSupportedOperationsForDevices");
        std::printf("NEURON_RESIDUAL_ADD_SUPPORT device=%s dtype=fp16 shape=1x19x19x128 supported=%s\n",
                    deviceName.c_str(),isSupported?"yes":"no");
        std::fflush(stdout);
        if(!isSupported) throw std::runtime_error("Selected device doesn't support this FP16 ADD shape");
        check(createComp(model,chosen,1,&comp),"NeuronCompilation_createForDevices");
        check(finishComp(comp),"NeuronCompilation_finish");
        check(createExec(comp,&exec),"NeuronExecution_create");
        constexpr size_t total=19*19*128;
        std::vector<_Float16> a(total,static_cast<_Float16>(0.25f));
        std::vector<_Float16> b(total,static_cast<_Float16>(0.5f));
        std::vector<_Float16> out(total,static_cast<_Float16>(-1.0f));
        check(setInput(exec,0,nullptr,a.data(),a.size()*sizeof(_Float16)),"NeuronExecution_setInput(A)");
        check(setInput(exec,1,nullptr,b.data(),b.size()*sizeof(_Float16)),"NeuronExecution_setInput(B)");
        check(setOutput(exec,0,nullptr,out.data(),out.size()*sizeof(_Float16)),"NeuronExecution_setOutput");
        for(int i=0;i<warmup;i++)check(compute(exec),"warmup");
        std::vector<double> times;
        times.reserve(runs);
        for(int i=0;i<runs;i++){
            auto t=std::chrono::steady_clock::now();
            check(compute(exec),"NeuronExecution_compute");
            times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());
        }
        size_t bad=0;
        for(const _Float16 value:out)if(!std::isfinite(float(value))||std::fabs(float(value)-0.75f)>0.001f)bad++;
        std::printf("NEURON_RESIDUAL_ADD_VERIFY wrong=%zu/%zu sample=%.4f\n",bad,out.size(),float(out[0]));
        if(bad)throw std::runtime_error("Output mismatch");
        std::sort(times.begin(),times.end());
        const double mean=std::accumulate(times.begin(),times.end(),0.0)/times.size();
        std::printf("NEURON_RESIDUAL_ADD_PASS device=%s dtype=fp16 mean=%.3f ms p50=%.3f ms p95=%.3f ms\n",
                    deviceName.c_str(),mean,pct(times,.50),pct(times,.95));
        std::puts("NOTE: synthetic FP16 ADD only; NOT the full KataGo residual block.");
    } catch(...) {
        if(exec)freeExec(exec);
        if(comp)freeComp(comp);
        if(model)freeModel(model);
        throw;
    }
    freeExec(exec);freeComp(comp);freeModel(model);
    return 0;
}
int main(int argc,char** argv) {
    try {
        if(argc>4)throw std::runtime_error("Usage: residual_add_android [mtk-mdla] [warmup=20] [runs=100]");
        const std::string device=argc>1?argv[1]:"mtk-mdla";
        const int warmup=argc>2?std::stoi(argv[2]):20;
        const int runs=argc>3?std::stoi(argv[3]):100;
        if(warmup<0||runs<1||runs>100000)throw std::runtime_error("Invalid iterations");
        void* lib=dlopen("libneuronusdk_adapter.9.mtk.so",RTLD_NOW|RTLD_LOCAL);
        if(!lib)lib=dlopen("libneuronusdk_adapter.mtk.so",RTLD_NOW|RTLD_LOCAL);
        if(!lib)throw std::runtime_error("Cannot dlopen Neuron Adapter");
        int rc=run(lib,device,warmup,runs);
        dlclose(lib);
        return rc;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"NEURON_RESIDUAL_ADD_FAIL: %s\n",e.what());
        return 1;
    }
}
