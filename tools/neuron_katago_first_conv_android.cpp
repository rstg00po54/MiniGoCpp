// Execute ONE real KataGo ONNX Conv's weights/input on native MediaTek Neuron MDLA.
// Packet comes from export_katago_first_conv.py. Compare against full-precision ONNX CPU.
// NOT a full KataGo network inference benchmark.
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

struct NeuronDevice;
struct NeuronModel;
struct NeuronCompilation;
struct NeuronExecution;
struct NeuronOperandType {
    int32_t type;
    uint32_t dimensionCount;
    const uint32_t* dimensions;
    float scale;
    int32_t zeroPoint;
};
static constexpr int32_t NEURON_INT32 = 1;
static constexpr int32_t NEURON_TENSOR_FLOAT16 = 8;
static constexpr int32_t NEURON_CONV_2D = 3;

template<typename F> F load(void* lib, const char* symbol) {
    void* p = dlsym(lib, symbol);
    if (!p) throw std::runtime_error(std::string("Missing Neuron API symbol: ") + symbol);
    return reinterpret_cast<F>(p);
}
static void check(int rc, const char* op) {
    if (rc) throw std::runtime_error(std::string(op) + " failed rc=" + std::to_string(rc));
}
static double percentile(const std::vector<double>& v, double q) {
    double i = (v.size()-1)*q;
    size_t a = size_t(i), b = std::min(a+1, v.size()-1);
    return v[a] + (v[b]-v[a])*(i-a);
}

struct Packet {
    uint32_t h,w,ci,co,kh,kw,pl,pr,pt,pb,oh,ow;
    std::vector<uint16_t> input, weight, bias;
    std::vector<float> golden;
};

static Packet readPacket(const char* file) {
    std::ifstream f(file, std::ios::binary);
    if (!f) throw std::runtime_error(std::string("Cannot open test data: ") + file);
    std::array<char,8> magic{};
    f.read(magic.data(), magic.size());
    if (std::memcmp(magic.data(), "K10CV001", 8)) throw std::runtime_error("Invalid packet magic");
    std::array<uint32_t,12> d{};
    f.read(reinterpret_cast<char*>(d.data()), sizeof(d));
    if (!f) throw std::runtime_error("Packet header truncated");
    Packet p{};
    p.h=d[0]; p.w=d[1]; p.ci=d[2]; p.co=d[3]; p.kh=d[4]; p.kw=d[5];
    p.pl=d[6]; p.pr=d[7]; p.pt=d[8]; p.pb=d[9]; p.oh=d[10]; p.ow=d[11];
    if (p.h<1 || p.h>128 || p.w<1 || p.w>128 || p.ci<1 || p.ci>2048 ||
        p.co<1 || p.co>2048 || p.kh<1 || p.kh>7 || p.kw<1 || p.kw>7 ||
        p.pl>7 || p.pr>7 || p.pt>7 || p.pb>7 || p.oh<1 || p.oh>128 || p.ow<1 || p.ow>128 ||
        p.h + p.pt + p.pb - p.kh + 1 != p.oh || p.w + p.pl + p.pr - p.kw + 1 != p.ow)
        throw std::runtime_error("Invalid or unsupported Conv dimensions/padding");
    p.input.resize(size_t(p.h)*p.w*p.ci);
    p.weight.resize(size_t(p.co)*p.kh*p.kw*p.ci);
    p.bias.resize(p.co);
    p.golden.resize(size_t(p.oh)*p.ow*p.co);
    auto take = [&](void* ptr, size_t bytes) {
        f.read(reinterpret_cast<char*>(ptr), bytes);
        if (!f) throw std::runtime_error("Packet data truncated");
    };
    take(p.input.data(), p.input.size()*2);
    take(p.weight.data(), p.weight.size()*2);
    take(p.bias.data(), p.bias.size()*2);
    take(p.golden.data(), p.golden.size()*4);
    char extra;
    if (f.read(&extra, 1)) throw std::runtime_error("Unexpected bytes after packet");
    return p;
}

static int evaluate(void* lib, const Packet& p, const std::string& deviceName, int warmup, int runs) {
    auto getDeviceCount = load<int (*)(uint32_t*)>(lib, "Neuron_getDeviceCount");
    auto getDevice = load<int (*)(uint32_t, NeuronDevice**)>(lib, "Neuron_getDevice");
    auto getDeviceName = load<int (*)(const NeuronDevice*, const char**)>(lib, "NeuronDevice_getName");
    auto createModel = load<int (*)(NeuronModel**)>(lib, "NeuronModel_create");
    auto freeModel = load<void (*)(NeuronModel*)>(lib, "NeuronModel_free");
    auto addOperand = load<int (*)(NeuronModel*, const NeuronOperandType*)>(lib, "NeuronModel_addOperand");
    auto setValue = load<int (*)(NeuronModel*, int32_t, const void*, size_t)>(lib, "NeuronModel_setOperandValue");
    auto addOperation = load<int (*)(NeuronModel*, int32_t, uint32_t, const uint32_t*, uint32_t, const uint32_t*)>(
        lib, "NeuronModel_addOperation");
    auto setIO = load<int (*)(NeuronModel*, uint32_t, const uint32_t*, uint32_t, const uint32_t*)>(
        lib, "NeuronModel_identifyInputsAndOutputs");
    auto finishModel = load<int (*)(NeuronModel*)>(lib, "NeuronModel_finish");
    auto querySupported = load<int (*)(const NeuronModel*, const NeuronDevice* const*, uint32_t, bool*)>(
        lib, "NeuronModel_getSupportedOperationsForDevices");
    auto createCompilation = load<int (*)(NeuronModel*, const NeuronDevice* const*, uint32_t, NeuronCompilation**)>(
        lib, "NeuronCompilation_createForDevices");
    auto finishCompilation = load<int (*)(NeuronCompilation*)>(lib, "NeuronCompilation_finish");
    auto freeCompilation = load<void (*)(NeuronCompilation*)>(lib, "NeuronCompilation_free");
    auto createExecution = load<int (*)(NeuronCompilation*, NeuronExecution**)>(lib, "NeuronExecution_create");
    auto setInput = load<int (*)(NeuronExecution*, int32_t, const NeuronOperandType*, const void*, size_t)>(
        lib, "NeuronExecution_setInput");
    auto setOutput = load<int (*)(NeuronExecution*, int32_t, const NeuronOperandType*, void*, size_t)>(
        lib, "NeuronExecution_setOutput");
    auto compute = load<int (*)(NeuronExecution*)>(lib, "NeuronExecution_compute");
    auto freeExecution = load<void (*)(NeuronExecution*)>(lib, "NeuronExecution_free");

    NeuronDevice* device = nullptr;
    uint32_t deviceCount = 0;
    check(getDeviceCount(&deviceCount), "Neuron_getDeviceCount");
    if (deviceCount > 64) throw std::runtime_error("Invalid device count");
    for (uint32_t i=0; i<deviceCount; ++i) {
        NeuronDevice* d=nullptr;
        const char* name=nullptr;
        check(getDevice(i, &d), "Neuron_getDevice");
        check(getDeviceName(d, &name), "NeuronDevice_getName");
        std::printf("Neuron device[%u]: %s\n", i, name ? name : "(unnamed)");
        if (name && deviceName == name) device=d;
    }
    if (!device) throw std::runtime_error("Requested device not found: " + deviceName);
    const NeuronDevice* devices[] = {device};

    NeuronModel* model=nullptr;
    NeuronCompilation* compilation=nullptr;
    NeuronExecution* exec=nullptr;
    try {
        check(createModel(&model), "NeuronModel_create");
        const uint32_t inputDims[4] = {1,p.h,p.w,p.ci};
        const uint32_t filterDims[4] = {p.co,p.kh,p.kw,p.ci};
        const uint32_t biasDims[1] = {p.co};
        const uint32_t outDims[4] = {1,p.oh,p.ow,p.co};
        const NeuronOperandType inputType = {NEURON_TENSOR_FLOAT16,4,inputDims,0,0};
        const NeuronOperandType filterType = {NEURON_TENSOR_FLOAT16,4,filterDims,0,0};
        const NeuronOperandType biasType = {NEURON_TENSOR_FLOAT16,1,biasDims,0,0};
        const NeuronOperandType scalar = {NEURON_INT32,0,nullptr,0,0};
        const NeuronOperandType outType = {NEURON_TENSOR_FLOAT16,4,outDims,0,0};
        check(addOperand(model,&inputType),"add input operand");
        check(addOperand(model,&filterType),"add filter operand");
        check(addOperand(model,&biasType),"add bias operand");
        const int32_t params[7] = {
            int32_t(p.pl),int32_t(p.pr),int32_t(p.pt),int32_t(p.pb),1,1,0
        };
        for (int i=0; i<7; ++i) {
            check(addOperand(model,&scalar),"add scalar operand");
            check(setValue(model,3+i,&params[i],4),"set scalar operand");
        }
        check(addOperand(model,&outType),"add output operand");
        check(setValue(model,1,p.weight.data(),p.weight.size()*2),"set original KataGo Conv weights (FP16)");
        check(setValue(model,2,p.bias.data(),p.bias.size()*2),"set KataGo bias (FP16)");
        const uint32_t operands[10] = {0,1,2,3,4,5,6,7,8,9};
        const uint32_t outputIndex[1] = {10}, inputIndex[1] = {0};
        check(addOperation(model,NEURON_CONV_2D,10,operands,1,outputIndex),"add CONV_2D op");
        check(setIO(model,1,inputIndex,1,outputIndex),"NeuronModel_identifyInputsAndOutputs");
        check(finishModel(model),"NeuronModel_finish");
        bool supported=false;
        check(querySupported(model,devices,1,&supported),"NeuronModel_getSupportedOperationsForDevices");
        std::printf("KATAGO_FIRST_CONV_SUPPORT device=%s dtype=fp16 input=1x%ux%ux%u output=1x%ux%ux%u kernel=%ux%u supported=%s\n",
                    deviceName.c_str(),p.h,p.w,p.ci,p.oh,p.ow,p.co,p.kh,p.kw,supported?"yes":"no");
        std::fflush(stdout);
        if (!supported) throw std::runtime_error("Real KataGo first Conv not supported by selected Neuron device");
        check(createCompilation(model,devices,1,&compilation),"NeuronCompilation_createForDevices");
        check(finishCompilation(compilation),"NeuronCompilation_finish");
        std::puts("Neuron compiled real KataGo first Conv exclusively for the requested device.");
        std::fflush(stdout);
        check(createExecution(compilation,&exec),"NeuronExecution_create");

        std::vector<uint16_t> actual(p.golden.size(),0);
        check(setInput(exec,0,nullptr,p.input.data(),p.input.size()*2),"NeuronExecution_setInput");
        check(setOutput(exec,0,nullptr,actual.data(),actual.size()*2),"NeuronExecution_setOutput");
        for (int i=0;i<warmup;i++) check(compute(exec),"NeuronExecution_compute(warmup)");
        std::vector<double> latencies;
        latencies.reserve(runs);
        for (int i=0;i<runs;i++) {
            auto start=std::chrono::steady_clock::now();
            check(compute(exec),"NeuronExecution_compute");
            latencies.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
        }

        size_t wrong=0;
        double maxError=0, avgError=0, maxRelative=0;
        size_t worstIndex=0;
        float worstGot=0, worstExpected=0;
        for (size_t i=0;i<actual.size();i++) {
            _Float16 half;
            std::memcpy(&half,&actual[i],sizeof(half));
            float got=static_cast<float>(half);
            float ref=p.golden[i];
            double delta=std::fabs(double(got)-double(ref));
            if (!std::isfinite(got) || !std::isfinite(ref)) delta=INFINITY;
            if (delta>maxError) { maxError=delta; worstIndex=i; worstGot=got; worstExpected=ref; }
            if (std::isfinite(delta)) avgError+=delta;
            if (std::isfinite(delta)) maxRelative=std::max(maxRelative,delta/(1e-2+std::fabs(double(ref))));
            if (delta>0.05+0.03*std::fabs(double(ref))) wrong++;
        }
        avgError/=actual.size();
        std::printf("KATAGO_FIRST_CONV_VERIFY wrong=%zu/%zu max_abs=%.6f mean_abs=%.6f max_scaled=%.5f worst_idx=%zu mdla=%.6f onnx=%.6f\n",
                    wrong,actual.size(),maxError,avgError,maxRelative,worstIndex,worstGot,worstExpected);
        if (wrong) throw std::runtime_error("MDLA first Conv differs from ONNX CPU reference beyond FP16 tolerances");
        std::sort(latencies.begin(),latencies.end());
        double mean=std::accumulate(latencies.begin(),latencies.end(),0.0)/latencies.size();
        std::printf("KATAGO_FIRST_CONV_PASS device=%s mean=%.3f ms p50=%.3f ms p95=%.3f ms conv_calls/s=%.2f\n",
                    deviceName.c_str(),mean,percentile(latencies,.5),percentile(latencies,.95),1000.0/mean);
        std::puts("NOTE: One REAL KataGo Conv only; NOT full KataGo NN evaluations or MCTS visits/s.");
    } catch (...) {
        if (exec) freeExecution(exec);
        if (compilation) freeCompilation(compilation);
        if (model) freeModel(model);
        throw;
    }
    freeExecution(exec);
    freeCompilation(compilation);
    freeModel(model);
    return 0;
}

int main(int argc,char** argv) {
    try {
        if (argc<2 || argc>5) {
            std::fprintf(stderr,"Usage: %s kata_first_conv.bin [mtk-mdla] [warmup=20] [runs=100]\n",argv[0]);
            return 2;
        }
        Packet p=readPacket(argv[1]);
        const std::string device=argc>=3?argv[2]:"mtk-mdla";
        int warmup=argc>=4?std::stoi(argv[3]):20;
        int runs=argc>=5?std::stoi(argv[4]):100;
        if (warmup<0 || runs<1 || runs>100000) throw std::runtime_error("Invalid warmup/runs");
        void* lib=dlopen("libneuronusdk_adapter.9.mtk.so",RTLD_NOW|RTLD_LOCAL);
        if (!lib) lib=dlopen("libneuronusdk_adapter.mtk.so",RTLD_NOW|RTLD_LOCAL);
        if (!lib) throw std::runtime_error("Failed to dlopen Neuron Adapter");
        std::printf("KataGo ONNX first Conv packet: %ux%ux%u => %ux%ux%u\n",p.h,p.w,p.ci,p.oh,p.ow,p.co);
        int result=evaluate(lib,p,device,warmup,runs);
        dlclose(lib);
        return result;
    } catch (const std::exception& e) {
        std::fprintf(stderr,"KATAGO_FIRST_CONV_FAIL: %s\n",e.what());
        return 1;
    }
}
