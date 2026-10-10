// MediaTek Neuron Adapter device-constrained 3x3 FP32 convolution smoke benchmark.
// This tests one synthetic convolution, NOT the KataGo 10b network.
// Operation values and operand ordering follow NeuronAdapter / NNAPI CONV_2D ABI.
#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
struct NeuronRuntimeVersion { uint8_t major, minor, patch; };
constexpr int32_t NEURON_INT32 = 1;
constexpr int32_t NEURON_TENSOR_FLOAT32 = 3;
constexpr int32_t NEURON_CONV_2D = 3;
constexpr int NEURON_NO_ERROR = 0;

template<typename F> F api(void* lib, const char* name) {
    void* symbol = dlsym(lib, name);
    if (!symbol) throw std::runtime_error(std::string("Missing Neuron symbol: ") + name);
    return reinterpret_cast<F>(symbol);
}
static void check(int rc, const char* name) {
    if (rc != NEURON_NO_ERROR) throw std::runtime_error(std::string(name) + " returned " + std::to_string(rc));
}
static double percentile(const std::vector<double>& sorted, double pct) {
    double p = (sorted.size() - 1) * pct;
    size_t a = static_cast<size_t>(p), b = std::min(a + 1, sorted.size() - 1);
    return sorted[a] + (sorted[b] - sorted[a]) * (p - a);
}

static int benchmark(void* lib, const std::string& deviceName, int warmup, int runs) {
    auto getVersion = api<int (*)(NeuronRuntimeVersion*)>(lib, "Neuron_getVersion");
    auto deviceCount = api<int (*)(uint32_t*)>(lib, "Neuron_getDeviceCount");
    auto getDevice = api<int (*)(uint32_t, NeuronDevice**)>(lib, "Neuron_getDevice");
    auto deviceGetName = api<int (*)(const NeuronDevice*, const char**)>(lib, "NeuronDevice_getName");
    auto modelCreate = api<int (*)(NeuronModel**)>(lib, "NeuronModel_create");
    auto modelFree = api<void (*)(NeuronModel*)>(lib, "NeuronModel_free");
    auto modelAddOperand = api<int (*)(NeuronModel*, const NeuronOperandType*)>(lib, "NeuronModel_addOperand");
    auto modelSetValue = api<int (*)(NeuronModel*, int32_t, const void*, size_t)>(lib, "NeuronModel_setOperandValue");
    auto modelAddOp = api<int (*)(NeuronModel*, int32_t, uint32_t, const uint32_t*, uint32_t, const uint32_t*)>(lib, "NeuronModel_addOperation");
    auto modelIO = api<int (*)(NeuronModel*, uint32_t, const uint32_t*, uint32_t, const uint32_t*)>(lib, "NeuronModel_identifyInputsAndOutputs");
    auto modelRelax = api<int (*)(NeuronModel*, bool)>(lib, "NeuronModel_relaxComputationFloat32toFloat16");
    auto modelFinish = api<int (*)(NeuronModel*)>(lib, "NeuronModel_finish");
    auto supportedForDevices = api<int (*)(const NeuronModel*, const NeuronDevice* const*, uint32_t, bool*)>(
        lib, "NeuronModel_getSupportedOperationsForDevices");
    auto compForDevices = api<int (*)(NeuronModel*, const NeuronDevice* const*, uint32_t, NeuronCompilation**)>(
        lib, "NeuronCompilation_createForDevices");
    auto compFinish = api<int (*)(NeuronCompilation*)>(lib, "NeuronCompilation_finish");
    auto compFree = api<void (*)(NeuronCompilation*)>(lib, "NeuronCompilation_free");
    auto execCreate = api<int (*)(NeuronCompilation*, NeuronExecution**)>(lib, "NeuronExecution_create");
    auto execSetInput = api<int (*)(NeuronExecution*, int32_t, const NeuronOperandType*, const void*, size_t)>(
        lib, "NeuronExecution_setInput");
    auto execSetOutput = api<int (*)(NeuronExecution*, int32_t, const NeuronOperandType*, void*, size_t)>(
        lib, "NeuronExecution_setOutput");
    auto execCompute = api<int (*)(NeuronExecution*)>(lib, "NeuronExecution_compute");
    auto execFree = api<void (*)(NeuronExecution*)>(lib, "NeuronExecution_free");

    NeuronRuntimeVersion version = {};
    check(getVersion(&version), "Neuron_getVersion");
    std::printf("Neuron SDK version: %u.%u.%u\n", version.major, version.minor, version.patch);
    uint32_t count = 0;
    check(deviceCount(&count), "Neuron_getDeviceCount");
    if (count > 64) throw std::runtime_error("Unreasonable Neuron device count");
    NeuronDevice* target = nullptr;
    for (uint32_t i = 0; i < count; i++) {
        NeuronDevice* candidate = nullptr;
        const char* name = nullptr;
        check(getDevice(i, &candidate), "Neuron_getDevice");
        check(deviceGetName(candidate, &name), "NeuronDevice_getName");
        std::printf("Neuron device[%u]: %s\n", i, name ? name : "(null)");
        if (name && deviceName == name) target = candidate;
    }
    if (!target) throw std::runtime_error("Neuron accelerator not found: " + deviceName);
    std::fflush(stdout);
    const NeuronDevice* selected[] = {target};

    NeuronModel* model = nullptr;
    NeuronCompilation* comp = nullptr;
    NeuronExecution* exec = nullptr;
    try {
        check(modelCreate(&model), "NeuronModel_create");
        constexpr uint32_t H = 19, W = 19, CI = 16, CO = 16, KH = 3, KW = 3;
        const uint32_t inputShape[4] = {1, H, W, CI};
        const uint32_t filterShape[4] = {CO, KH, KW, CI};  // OHWI
        const uint32_t biasShape[1] = {CO};
        const uint32_t outputShape[4] = {1, H, W, CO};
        const NeuronOperandType inputType = {NEURON_TENSOR_FLOAT32, 4, inputShape, 0, 0};
        const NeuronOperandType filterType = {NEURON_TENSOR_FLOAT32, 4, filterShape, 0, 0};
        const NeuronOperandType biasType = {NEURON_TENSOR_FLOAT32, 1, biasShape, 0, 0};
        const NeuronOperandType outputType = {NEURON_TENSOR_FLOAT32, 4, outputShape, 0, 0};
        const NeuronOperandType scalar = {NEURON_INT32, 0, nullptr, 0, 0};

        // 0=NHWC input, 1=OHWI filter, 2=bias
        check(modelAddOperand(model, &inputType), "input operand");
        check(modelAddOperand(model, &filterType), "filter operand");
        check(modelAddOperand(model, &biasType), "bias operand");
        // 3..9: padLeft, padRight, padTop, padBottom, strideW, strideH, activation
        const int32_t scalarValues[7] = {1, 1, 1, 1, 1, 1, 0};
        for (int i = 0; i < 7; i++) {
            check(modelAddOperand(model, &scalar), "scalar operand");
            check(modelSetValue(model, 3 + i, &scalarValues[i], sizeof(int32_t)), "scalar value");
        }
        check(modelAddOperand(model, &outputType), "output operand"); // index 10
        std::vector<float> weights(CO * KH * KW * CI, 1.0f / (CI * KH * KW));
        std::vector<float> biases(CO, 0.5f);
        check(modelSetValue(model, 1, weights.data(), weights.size() * sizeof(float)), "filter constants");
        check(modelSetValue(model, 2, biases.data(), biases.size() * sizeof(float)), "bias constants");
        const uint32_t opInputs[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        const uint32_t out[1] = {10};
        check(modelAddOp(model, NEURON_CONV_2D, 10, opInputs, 1, out), "NeuronModel_addOperation(CONV_2D)");
        const uint32_t in[1] = {0};
        check(modelIO(model, 1, in, 1, out), "NeuronModel_identifyInputsAndOutputs");

        // Most MDLA configurations require reduced FP32 precision for acceleration.
        check(modelRelax(model, true), "NeuronModel_relaxComputationFloat32toFloat16");
        check(modelFinish(model), "NeuronModel_finish");
        bool supported = false;
        check(supportedForDevices(model, selected, 1, &supported), "NeuronModel_getSupportedOperationsForDevices");
        std::printf("NEURON_CONV_2D support device=%s shape=1x19x19x16 -> 1x19x19x16 fp32_relaxed_fp16=yes supported=%s\n",
                    deviceName.c_str(), supported ? "yes" : "no");
        std::fflush(stdout);
        if (!supported) throw std::runtime_error("Selected device reports unsupported CONV_2D. No inference was executed.");
        check(compForDevices(model, selected, 1, &comp), "NeuronCompilation_createForDevices");
        std::puts("Neuron compilation constrained to requested device.");
        std::fflush(stdout);
        check(compFinish(comp), "NeuronCompilation_finish");
        check(execCreate(comp, &exec), "NeuronExecution_create");

        std::vector<float> input(H * W * CI, .25f), output(H * W * CO, NAN);
        check(execSetInput(exec, 0, nullptr, input.data(), input.size() * sizeof(float)), "NeuronExecution_setInput");
        check(execSetOutput(exec, 0, nullptr, output.data(), output.size() * sizeof(float)), "NeuronExecution_setOutput");
        auto infer = [&]() { check(execCompute(exec), "NeuronExecution_compute"); };
        for (int i = 0; i < warmup; i++) infer();

        std::vector<double> times;
        times.reserve(runs);
        for (int i = 0; i < runs; i++) {
            auto start = std::chrono::steady_clock::now();
            infer();
            times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        }
        size_t errors = 0;
        float maxError = 0.0f;
        for (uint32_t y = 0; y < H; y++) {
            for (uint32_t x = 0; x < W; x++) {
                int ky = 0, kx = 0;
                for (int dy = -1; dy <= 1; dy++) if (static_cast<int>(y) + dy >= 0 && y + dy < H) ky++;
                for (int dx = -1; dx <= 1; dx++) if (static_cast<int>(x) + dx >= 0 && x + dx < W) kx++;
                float expected = .5f + .25f * ky * kx / 9.0f;
                for (uint32_t c = 0; c < CO; c++) {
                    float got = output[(y * W + x) * CO + c];
                    if (!std::isfinite(got) || std::fabs(got - expected) > .03f) errors++;
                    if (std::isfinite(got)) maxError = std::max(maxError, std::fabs(got - expected));
                }
            }
        }
        std::printf("CONV output sample=%.5f max_abs_error=%.6f wrong=%zu/%zu\n",
                    output[(W + 1) * CO], maxError, errors, output.size());
        if (errors) throw std::runtime_error("Convolution output failed correctness check");
        double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        std::sort(times.begin(), times.end());
        std::printf("NEURON_CONV_PASS: device=%s mean=%.3f ms p50=%.3f ms p95=%.3f ms calls/s=%.2f\n",
                    deviceName.c_str(), mean, percentile(times, .5), percentile(times, .95), 1000.0 / mean);
        std::puts("NOTE: CONV_2D only; NOT KataGo 10b and NOT KataGo NN evals/s.");
    } catch (...) {
        if (exec) execFree(exec);
        if (comp) compFree(comp);
        if (model) modelFree(model);
        throw;
    }
    execFree(exec);
    compFree(comp);
    modelFree(model);
    return 0;
}

int main(int argc, char** argv) {
    try {
        if (argc > 4) throw std::runtime_error("Usage: neuron_conv_android [mtk-mdla|mtk-dsp] [warmup=20] [runs=100]");
        std::string device = argc > 1 ? argv[1] : "mtk-mdla";
        int warmup = argc > 2 ? std::stoi(argv[2]) : 20;
        int runs = argc > 3 ? std::stoi(argv[3]) : 100;
        if (warmup < 0 || runs < 1 || runs > 100000)
            throw std::runtime_error("Invalid warmup/runs");
        void* library = dlopen("libneuronusdk_adapter.9.mtk.so", RTLD_NOW | RTLD_LOCAL);
        if (!library) library = dlopen("libneuronusdk_adapter.mtk.so", RTLD_NOW | RTLD_LOCAL);
        if (!library) throw std::runtime_error(std::string("Cannot dlopen Neuron Adapter: ") + dlerror());
        std::puts("Loaded Neuron Adapter shared library.");
        int rc = benchmark(library, device, warmup, runs);
        dlclose(library);
        return rc;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "NEURON_CONV_FAIL: %s\n", e.what());
        return 1;
    }
}
