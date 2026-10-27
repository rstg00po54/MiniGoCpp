// Standalone FP16 CONV_2D throughput probe for MediaTek Neuron Adapter.
// Device-constrained: no NNAPI/CPU fallback. Measures *effective* model ops,
// NOT silicon peak TOPS, and NOT INT8 TOPS.
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
constexpr int32_t INT32_TYPE = 1;
constexpr int32_t FP16_TYPE = 8;
constexpr int32_t CONV_2D = 3;
constexpr int32_t NO_ERROR = 0;

template<class T> T symbol(void* lib, const char* name) {
    void* p = dlsym(lib, name);
    if (!p) throw std::runtime_error(std::string("Missing Neuron symbol: ") + name);
    return reinterpret_cast<T>(p);
}
void checked(int rc, const char* action) {
    if (rc != NO_ERROR) throw std::runtime_error(std::string(action) + " rc=" + std::to_string(rc));
}
double percentile(const std::vector<double>& v, double fraction) {
    double pos = (v.size() - 1) * fraction;
    size_t lower = static_cast<size_t>(pos);
    size_t upper = std::min(lower + 1, v.size() - 1);
    return v[lower] + (v[upper] - v[lower]) * (pos - lower);
}
struct Case { const char* name; uint32_t h, w, ci, co; };
const Case cases[] = {
    {"go19", 19, 19, 128, 128},
    {"small", 32, 32, 128, 128},
    {"medium", 64, 64, 128, 128},
    {"large", 64, 64, 256, 256},
    {"heavy", 96, 96, 256, 256}
};

struct Neuron {
    void* lib = nullptr;
    using DV = NeuronDevice;
    using M = NeuronModel;
    using C = NeuronCompilation;
    using E = NeuronExecution;
    int (*getVersion)(NeuronRuntimeVersion*);
    int (*getDeviceCount)(uint32_t*);
    int (*getDevice)(uint32_t, DV**);
    int (*getDeviceName)(const DV*, const char**);
    int (*modelCreate)(M**);
    void (*modelFree)(M*);
    int (*addOperand)(M*, const NeuronOperandType*);
    int (*setValue)(M*, int32_t, const void*, size_t);
    int (*addOperation)(M*, int32_t, uint32_t, const uint32_t*, uint32_t, const uint32_t*);
    int (*identifyIO)(M*, uint32_t, const uint32_t*, uint32_t, const uint32_t*);
    int (*finishModel)(M*);
    int (*getSupported)(const M*, const DV* const*, uint32_t, bool*);
    int (*compileForDevices)(M*, const DV* const*, uint32_t, C**);
    int (*finishCompilation)(C*);
    void (*freeCompilation)(C*);
    int (*createExecution)(C*, E**);
    int (*setInput)(E*, int32_t, const NeuronOperandType*, const void*, size_t);
    int (*setOutput)(E*, int32_t, const NeuronOperandType*, void*, size_t);
    int (*compute)(E*);
    void (*freeExecution)(E*);

    Neuron() {
        // Match the library order used by the working KataGo Neuron runner.
        // Keep the actual dlerror() details: a missing dependency or Android
        // linker namespace rejection is not the same as a missing .so.
        const char* libs[] = {
            "libneuronusdk_adapter.9.mtk.so",
            "libneuronusdk_adapter.mtk.so",
            "/system_ext/lib64/libneuronusdk_adapter.9.mtk.so",
            "/system_ext/lib64/libneuronusdk_adapter.mtk.so"
        };
        std::string failures;
        for (const char* name : libs) {
            dlerror();
            lib = dlopen(name, RTLD_NOW | RTLD_LOCAL);
            if (lib) {
                std::printf("Adapter library: %s\n", name);
                break;
            }
            const char* err = dlerror();
            failures += std::string("\n  ") + name + ": " +
                        (err ? err : "unknown Android linker error");
        }
        if (!lib) {
            throw std::runtime_error(
                "MediaTek Neuron Adapter dlopen failed:" + failures +
                "\nCheck the library/dependency paths and Android linker namespace.");
        }
        try {
            getVersion = symbol<decltype(getVersion)>(lib, "Neuron_getVersion");
            getDeviceCount = symbol<decltype(getDeviceCount)>(lib, "Neuron_getDeviceCount");
            getDevice = symbol<decltype(getDevice)>(lib, "Neuron_getDevice");
            getDeviceName = symbol<decltype(getDeviceName)>(lib, "NeuronDevice_getName");
            modelCreate = symbol<decltype(modelCreate)>(lib, "NeuronModel_create");
            modelFree = symbol<decltype(modelFree)>(lib, "NeuronModel_free");
            addOperand = symbol<decltype(addOperand)>(lib, "NeuronModel_addOperand");
            setValue = symbol<decltype(setValue)>(lib, "NeuronModel_setOperandValue");
            addOperation = symbol<decltype(addOperation)>(lib, "NeuronModel_addOperation");
            identifyIO = symbol<decltype(identifyIO)>(lib, "NeuronModel_identifyInputsAndOutputs");
            finishModel = symbol<decltype(finishModel)>(lib, "NeuronModel_finish");
            getSupported = symbol<decltype(getSupported)>(lib, "NeuronModel_getSupportedOperationsForDevices");
            compileForDevices = symbol<decltype(compileForDevices)>(lib, "NeuronCompilation_createForDevices");
            finishCompilation = symbol<decltype(finishCompilation)>(lib, "NeuronCompilation_finish");
            freeCompilation = symbol<decltype(freeCompilation)>(lib, "NeuronCompilation_free");
            createExecution = symbol<decltype(createExecution)>(lib, "NeuronExecution_create");
            setInput = symbol<decltype(setInput)>(lib, "NeuronExecution_setInput");
            setOutput = symbol<decltype(setOutput)>(lib, "NeuronExecution_setOutput");
            compute = symbol<decltype(compute)>(lib, "NeuronExecution_compute");
            freeExecution = symbol<decltype(freeExecution)>(lib, "NeuronExecution_free");
        } catch (...) { dlclose(lib); throw; }
    }
    ~Neuron() { if (lib) dlclose(lib); }
    Neuron(const Neuron&) = delete;
    Neuron& operator=(const Neuron&) = delete;
    const DV* choose(const std::string& requested) {
        NeuronRuntimeVersion ver{};
        checked(getVersion(&ver), "Neuron_getVersion");
        std::printf("Neuron SDK: %u.%u.%u\n", ver.major, ver.minor, ver.patch);
        uint32_t count = 0;
        checked(getDeviceCount(&count), "Neuron_getDeviceCount");
        if (count > 64) throw std::runtime_error("Invalid device count");
        const DV* chosen = nullptr;
        for (uint32_t i = 0; i < count; ++i) {
            DV* p = nullptr;
            const char* name = nullptr;
            checked(getDevice(i, &p), "Neuron_getDevice");
            checked(getDeviceName(p, &name), "NeuronDevice_getName");
            std::printf("  device[%u]: %s\n", i, name ? name : "(unnamed)");
            if (name && requested == name) chosen = p;
        }
        if (!chosen) throw std::runtime_error("Requested device not found: " + requested);
        return chosen;
    }
};

struct Handles {
    Neuron& n;
    Neuron::M* model = nullptr;
    Neuron::C* compilation = nullptr;
    Neuron::E* execution = nullptr;
    explicit Handles(Neuron& api) : n(api) {}
    ~Handles() {
        if (execution) n.freeExecution(execution);
        if (compilation) n.freeCompilation(compilation);
        if (model) n.modelFree(model);
    }
};

struct Result { double meanTops, fastestTops; };
Result runCase(Neuron& n, const NeuronDevice* device, const Case& c,
               int warmup, int runs) {
    constexpr uint32_t K = 3;
    const uint32_t inputDims[4] = {1, c.h, c.w, c.ci};
    const uint32_t filterDims[4] = {c.co, K, K, c.ci};
    const uint32_t biasDims[1] = {c.co};
    const uint32_t outputDims[4] = {1, c.h, c.w, c.co};
    const NeuronOperandType inType = {FP16_TYPE, 4, inputDims, 0, 0};
    const NeuronOperandType filterType = {FP16_TYPE, 4, filterDims, 0, 0};
    const NeuronOperandType biasType = {FP16_TYPE, 1, biasDims, 0, 0};
    const NeuronOperandType outType = {FP16_TYPE, 4, outputDims, 0, 0};
    const NeuronOperandType scalar = {INT32_TYPE, 0, nullptr, 0, 0};
    Handles h(n);
    checked(n.modelCreate(&h.model), "NeuronModel_create");
    checked(n.addOperand(h.model, &inType), "add input");
    checked(n.addOperand(h.model, &filterType), "add filter");
    checked(n.addOperand(h.model, &biasType), "add bias");
    // Explicit SAME padding=1, stride=1, fused activation=NONE (0).
    const int32_t scalars[7] = {1, 1, 1, 1, 1, 1, 0};
    for (int i = 0; i < 7; ++i) {
        checked(n.addOperand(h.model, &scalar), "add scalar");
        checked(n.setValue(h.model, 3 + i, &scalars[i], sizeof(int32_t)), "set scalar");
    }
    checked(n.addOperand(h.model, &outType), "add output");

    const size_t weightCount = size_t(c.co) * K * K * c.ci;
    std::vector<_Float16> weight(weightCount);
    for (uint32_t oc = 0; oc < c.co; ++oc)
        for (uint32_t ky = 0; ky < K; ++ky)
            for (uint32_t kx = 0; kx < K; ++kx)
                for (uint32_t ic = 0; ic < c.ci; ++ic) {
                    size_t idx = ((size_t(oc) * K + ky) * K + kx) * c.ci + ic;
                    weight[idx] = static_cast<_Float16>(
                        (0.75f + 0.05f * ((oc + ic + ky + kx) % 5)) / (K * K * c.ci));
                }
    std::vector<_Float16> bias(c.co);
    for (uint32_t oc = 0; oc < c.co; ++oc) bias[oc] = static_cast<_Float16>(0.2f + .01f * (oc % 4));
    checked(n.setValue(h.model, 1, weight.data(), weight.size() * sizeof(_Float16)), "set filters");
    checked(n.setValue(h.model, 2, bias.data(), bias.size() * sizeof(_Float16)), "set bias");
    const uint32_t opInputs[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const uint32_t opOutputs[1] = {10};
    const uint32_t inputs[1] = {0};
    checked(n.addOperation(h.model, CONV_2D, 10, opInputs, 1, opOutputs), "add CONV_2D");
    checked(n.identifyIO(h.model, 1, inputs, 1, opOutputs), "identify IO");
    checked(n.finishModel(h.model), "finish model");
    const NeuronDevice* devices[1] = {device};
    bool supported = false;
    checked(n.getSupported(h.model, devices, 1, &supported), "get supported operations");
    if (!supported) throw std::runtime_error("CONV_2D unsupported by selected device; no CPU fallback");
    checked(n.compileForDevices(h.model, devices, 1, &h.compilation), "compile for selected device");
    checked(n.finishCompilation(h.compilation), "finish compilation");
    checked(n.createExecution(h.compilation, &h.execution), "create execution");

    std::vector<_Float16> input(size_t(c.h) * c.w * c.ci);
    std::vector<_Float16> output(size_t(c.h) * c.w * c.co, _Float16(-999.0f));
    for (uint32_t y = 0; y < c.h; ++y)
        for (uint32_t x = 0; x < c.w; ++x)
            for (uint32_t ic = 0; ic < c.ci; ++ic)
                input[(size_t(y) * c.w + x) * c.ci + ic] =
                    static_cast<_Float16>(0.125f + .004f * ((y * c.w + x + ic) % 13));
    checked(n.setInput(h.execution, 0, nullptr, input.data(), input.size() * sizeof(_Float16)), "bind input");
    checked(n.setOutput(h.execution, 0, nullptr, output.data(), output.size() * sizeof(_Float16)), "bind output");

    auto infer = [&]() { checked(n.compute(h.execution), "NeuronExecution_compute"); };
    infer();
    // Check FP16 output against direct CPU convolution at corners/edges/center.
    const uint32_t ys[] = {0, 1, c.h / 2, c.h - 2, c.h - 1};
    const uint32_t xs[] = {0, 1, c.w / 2, c.w - 2, c.w - 1};
    const uint32_t channels[] = {0, c.co / 2, c.co - 1};
    float maxError = 0;
    int checkedCount = 0;
    for (uint32_t y : ys) for (uint32_t x : xs) for (uint32_t oc : channels) {
        float expected = static_cast<float>(bias[oc]);
        for (int ky = 0; ky < 3; ++ky) for (int kx = 0; kx < 3; ++kx) {
            int iy = int(y) + ky - 1;
            int ix = int(x) + kx - 1;
            if (iy < 0 || ix < 0 || iy >= int(c.h) || ix >= int(c.w)) continue;
            for (uint32_t ic = 0; ic < c.ci; ++ic) {
                size_t wi = ((size_t(oc) * 3 + ky) * 3 + kx) * c.ci + ic;
                size_t ii = (size_t(iy) * c.w + ix) * c.ci + ic;
                expected += static_cast<float>(weight[wi]) * static_cast<float>(input[ii]);
            }
        }
        const float got = static_cast<float>(output[(size_t(y) * c.w + x) * c.co + oc]);
        if (!std::isfinite(got)) throw std::runtime_error("Nonfinite output");
        const float error = std::fabs(got - expected);
        maxError = std::max(maxError, error);
        if (error > .03f) throw std::runtime_error("Output verification mismatch");
        ++checkedCount;
    }
    std::printf("VERIFY_PASS case=%s samples=%d max_abs_error=%.6f\n", c.name, checkedCount, maxError);
    for (int i = 0; i < warmup; ++i) infer();
    std::vector<double> times;
    times.reserve(size_t(runs));
    for (int i = 0; i < runs; ++i) {
        auto start = std::chrono::steady_clock::now();
        infer();
        times.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count());
    }
    const double meanMs = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    std::sort(times.begin(), times.end());
    // Conventional dense CONV nominal operation count, includes zero padding.
    // One MAC = 2 floating point operations. It is *not* actual hardware op telemetry.
    const double ops = 2.0 * c.h * c.w * c.co * K * K * c.ci;
    const double meanTops = ops / (meanMs * 1.0e9);
    const double fastestTops = ops / (times.front() * 1.0e9);
    std::printf("TOPS_PASS case=%s shape=1x%ux%ux%u->%u FP16 ops=%.3fG "
                "mean=%.4fms p50=%.4fms p95=%.4fms fastest=%.4fms "
                "TOPS_mean=%.4f TOPS_fastest=%.4f\n",
                c.name, c.h, c.w, c.ci, c.co, ops / 1.0e9, meanMs,
                percentile(times, .5), percentile(times, .95), times.front(),
                meanTops, fastestTops);
    std::fflush(stdout);
    return {meanTops, fastestTops};
}
int main(int argc, char** argv) {
    try {
        // tops_bench [all|go19|small|medium|large|heavy] [mtk-mdla|mtk-dsp] [warmup] [runs]
        if (argc > 5) throw std::runtime_error("Usage: neuron_tops_android [profile] [device] [warmup] [runs]");
        const std::string profile = argc > 1 ? argv[1] : "all";
        const std::string deviceName = argc > 2 ? argv[2] : "mtk-mdla";
        int warmup = argc > 3 ? std::stoi(argv[3]) : 20;
        int runs = argc > 4 ? std::stoi(argv[4]) : 100;
        if (deviceName != "mtk-mdla" && deviceName != "mtk-dsp")
            throw std::runtime_error("Device must be mtk-mdla or mtk-dsp");
        if (warmup < 0 || warmup > 10000 || runs < 1 || runs > 10000)
            throw std::runtime_error("Invalid warmup/runs (max 10000)");
        bool validProfile = profile == "all";
        for (const auto& c : cases) validProfile |= profile == c.name;
        if (!validProfile) throw std::runtime_error("Invalid profile; choose all/go19/small/medium/large/heavy");

        Neuron n;
        const NeuronDevice* device = n.choose(deviceName);
        std::printf("PINNED_DEVICE=%s precision=FP16 warmup=%d runs=%d\n",
                    deviceName.c_str(), warmup, runs);
        double maxMean = 0, maxFastest = 0;
        int success = 0, failures = 0;
        for (const auto& c : cases) {
            if (profile == "all" && std::string(c.name) == "heavy") continue;
            if (profile != "all" && profile != c.name) continue;
            std::printf("\nCASE_START %s\n", c.name);
            std::fflush(stdout);
            try {
                Result r = runCase(n, device, c, warmup, runs);
                maxMean = std::max(maxMean, r.meanTops);
                maxFastest = std::max(maxFastest, r.fastestTops);
                ++success;
            } catch (const std::exception& e) {
                std::fprintf(stderr, "TOPS_CASE_FAIL case=%s reason=%s\n", c.name, e.what());
                ++failures;
            }
        }
        std::printf("\nTOPS_SUMMARY device=%s successful=%d failed=%d "
                    "best_mean_effective_FP16_TOPS=%.4f best_single_effective_FP16_TOPS=%.4f\n",
                    deviceName.c_str(), success, failures, maxMean, maxFastest);
        std::puts("WARNING: Model-equivalent FP16 CONV throughput, not rated chip peak/INT8 TOPS.");
        std::puts("Includes host-side synchronous Neuron API overhead; no per-NPU hardware counters.");
        std::puts("Pinned device rules out requested CPU fallback, but vendor-internal scheduling is opaque.");
        return failures || !success ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "TOPS_BENCH_FAIL: %s\n", e.what());
        return 1;
    }
}
