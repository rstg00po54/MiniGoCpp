// Native FP16 and signed INT8 CONV_2D benchmarks for MediaTek Neuron Adapter.
// Device-pinned: no selected-CPU fallback. Measures effective model-equivalent
// throughput, never the silicon peak TOPS or internal hardware counters.
#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <fstream>
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
struct NeuronSymmPerChannelQuantParams {
    uint32_t channelDim;
    uint32_t scaleCount;
    const float* scales;
};
constexpr int32_t INT32_TYPE = 1;
constexpr int32_t FP16_TYPE = 8;
constexpr int32_t INT8_SIGNED_ASYMM = 14;
constexpr int32_t INT8_SIGNED_PER_CHANNEL = 11;
constexpr int32_t INT32_TENSOR = 4;
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
    {"heavy", 96, 96, 256, 256},
    {"x128", 128, 128, 256, 256},
    {"x128c512", 128, 128, 512, 512},
    {"x256", 256, 256, 256, 256},
    {"x256c512", 256, 256, 512, 512}
};


std::string npuTemperature() {
    // Read only explicitly labelled NPU/MDLA/APU zones. Some devices forbid sysfs access.
    for (int i = 0; i < 128; ++i) {
        const std::string root = "/sys/class/thermal/thermal_zone" + std::to_string(i);
        std::ifstream type(root + "/type");
        std::string label;
        if (!type || !std::getline(type, label)) continue;
        std::string lower = label;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (lower.find("npu") == std::string::npos &&
            lower.find("apu") == std::string::npos &&
            lower.find("mdla") == std::string::npos) continue;
        std::ifstream temp(root + "/temp");
        double v = 0;
        if (temp >> v) {
            if (v > 1000) v /= 1000;
            if (v > -40 && v < 150) {
                char result[160] = {};
                std::snprintf(result, sizeof(result), "%.1fC(%s)", v, label.c_str());
                return result;
            }
        }
    }
    return "unavailable";
}
struct Measurement {
    double meanMs = 0, p50Ms = 0, p95Ms = 0, fastestMs = 0;
    double elapsedSec = 0, firstWindowTops = 0, lastWindowTops = 0;
    size_t count = 0;
};
template <typename Infer>
Measurement measure(Infer infer, int warmup, int runs, int durationSec,
                    double ops, const char* caseName, const char* precision) {
    for (int i = 0; i < warmup; ++i) infer();
    using Clock = std::chrono::steady_clock;
    std::vector<double> times;
    times.reserve(durationSec ? size_t(durationSec) * 1000 : size_t(runs));
    const auto start = Clock::now();
    auto windowStart = start;
    size_t windowCount = 0, windowIndex = 0;
    double firstWindow = 0, lastWindow = 0;
    auto emit = [&](Clock::time_point end, bool final) {
        const double secs = std::chrono::duration<double>(end - windowStart).count();
        if (windowCount && secs > 0) {
            const double tops = double(windowCount) * ops / (secs * 1e12);
            if (windowIndex == 0) firstWindow = tops;
            lastWindow = tops;
            ++windowIndex;
            std::printf("SUSTAIN_WINDOW case=%s precision=%s window=%zu elapsed_s=%.2f "
                        "samples=%zu mean_ms=%.4f effective_TOPS=%.4f temp=%s final=%s\n",
                        caseName, precision, windowIndex,
                        std::chrono::duration<double>(end - start).count(),
                        windowCount, 1000.0 * secs / windowCount, tops,
                        npuTemperature().c_str(), final ? "yes" : "no");
            std::fflush(stdout);
        }
        windowStart = end;
        windowCount = 0;
    };
    do {
        const auto begin = Clock::now();
        infer();
        const auto end = Clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
        ++windowCount;
        if (durationSec &&
            std::chrono::duration<double>(end - windowStart).count() >= 5) emit(end, false);
        if (!durationSec && times.size() >= size_t(runs)) break;
        if (durationSec && std::chrono::duration<double>(end - start).count() >= durationSec) break;
    } while (true);
    const auto finish = Clock::now();
    if (durationSec && windowCount) emit(finish, true);
    const double elapsed = std::chrono::duration<double>(finish - start).count();
    const size_t count = times.size();
    const double mean = std::accumulate(times.begin(), times.end(), 0.0) / double(count);
    std::sort(times.begin(), times.end());
    const Measurement result{mean, percentile(times,.5), percentile(times,.95),
                             times.front(), elapsed, firstWindow, lastWindow, count};
    if (durationSec) {
        std::printf("SUSTAIN_SUMMARY case=%s precision=%s requested_s=%d elapsed_s=%.3f "
                    "samples=%zu mean_effective_TOPS=%.4f wall_effective_TOPS=%.4f "
                    "first_window_TOPS=%.4f last_window_TOPS=%.4f last_over_first=%.3f\n",
                    caseName, precision, durationSec, elapsed, count,
                    ops / (mean * 1e9), ops * count / (elapsed * 1e12),
                    firstWindow, lastWindow, firstWindow > 0 ? lastWindow / firstWindow : 0);
    }
    return result;
}

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
    int (*setPerChannelQuant)(M*, int32_t, const NeuronSymmPerChannelQuantParams*) = nullptr;
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
            setPerChannelQuant = reinterpret_cast<decltype(setPerChannelQuant)>(
                dlsym(lib, "NeuronModel_setOperandSymmPerChannelQuantParams"));
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
               int warmup, int runs, int durationSec) {
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
    const double ops = 2.0 * c.h * c.w * c.co * K * K * c.ci;
    const Measurement m = measure(infer, warmup, runs, durationSec, ops, c.name, "fp16");
    const double meanTops = ops / (m.meanMs * 1e9);
    const double fastestTops = ops / (m.fastestMs * 1e9);
    std::printf("TOPS_PASS case=%s shape=1x%ux%ux%u->%u FP16 ops=%.3fG "
                "mean=%.4fms p50=%.4fms p95=%.4fms fastest=%.4fms "
                "TOPS_mean=%.4f TOPS_fastest=%.4f\n",
                c.name, c.h, c.w, c.ci, c.co, ops / 1.0e9, m.meanMs,
                m.p50Ms, m.p95Ms, m.fastestMs,
                meanTops, fastestTops);
    std::fflush(stdout);
    return {meanTops, fastestTops};
}
// Native signed INT8 asymmetric activations + symmetric signed INT8 weights.
// A per-channel signed filter is attempted first; a per-tensor signed asymmetric
// filter is the fallback. A model is NEVER run on an unselected CPU backend.
Result runInt8Variant(Neuron& n, const NeuronDevice* device, const Case& c,
                      int warmup, int runs, bool perChannel, int durationSec) {
    constexpr uint32_t K = 3;
    constexpr float inputScale = 1.0f / 16.0f;
    constexpr float filterScale = 1.0f / 32.0f;
    constexpr float outputScale = 1.0f / 8.0f;
    constexpr float biasScale = inputScale * filterScale;
    const uint32_t inputDims[4] = {1, c.h, c.w, c.ci};
    const uint32_t filterDims[4] = {c.co, K, K, c.ci};
    const uint32_t biasDims[1] = {c.co};
    const uint32_t outputDims[4] = {1, c.h, c.w, c.co};
    const NeuronOperandType inType = {INT8_SIGNED_ASYMM, 4, inputDims, inputScale, 0};
    const NeuronOperandType filterType = {
        perChannel ? INT8_SIGNED_PER_CHANNEL : INT8_SIGNED_ASYMM,
        4, filterDims, perChannel ? 0.0f : filterScale, 0
    };
    const NeuronOperandType biasType = {INT32_TENSOR, 1, biasDims, biasScale, 0};
    const NeuronOperandType outType = {INT8_SIGNED_ASYMM, 4, outputDims, outputScale, 0};
    const NeuronOperandType scalar = {INT32_TYPE, 0, nullptr, 0, 0};
    if (perChannel && !n.setPerChannelQuant)
        throw std::runtime_error("NeuronModel_setOperandSymmPerChannelQuantParams absent");

    Handles h(n);
    checked(n.modelCreate(&h.model), "INT8 create model");
    checked(n.addOperand(h.model, &inType), "INT8 add input");
    checked(n.addOperand(h.model, &filterType), "INT8 add filter");
    // Keep the scale buffer alive until compilation finishes: vendor APIs
    // may defer copying tensor metadata.
    std::vector<float> scales(c.co, filterScale);
    if (perChannel) {
        const NeuronSymmPerChannelQuantParams params = {0, c.co, scales.data()};
        checked(n.setPerChannelQuant(h.model, 1, &params), "INT8 per-channel filter scales");
    }
    checked(n.addOperand(h.model, &biasType), "INT8 add bias");
    const int32_t scalars[7] = {1, 1, 1, 1, 1, 1, 0};
    for (int i = 0; i < 7; ++i) {
        checked(n.addOperand(h.model, &scalar), "INT8 scalar");
        checked(n.setValue(h.model, 3 + i, &scalars[i], sizeof(int32_t)), "INT8 scalar value");
    }
    checked(n.addOperand(h.model, &outType), "INT8 output");

    std::vector<int8_t> weights(size_t(c.co) * K * K * c.ci);
    for (uint32_t oc = 0; oc < c.co; ++oc)
        for (uint32_t ky = 0; ky < K; ++ky)
            for (uint32_t kx = 0; kx < K; ++kx)
                for (uint32_t ic = 0; ic < c.ci; ++ic) {
                    size_t idx = ((size_t(oc) * K + ky) * K + kx) * c.ci + ic;
                    weights[idx] = static_cast<int8_t>(
                        int((oc * 13u + ic * 7u + ky * 11u + kx * 17u) % 15u) - 7);
                }
    std::vector<int32_t> bias(c.co);
    for (uint32_t oc = 0; oc < c.co; ++oc)
        bias[oc] = (int(oc % 7u) - 3) * 24;

    checked(n.setValue(h.model, 1, weights.data(), weights.size()), "INT8 set weights");
    checked(n.setValue(h.model, 2, bias.data(), bias.size() * sizeof(int32_t)), "INT8 set bias");
    const uint32_t opInputs[10] = {0,1,2,3,4,5,6,7,8,9};
    const uint32_t opOutputs[1] = {10};
    const uint32_t inputs[1] = {0};
    checked(n.addOperation(h.model, CONV_2D, 10, opInputs, 1, opOutputs), "INT8 CONV_2D");
    checked(n.identifyIO(h.model, 1, inputs, 1, opOutputs), "INT8 IO");
    checked(n.finishModel(h.model), "INT8 finish model");

    const NeuronDevice* selected[] = {device};
    bool supported = false;
    checked(n.getSupported(h.model, selected, 1, &supported), "INT8 query supported ops");
    std::printf("INT8_OP_SUPPORT case=%s weights=%s supported=%s\n",
                c.name, perChannel ? "per_channel" : "per_tensor",
                supported ? "yes" : "no");
    std::fflush(stdout);
    if (!supported) throw std::runtime_error("Selected device does not support this INT8 CONV_2D");
    checked(n.compileForDevices(h.model, selected, 1, &h.compilation), "INT8 device-pinned compilation");
    checked(n.finishCompilation(h.compilation), "INT8 compilation finish");
    checked(n.createExecution(h.compilation, &h.execution), "INT8 execution create");

    std::vector<int8_t> input(size_t(c.h) * c.w * c.ci);
    std::vector<int8_t> output(size_t(c.h) * c.w * c.co, int8_t(-128));
    for (uint32_t y = 0; y < c.h; ++y)
        for (uint32_t x = 0; x < c.w; ++x)
            for (uint32_t ic = 0; ic < c.ci; ++ic) {
                const uint32_t mix = (y * 29u + x * 17u + ic * 11u) ^
                                     (ic * 7u + y * 3u + x * 13u);
                input[(size_t(y) * c.w + x) * c.ci + ic] =
                    static_cast<int8_t>(int(mix % 63u) - 31);
            }
    checked(n.setInput(h.execution, 0, nullptr, input.data(), input.size()), "INT8 set input");
    checked(n.setOutput(h.execution, 0, nullptr, output.data(), output.size()), "INT8 set output");

    auto infer = [&]() { checked(n.compute(h.execution), "INT8 compute"); };
    infer();

    // Compute reference *integer* dot products, then requantize.
    // Padding means zero in the quantized domain because all zero points are 0.
    const uint32_t ys[] = {0, 1, c.h / 2, c.h - 2, c.h - 1};
    const uint32_t xs[] = {0, 1, c.w / 2, c.w - 2, c.w - 1};
    const uint32_t channels[] = {0, c.co / 2, c.co - 1};
    int maxDiff = 0;
    int checkedCount = 0;
    bool varied = false;
    int firstOutput = 0;
    for (uint32_t y : ys) for (uint32_t x : xs) for (uint32_t oc : channels) {
        int64_t acc = bias[oc];
        for (int ky = 0; ky < 3; ++ky) for (int kx = 0; kx < 3; ++kx) {
            int iy = int(y) + ky - 1;
            int ix = int(x) + kx - 1;
            if (iy < 0 || ix < 0 || iy >= int(c.h) || ix >= int(c.w)) continue;
            for (uint32_t ic = 0; ic < c.ci; ++ic) {
                size_t wi = ((size_t(oc) * K + uint32_t(ky)) * K + uint32_t(kx)) * c.ci + ic;
                size_t ii = (size_t(iy) * c.w + size_t(ix)) * c.ci + ic;
                acc += int(input[ii]) * int(weights[wi]);
            }
        }
        const double value = double(acc) * (double(biasScale) / double(outputScale));
        const long q = std::lround(value);
        const int expected = int(std::max(-128L, std::min(127L, q)));
        const int got = int(output[(size_t(y) * c.w + x) * c.co + oc]);
        const int diff = std::abs(got - expected);
        maxDiff = std::max(maxDiff, diff);
        if (checkedCount == 0) firstOutput = got;
        else if (got != firstOutput) varied = true;
        if (diff > 2) {
            throw std::runtime_error("INT8 output mismatch expected=" +
                std::to_string(expected) + " got=" + std::to_string(got) +
                " y=" + std::to_string(y) + " x=" + std::to_string(x) +
                " oc=" + std::to_string(oc));
        }
        ++checkedCount;
    }
    if (!varied) throw std::runtime_error("INT8 outputs are constant; benchmark not credible");
    std::printf("INT8_VERIFY_PASS case=%s weights=%s samples=%d max_quantized_LSB_error=%d\n",
                c.name, perChannel ? "per_channel" : "per_tensor", checkedCount, maxDiff);
    std::fflush(stdout);

    const double nominalOps = 2.0 * c.h * c.w * c.co * K * K * c.ci;
    const Measurement m = measure(infer, warmup, runs, durationSec, nominalOps, c.name, "int8");
    const double meanTops = nominalOps / (m.meanMs * 1e9);
    const double fastestTops = nominalOps / (m.fastestMs * 1e9);
    std::printf("INT8_TOPS_PASS case=%s weights=%s shape=1x%ux%ux%u->%u "
                "ops=%.3fG mean=%.4fms p50=%.4fms p95=%.4fms fastest=%.4fms "
                "TOPS_mean=%.4f TOPS_fastest=%.4f\n",
                c.name, perChannel ? "per_channel" : "per_tensor",
                c.h, c.w, c.ci, c.co, nominalOps / 1e9, m.meanMs,
                m.p50Ms, m.p95Ms, m.fastestMs,
                meanTops, fastestTops);
    std::fflush(stdout);
    return {meanTops, fastestTops};
}

int main(int argc, char** argv) {
    try {
        // [profile] [mtk-mdla|mtk-dsp] [warmup] [runs] [fp16|int8]
        if (argc > 7) throw std::runtime_error("Usage: neuron_tops_android [profile] [device] [warmup] [runs] [fp16|int8] [durationSec]");
        const std::string profile = argc > 1 ? argv[1] : "all";
        const std::string deviceName = argc > 2 ? argv[2] : "mtk-mdla";
        int warmup = argc > 3 ? std::stoi(argv[3]) : 20;
        int runs = argc > 4 ? std::stoi(argv[4]) : 100;
        const std::string precision = argc > 5 ? argv[5] : "fp16";
        const int durationSec = argc > 6 ? std::stoi(argv[6]) : 0;
        if (durationSec < 0 || durationSec > 180)
            throw std::runtime_error("durationSec must be 0..180");
        if (precision != "fp16" && precision != "int8")
            throw std::runtime_error("precision must be fp16 or int8");
        if (deviceName != "mtk-mdla" && deviceName != "mtk-dsp")
            throw std::runtime_error("Device must be mtk-mdla or mtk-dsp");
        if (warmup < 0 || warmup > 10000 || runs < 1 || runs > 10000)
            throw std::runtime_error("Invalid warmup/runs (max 10000)");
        bool validProfile = profile == "all" || profile == "scale";
        for (const auto& c : cases) validProfile |= profile == c.name;
        if (!validProfile) throw std::runtime_error("Invalid profile; choose all/scale/go19/small/medium/large/heavy/x128/x128c512/x256/x256c512");

        Neuron n;
        const NeuronDevice* device = n.choose(deviceName);
        std::printf("PINNED_DEVICE=%s precision=%s warmup=%d runs=%d duration_s=%d\n",
                    deviceName.c_str(), precision.c_str(), warmup, runs, durationSec);
        double maxMean = 0, maxFastest = 0;
        int success = 0, failures = 0;
        for (const auto& c : cases) {
            const std::string caseName(c.name);
            const bool extra = caseName == "x128" || caseName == "x128c512" ||
                               caseName == "x256" || caseName == "x256c512";
            if (profile == "all" && (caseName == "heavy" || extra)) continue;
            if (profile == "scale" && !extra) continue;
            if (profile != "all" && profile != "scale" && profile != caseName) continue;
            std::printf("\nCASE_START %s\n", c.name);
            std::fflush(stdout);
            try {
                Result r{};
                if (precision == "fp16") {
                    r = runCase(n, device, c, warmup, runs, durationSec);
                } else {
                    try {
                        std::puts("INT8 trying signed asymmetric activations with per-channel signed weights");
                        r = runInt8Variant(n, device, c, warmup, runs, true, durationSec);
                    } catch (const std::exception& firstError) {
                        std::fprintf(stderr, "INT8_PER_CHANNEL_FAIL case=%s reason=%s; trying per-tensor signed weights\n",
                                     c.name, firstError.what());
                        r = runInt8Variant(n, device, c, warmup, runs, false, durationSec);
                    }
                }
                maxMean = std::max(maxMean, r.meanTops);
                maxFastest = std::max(maxFastest, r.fastestTops);
                ++success;
            } catch (const std::exception& e) {
                std::fprintf(stderr, "TOPS_CASE_FAIL case=%s reason=%s\n", c.name, e.what());
                ++failures;
            }
        }
        std::printf("\nTOPS_SUMMARY device=%s precision=%s successful=%d failed=%d "
                    "best_mean_effective_TOPS=%.4f best_single_effective_TOPS=%.4f\n",
                    deviceName.c_str(), precision.c_str(), success, failures, maxMean, maxFastest);
        std::puts("WARNING: Effective model-equivalent CONV throughput, NOT rated chip peak TOPS.");
        std::puts("Includes host-side synchronous Neuron API overhead; no per-NPU hardware counters.");
        std::puts("Pinned device rules out requested CPU fallback, but vendor-internal scheduling is opaque.");
        return failures || !success ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "TOPS_BENCH_FAIL: %s\n", e.what());
        return 1;
    }
}
