// MediaTek Neuron Adapter native Android API smoke test.
// Verifies: dlopen -> API symbols -> FP32 ADD graph compilation -> correct inference.
// IMPORTANT: This is NOT KataGo 10b and does not establish APU/NPU execution.
#include <dlfcn.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

struct NeuronModel;
struct NeuronDevice;
struct NeuronCompilation;
struct NeuronExecution;
struct NeuronRuntimeVersion { uint8_t major, minor, patch; };
struct NeuronOperandType {
    int32_t type;
    uint32_t dimensionCount;
    const uint32_t* dimensions;
    float scale;
    int32_t zeroPoint;
};
static constexpr int NEURON_NO_ERROR = 0;
static constexpr int32_t NEURON_TENSOR_FLOAT32 = 3;
static constexpr int32_t NEURON_INT32 = 1;
static constexpr int32_t NEURON_ADD = 0;

template<typename F> F get(void* library, const char* name) {
    void* p = dlsym(library, name);
    if (!p) throw std::runtime_error(std::string("Missing Neuron symbol: ") + name);
    return reinterpret_cast<F>(p);
}
static void check(int rc, const char* name) {
    if (rc != NEURON_NO_ERROR) throw std::runtime_error(std::string(name) + " returned " + std::to_string(rc));
}

static int run(void* library, const std::string& requestedDevice) {
    auto get_version = get<int (*)(NeuronRuntimeVersion*)>(library, "Neuron_getVersion");
    auto model_create = get<int (*)(NeuronModel**)>(library, "NeuronModel_create");
    auto model_free = get<void (*)(NeuronModel*)>(library, "NeuronModel_free");
    auto model_add_operand = get<int (*)(NeuronModel*, const NeuronOperandType*)>(library, "NeuronModel_addOperand");
    auto model_set_value = get<int (*)(NeuronModel*, int32_t, const void*, size_t)>(library, "NeuronModel_setOperandValue");
    auto model_add_op = get<int (*)(NeuronModel*, int32_t, uint32_t, const uint32_t*, uint32_t, const uint32_t*)>(library, "NeuronModel_addOperation");
    auto model_io = get<int (*)(NeuronModel*, uint32_t, const uint32_t*, uint32_t, const uint32_t*)>(library, "NeuronModel_identifyInputsAndOutputs");
    auto model_finish = get<int (*)(NeuronModel*)>(library, "NeuronModel_finish");
    auto comp_create = get<int (*)(NeuronModel*, NeuronCompilation**)>(library, "NeuronCompilation_create");
    auto comp_finish = get<int (*)(NeuronCompilation*)>(library, "NeuronCompilation_finish");
    auto comp_free = get<void (*)(NeuronCompilation*)>(library, "NeuronCompilation_free");
    auto exec_create = get<int (*)(NeuronCompilation*, NeuronExecution**)>(library, "NeuronExecution_create");
    auto exec_input = get<int (*)(NeuronExecution*, int32_t, const NeuronOperandType*, const void*, size_t)>(library, "NeuronExecution_setInput");
    auto exec_output = get<int (*)(NeuronExecution*, int32_t, const NeuronOperandType*, void*, size_t)>(library, "NeuronExecution_setOutput");
    auto exec_compute = get<int (*)(NeuronExecution*)>(library, "NeuronExecution_compute");
    auto exec_free = get<void (*)(NeuronExecution*)>(library, "NeuronExecution_free");

    // Neuron device enumeration is separate from Android's legacy NNAPI devices.
    // A Neuron API session alone does not prove that the ADD graph used hardware.
    auto device_count = reinterpret_cast<int (*)(uint32_t*)>(dlsym(library, "Neuron_getDeviceCount"));
    auto get_device = reinterpret_cast<int (*)(uint32_t, NeuronDevice**)>(dlsym(library, "Neuron_getDevice"));
    auto get_device_name = reinterpret_cast<int (*)(const NeuronDevice*, const char**)>(dlsym(library, "NeuronDevice_getName"));
    auto create_for_devices = reinterpret_cast<int (*)(NeuronModel*, const NeuronDevice* const*, uint32_t, NeuronCompilation**)>(
        dlsym(library, "NeuronCompilation_createForDevices"));
    auto supported_for_devices = reinterpret_cast<int (*)(const NeuronModel*, const NeuronDevice* const*, uint32_t, bool*)>(
        dlsym(library, "NeuronModel_getSupportedOperationsForDevices"));

    NeuronDevice* selectedDevice = nullptr;
    if (!device_count || !get_device || !get_device_name) {
        std::puts("Neuron device enumeration API not exported; cannot verify hardware device selection.");
        if (!requestedDevice.empty()) throw std::runtime_error("Explicit device requires Neuron device enumeration symbols");
    } else {
        uint32_t count = 0;
        check(device_count(&count), "Neuron_getDeviceCount");
        std::printf("Neuron devices: %u\n", count);
        if (count > 64) throw std::runtime_error("Unreasonable device count");
        for (uint32_t i = 0; i < count; i++) {
            NeuronDevice* dev = nullptr;
            const char* name = nullptr;
            int rc = get_device(i, &dev);
            if (rc == NEURON_NO_ERROR && dev) rc = get_device_name(dev, &name);
            std::printf("Neuron device[%u]: %s (status=%d)\n", i, name ? name : "(unavailable)", rc);
            if (rc == NEURON_NO_ERROR && name && requestedDevice == name) selectedDevice = dev;
        }
    }
    if (!requestedDevice.empty() && !selectedDevice)
        throw std::runtime_error("Neuron device not found: " + requestedDevice);
    if (selectedDevice && !create_for_devices)
        throw std::runtime_error("NeuronCompilation_createForDevices not exported");
    std::fflush(stdout);

    NeuronRuntimeVersion version = {};
    check(get_version(&version), "Neuron_getVersion");
    std::printf("Neuron SDK version: %u.%u.%u\n", version.major, version.minor, version.patch);
    std::fflush(stdout);

    NeuronModel* model = nullptr;
    NeuronCompilation* comp = nullptr;
    NeuronExecution* exec = nullptr;
    try {
        check(model_create(&model), "NeuronModel_create");
        if (!model) throw std::runtime_error("NeuronModel_create returned null");

        constexpr uint32_t count = 64;
        const uint32_t shape[2] = {1, count};
        const NeuronOperandType tensor = {NEURON_TENSOR_FLOAT32, 2, shape, 0.0f, 0};
        const NeuronOperandType activation = {NEURON_INT32, 0, nullptr, 0.0f, 0};
        // 0=first input, 1=second input, 2=fused activation constant, 3=output.
        check(model_add_operand(model, &tensor), "add operand 0");
        check(model_add_operand(model, &tensor), "add operand 1");
        check(model_add_operand(model, &activation), "add activation operand");
        check(model_add_operand(model, &tensor), "add output operand");
        int32_t none = 0;
        check(model_set_value(model, 2, &none, sizeof(none)), "activation=NONE");
        const uint32_t in[3] = {0, 1, 2}, out[1] = {3};
        check(model_add_op(model, NEURON_ADD, 3, in, 1, out), "add NEURON_ADD operation");
        const uint32_t model_input[2] = {0, 1};
        check(model_io(model, 2, model_input, 1, out), "identify inputs/outputs");
        check(model_finish(model), "NeuronModel_finish");

        std::printf("Compiling Neuron Adapter FP32 ADD graph...\n");
        std::fflush(stdout);
        if (selectedDevice) {
            if (supported_for_devices) {
                const NeuronDevice* devices[] = {selectedDevice};
                bool supported = false;
                check(supported_for_devices(model, devices, 1, &supported), "NeuronModel_getSupportedOperationsForDevices");
                std::printf("Selected Neuron device '%s' ADD op supported=%s\n",
                            requestedDevice.c_str(), supported ? "yes" : "no");
                if (!supported) throw std::runtime_error("Selected Neuron device reports unsupported ADD operation");
            } else {
                std::puts("Neuron per-device support query not exported; compilation will decide.");
            }
            const NeuronDevice* devices[] = {selectedDevice};
            check(create_for_devices(model, devices, 1, &comp), "NeuronCompilation_createForDevices");
            std::printf("Neuron compilation constrained to device '%s'.\n", requestedDevice.c_str());
        } else {
            check(comp_create(model, &comp), "NeuronCompilation_create");
            std::puts("Neuron compilation used automatic device selection; may use fallback.");
        }
        if (!comp) throw std::runtime_error("Compilation handle is null");
        check(comp_finish(comp), "NeuronCompilation_finish");

        check(exec_create(comp, &exec), "NeuronExecution_create");
        if (!exec) throw std::runtime_error("Execution handle is null");
        std::vector<float> a(count, 1.25f), b(count, 2.5f), result(count, NAN);
        check(exec_input(exec, 0, nullptr, a.data(), count * sizeof(float)), "NeuronExecution_setInput 0");
        check(exec_input(exec, 1, nullptr, b.data(), count * sizeof(float)), "NeuronExecution_setInput 1");
        check(exec_output(exec, 0, nullptr, result.data(), count * sizeof(float)), "NeuronExecution_setOutput");
        auto start = std::chrono::steady_clock::now();
        check(exec_compute(exec), "NeuronExecution_compute");
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        size_t incorrect = 0;
        for (float value : result) if (!std::isfinite(value) || std::fabs(value - 3.75f) > 0.01f) incorrect++;
        std::printf("Neuron FP32 ADD result[0]=%.4f expected=3.7500, wrong=%zu/%u, single-call=%.3f ms\n",
                    result[0], incorrect, count, ms);
        if (incorrect) throw std::runtime_error("Neuron result failed correctness check");
        std::puts("NEURON_ADAPTER_SMOKE_PASS: graph compiled and inference result verified.");
        std::puts("NOTE: This is a tiny ADD test, NOT KataGo and NOT proof of NPU offload.");
    } catch (...) {
        if (exec) exec_free(exec);
        if (comp) comp_free(comp);
        if (model) model_free(model);
        throw;
    }
    exec_free(exec);
    comp_free(comp);
    model_free(model);
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 2) {
        std::fprintf(stderr, "Usage: %s [NeuronDeviceName]\n", argv[0]);
        return 2;
    }
    const std::string requestedDevice = argc == 2 ? argv[1] : "";
    const char* libraries[] = {
        "libneuronusdk_adapter.9.mtk.so",
        "libneuronusdk_adapter.mtk.so",
        "/system_ext/lib64/libneuronusdk_adapter.9.mtk.so",
        "/system_ext/lib64/libneuronusdk_adapter.mtk.so"
    };
    for (const char* name : libraries) {
        dlerror();
        void* library = dlopen(name, RTLD_NOW | RTLD_LOCAL);
        if (!library) {
            const char* error = dlerror();
            std::fprintf(stderr, "dlopen(%s): %s\n", name, error ? error : "(unknown error)");
            continue;
        }
        std::printf("Loaded Neuron library: %s\n", name);
        std::fflush(stdout);
        try {
            int code = run(library, requestedDevice);
            dlclose(library);
            return code;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "NEURON_ADAPTER_SMOKE_FAIL: %s\n", e.what());
            dlclose(library);
            return 1;
        }
    }
    std::puts("NEURON_ADAPTER_UNAVAILABLE: No adapter could be loaded by adb shell.");
    return 2;
}
