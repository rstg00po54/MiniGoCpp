// Unified NN-only ONNX Runtime CPU / Android NNAPI benchmark.
// NNAPI provider assignment is verified from ORT profile: NNAPI does not imply NPU.
#include <android/NeuralNetworks.h>
#include "onnxruntime_cxx_api.h"
#include "rk3588_compare_input.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <numeric>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" OrtStatus* OrtSessionOptionsAppendExecutionProvider_Nnapi(OrtSessionOptions*, uint32_t);
using Clock = std::chrono::steady_clock;

static double percentile(const std::vector<double>& sorted, double p) {
    double index = (sorted.size() - 1) * p;
    size_t a = static_cast<size_t>(index), b = std::min(a + 1, sorted.size() - 1);
    return sorted[a] + (sorted[b] - sorted[a]) * (index - a);
}

static void listDevices() {
    uint32_t count = 0;
    int result = ANeuralNetworks_getDeviceCount(&count);
    if (result != ANEURALNETWORKS_NO_ERROR) {
        std::printf("NNAPI devices: unavailable (error=%d)\n", result);
        return;
    }
    std::printf("NNAPI devices: %u\n", count);
    for (uint32_t i = 0; i < count; i++) {
        ANeuralNetworksDevice* device = nullptr;
        if (ANeuralNetworks_getDevice(i, &device) != ANEURALNETWORKS_NO_ERROR) continue;
        const char* name = nullptr;
        int32_t type = -1;
        ANeuralNetworksDevice_getName(device, &name);
        ANeuralNetworksDevice_getType(device, &type);
        const char* category = type == ANEURALNETWORKS_DEVICE_ACCELERATOR ? "ACCELERATOR" :
                               type == ANEURALNETWORKS_DEVICE_GPU ? "GPU" :
                               type == ANEURALNETWORKS_DEVICE_CPU ? "CPU" : "OTHER";
        std::printf("NNAPI[%u]: %s type=%s\n", i, name ? name : "?", category);
    }
}

static int profileCount(const std::string& profile, const std::string& name) {
    std::ifstream file(profile);
    if (!file) return -1;
    std::string json((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const std::regex pattern(R"re("provider"\s*:\s*"([^"]+)")re");
    int count = 0;
    for (auto i = std::sregex_iterator(json.begin(), json.end(), pattern); i != std::sregex_iterator(); ++i) {
        if ((*i)[1].str() == name) count++;
    }
    return count;
}

static int run(const std::string& model, const std::string& mode, int warmup, int runs, int threads, const std::string& comparePath) {
    std::printf("ONNX Runtime device benchmark: mode=%s model=%s\n", mode.c_str(), model.c_str());
    std::printf("Input: synthetic 19x19 tensors; batch=1; NN-only; NOT KataGo MCTS visits/s.\n");
    listDevices();
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MiniGoNnBench");
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetIntraOpNumThreads(threads);
    if (mode == "nnapi") {
        // CPU_DISABLED: forbid NNAPI's reference CPU device; ORT may still handle unsupported ops on CPU.
        constexpr uint32_t NNAPI_CPU_DISABLED = 0x004;
        constexpr uint32_t NNAPI_USE_NCHW = 0x002;
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(options, NNAPI_CPU_DISABLED | NNAPI_USE_NCHW));
        options.EnableProfiling("minigo_ort_nnapi");
    }
    auto start = Clock::now();
    Ort::Session session(env, model.c_str(), options);
    double initMs = std::chrono::duration<double, std::milli>(Clock::now() - start).count();

    const std::vector<std::string> expected = {"InputSpatial", "InputGlobal", "InputMask"};
    if (session.GetInputCount() != 3 || session.GetOutputCount() != 5)
        throw std::runtime_error("Expected 3 input and 5 output tensors");
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<Ort::AllocatedStringPtr> inputOwned, outputOwned;
    std::vector<const char*> inputNames, outputNames;
    for (size_t i = 0; i < session.GetInputCount(); i++) {
        inputOwned.push_back(session.GetInputNameAllocated(i, allocator));
        inputNames.push_back(inputOwned.back().get());
        std::printf("input[%zu]: %s\n", i, inputNames.back());
        if (std::find(expected.begin(), expected.end(), inputNames.back()) == expected.end())
            throw std::runtime_error("Unknown model input name: " + std::string(inputNames.back()));
    }
    for (size_t i = 0; i < session.GetOutputCount(); i++) {
        outputOwned.push_back(session.GetOutputNameAllocated(i, allocator));
        outputNames.push_back(outputOwned.back().get());
    }

    // Same shapes and values as the RK3588 RKNN benchmark, but these are NOT complete game features.
    const std::vector<int64_t> shapeSpatial = {1, 22, 19, 19};
    const std::vector<int64_t> shapeGlobal  = {1, 19, 1, 1};
    const std::vector<int64_t> shapeMask    = {1, 1, 19, 19};
    std::vector<float> spatial(22 * 19 * 19, 0.0f);
    std::vector<float> globals(19, 0.0f);
    std::vector<float> mask(19 * 19, 1.0f);
    std::unique_ptr<KataGoComparisonInput> comparison;
    if(!comparePath.empty()) {
        comparison.reset(new KataGoComparisonInput(comparePath));
        spatial=comparison->spatial;globals=comparison->global;mask=comparison->mask;
        std::printf("COMPARE_INPUT backend=%s source=%s\\n",mode.c_str(),comparePath.c_str());
    }
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> values;
    for (const char* name : inputNames) {
        std::string x(name);
        if (x == "InputSpatial")
            values.emplace_back(Ort::Value::CreateTensor<float>(memory, spatial.data(), spatial.size(), shapeSpatial.data(), 4));
        else if (x == "InputGlobal")
            values.emplace_back(Ort::Value::CreateTensor<float>(memory, globals.data(), globals.size(), shapeGlobal.data(), 4));
        else if (x == "InputMask")
            values.emplace_back(Ort::Value::CreateTensor<float>(memory, mask.data(), mask.size(), shapeMask.data(), 4));
    }
    auto infer = [&](bool verify) {
        auto outputs = session.Run(Ort::RunOptions{nullptr}, inputNames.data(), values.data(), values.size(),
                                   outputNames.data(), outputNames.size());
        if (outputs.size() != 5) throw std::runtime_error("Unexpected output count");
        bool accurate=true;
        for (size_t k=0;k<outputs.size();k++) {
            auto& output=outputs[k];
            if (!output.IsTensor()) throw std::runtime_error("Unexpected non-tensor output");
            auto info = output.GetTensorTypeAndShapeInfo();
            auto len = info.GetElementCount();
            if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || len == 0)
                throw std::runtime_error("Unexpected output element type or empty tensor");
            const float* ptr = output.GetTensorData<float>();
            for (size_t i = 0; i < len; i++) {
                if (!std::isfinite(ptr[i])) throw std::runtime_error("Nonfinite model output");
            }
            if(comparison && verify) {
                std::string name(outputNames[k]);
                size_t index=0;
                for(;index<5;index++)if(name==KataGoComparisonInput::head(index))break;
                if(index==5)throw std::runtime_error("Unrecognized ONNX reference output name: "+name);
                accurate=comparison->check(index,ptr,len,"cpu")&&accurate;
            }
        }
        if(!accurate)throw std::runtime_error("ONNX CPU comparison validation failed");
    };
    std::printf("Session initialization: %.2f ms\n", initMs);
    for (int i = 0; i < warmup; i++) infer(false);
    if(comparison)infer(true);
    std::vector<double> latencies;
    latencies.reserve(runs);
    for (int i = 0; i < runs; i++) {
        auto t = Clock::now();
        infer(false);
        latencies.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t).count());
    }
    double mean = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    std::sort(latencies.begin(), latencies.end());
    std::printf("RESULT backend=%s mean=%.3f ms p50=%.3f ms p95=%.3f ms nnEvals/s=%.2f\n",
                mode.c_str(), mean, percentile(latencies, .5), percentile(latencies, .95), 1000.0 / mean);
    if (mode == "nnapi") {
        auto profileName = session.EndProfilingAllocated(allocator);
        std::string profile = profileName.get() ? profileName.get() : "";
        int nnapi = profileCount(profile, "NnapiExecutionProvider");
        int cpu = profileCount(profile, "CPUExecutionProvider");
        std::printf("PROVIDER_PROFILE NNAPI_nodes=%d CPU_nodes=%d path=%s\n", nnapi, cpu, profile.c_str());
        if (nnapi <= 0) std::printf("WARNING: No proven NNAPI execution. Do NOT label this as NPU speed.\n");
        else std::printf("WARNING: NNAPI nodes were used, but driver may select GPU or NPU; this is NOT a verified MediaTek NPU-only result.\n");
    }
    return 0;
}

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::fprintf(stderr, "Usage: %s MODEL.onnx cpu|nnapi [warmup=20] [runs=100] [threads=4]\n", argv[0]);
            return 2;
        }
        std::string mode(argv[2]);
        int warmup = argc > 3 ? std::stoi(argv[3]) : 20;
        int runs = argc > 4 ? std::stoi(argv[4]) : 100;
        int threads = argc > 5 ? std::stoi(argv[5]) : 4;
        if ((mode != "cpu" && mode != "nnapi") || warmup < 0 || runs < 1 || runs > 100000 || threads < 1 || threads > 32)
            throw std::runtime_error("Invalid mode or counts");
        const std::string comparisonPath=argc>6?argv[6]:"";
        return run(argv[1], mode, warmup, runs, threads, comparisonPath);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
