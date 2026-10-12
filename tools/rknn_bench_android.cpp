// RK3588 Android RKNN v2.3.x native NPU benchmark; not a KataGo MCTS benchmark.
#include "rknn_api.h"
#include "rk3588_compare_input.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <numeric>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static void check(int rc, const char* api) {
    if (rc != RKNN_SUCC) throw std::runtime_error(std::string(api) + " failed: " + std::to_string(rc));
}

static double percentile(const std::vector<double>& sorted, double p) {
    double index = (sorted.size() - 1) * p;
    size_t a = static_cast<size_t>(index);
    size_t b = std::min(a + 1, sorted.size() - 1);
    return sorted[a] + (sorted[b] - sorted[a]) * (index - a);
}

static int benchmark(const std::string& path, const std::string& core, int warmup, int runs, const std::string& comparePath, const std::string& inputLayout) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open RKNN model: " + path);
    std::vector<char> model((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (model.empty()) throw std::runtime_error("RKNN model is empty");

    if(inputLayout!="nchw"&&inputLayout!="nhwc")
        throw std::runtime_error("RKNN_INPUT_LAYOUT must be nchw or nhwc");
    std::printf("RKNN_INPUT_LAYOUT requested=%s\n",inputLayout.c_str());
    rknn_context ctx = 0;
    check(rknn_init(&ctx, model.data(), static_cast<uint32_t>(model.size()), 0, nullptr), "rknn_init");
    try {
        rknn_sdk_version version = {};
        if (rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(version)) == RKNN_SUCC)
            std::printf("RKNN runtime: %s | NPU driver: %s\n", version.api_version, version.drv_version);

        const rknn_core_mask mask = core == "0" ? RKNN_NPU_CORE_0 : core == "1" ? RKNN_NPU_CORE_1 :
                                    core == "2" ? RKNN_NPU_CORE_2 : core == "auto" ? RKNN_NPU_CORE_AUTO :
                                    core == "01" ? RKNN_NPU_CORE_0_1 : RKNN_NPU_CORE_0_1_2;
        check(rknn_set_core_mask(ctx, mask), "rknn_set_core_mask");

        rknn_input_output_num io = {};
        check(rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)), "RKNN_QUERY_IN_OUT_NUM");
        if (io.n_input != 3 || io.n_output != 5) {
            throw std::runtime_error("Unexpected tensor count: inputs=" + std::to_string(io.n_input) +
                                     " outputs=" + std::to_string(io.n_output) + " (expected 3/5)");
        }

        std::unique_ptr<KataGoComparisonInput> comparison;
        if(!comparePath.empty()) {
            comparison.reset(new KataGoComparisonInput(comparePath));
            std::printf("COMPARE_INPUT backend=rknn core=%s source=%s\n",core.c_str(),comparePath.c_str());
        }
        std::vector<std::vector<float>> buffers(io.n_input);
        std::vector<size_t> referenceIndices(io.n_output, 99);
        if(comparison) {
            for(uint32_t i=0;i<io.n_output;i++) {
                rknn_tensor_attr attr={};attr.index=i;
                check(rknn_query(ctx,RKNN_QUERY_OUTPUT_ATTR,&attr,sizeof(attr)),"RKNN_QUERY_OUTPUT_ATTR");
                std::printf("output[%u] %s elements=%u\n",i,attr.name,attr.n_elems);
                for(size_t k=0;k<5;k++)if(std::string(attr.name)==KataGoComparisonInput::head(k))referenceIndices[i]=k;
                if(referenceIndices[i]==99)
                    throw std::runtime_error("RKNN output name not mapped to KataGo reference: "+std::string(attr.name));
            }
        }
        std::vector<rknn_input> inputs(io.n_input);
        bool got_spatial = false, got_global = false, got_mask = false;
        for (uint32_t i = 0; i < io.n_input; i++) {
            rknn_tensor_attr attr = {};
            attr.index = i;
            check(rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr)), "RKNN_QUERY_INPUT_ATTR");
            std::printf("input[%u] %s elements=%u format=%d type=%d\n",
                        i, attr.name, attr.n_elems, static_cast<int>(attr.fmt), static_cast<int>(attr.type));
            std::string name(attr.name);
            bool spatial = name.find("InputSpatial") != std::string::npos;
            bool global = name.find("InputGlobal") != std::string::npos;
            bool board_mask = name.find("InputMask") != std::string::npos;
            if ((spatial + global + board_mask) != 1)
                throw std::runtime_error("Unrecognized RKNN input name: " + name);
            const uint32_t expected = spatial ? 22 * 19 * 19 : (global ? 19 : 19 * 19);
            if (attr.n_elems != expected)
                throw std::runtime_error("Unexpected size for " + name + ": " + std::to_string(attr.n_elems));
            if (spatial) got_spatial = true;
            if (global) got_global = true;
            if (board_mask) got_mask = true;
            buffers[i].assign(expected, board_mask ? 1.0f : 0.0f);
            if(comparison)buffers[i]=spatial?comparison->spatial:
                                       (global?comparison->global:comparison->mask);
            // Unquantized host float arrays, NCHW as exported by KataGo.
            inputs[i] = {};
            inputs[i].index = i;
            inputs[i].type = RKNN_TENSOR_FLOAT32;
            inputs[i].fmt = RKNN_TENSOR_NCHW;
            inputs[i].size = static_cast<uint32_t>(buffers[i].size() * sizeof(float));
            inputs[i].buf = buffers[i].data();
        }
        if(inputLayout=="nhwc") {
            for(uint32_t i=0;i<io.n_input;i++) {
                const rknn_input& input=inputs[i];
                if(input.size!=22*19*19*sizeof(float))continue;
                std::vector<float> reshaped(22*19*19);
                for(size_t c=0;c<22;c++)for(size_t y=0;y<19;y++)for(size_t x=0;x<19;x++)
                    reshaped[(y*19+x)*22+c]=buffers[i][c*361+y*19+x];
                buffers[i]=std::move(reshaped);
                inputs[i].buf=buffers[i].data();
            }
            for(rknn_input& input:inputs)input.fmt=RKNN_TENSOR_NHWC;
        }
        if (!got_spatial || !got_global || !got_mask)
            throw std::runtime_error("RKNN model has duplicate/missing input names");

        auto infer = [&](bool verify) {
            check(rknn_inputs_set(ctx, io.n_input, inputs.data()), "rknn_inputs_set");
            check(rknn_run(ctx, nullptr), "rknn_run");
            std::vector<rknn_output> outputs(io.n_output);
            for (uint32_t i = 0; i < io.n_output; i++) {
                outputs[i].index = i;
                outputs[i].want_float = 1;
            }
            check(rknn_outputs_get(ctx, io.n_output, outputs.data(), nullptr), "rknn_outputs_get");
            bool finite = true;
            bool nonempty = true;
            bool accurate = true;
            for (uint32_t i = 0; i < io.n_output; i++) {
                if (!outputs[i].buf || outputs[i].size == 0 || outputs[i].size % sizeof(float)) {
                    nonempty = false;
                    break;
                }
                const auto* values = static_cast<const float*>(outputs[i].buf);
                if(comparison && verify) {
                    const size_t length=outputs[i].size/sizeof(float);
                    auto extremes=std::minmax_element(values,values+length);
                    std::printf("RKNN_OUTPUT_DIAG index=%u head=%s floats=%zu first=%g second=%g min=%g max=%g\\n",
                                i,KataGoComparisonInput::head(referenceIndices[i]),length,
                                values[0],length>1?values[1]:values[0],*extremes.first,*extremes.second);
                }
                for (size_t j = 0; j < outputs[i].size / sizeof(float); j++) {
                    if (!std::isfinite(values[j])) { finite = false; break; }
                }
                if(comparison && verify)
                    accurate=comparison->check(referenceIndices[i],values,outputs[i].size/sizeof(float),"rknn")&&accurate;
            }
            check(rknn_outputs_release(ctx, io.n_output, outputs.data()), "rknn_outputs_release");
            if (!nonempty) throw std::runtime_error("Missing/empty RKNN output");
            if (!finite) throw std::runtime_error("Non-finite RKNN output (NaN/Inf); model is not validated");
            if (!accurate) throw std::runtime_error("RKNN network output precision mismatch vs ONNX CPU reference");
        };

        std::printf("Running RK3588 Android NPU core=%s warmup=%d runs=%d\n", core.c_str(), warmup, runs);
        std::printf("Input tensors are synthetic, NOT a legal KataGo board encoding. NN inference only.\n");
        std::fflush(stdout);
        for (int i = 0; i < warmup; i++) infer(false);
        if(comparison)infer(true);
        std::vector<double> times;
        times.reserve(runs);
        for (int i = 0; i < runs; i++) {
            auto t0 = Clock::now();
            infer(false);
            auto t1 = Clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        double sum = std::accumulate(times.begin(), times.end(), 0.0);
        double mean = sum / times.size();
        std::sort(times.begin(), times.end());
        std::printf("RESULT core=%s mean=%.3f ms p50=%.3f ms p95=%.3f ms nnEvals/s=%.2f\n",
                    core.c_str(), mean, percentile(times, 0.50), percentile(times, 0.95), 1000.0 / mean);
    } catch (...) {
        rknn_destroy(ctx);
        throw;
    }
    check(rknn_destroy(ctx), "rknn_destroy");
    return 0;
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::fprintf(stderr, "Usage: %s MODEL.rknn [0|1|2|01|all|auto] [warmup=20] [runs=100]\n", argv[0]);
            return 2;
        }
        const std::string core = argc >= 3 ? argv[2] : "all";
        int warmup = argc >= 4 ? std::stoi(argv[3]) : 20;
        int runs = argc >= 5 ? std::stoi(argv[4]) : 100;
        if ((core != "0" && core != "1" && core != "2" && core != "01" && core != "all" && core != "auto") || warmup < 0 || runs <= 0 || runs > 1000000) {
            throw std::runtime_error("core must be 0|1|2|01|all|auto; warmup >= 0; 0 < runs <= 1000000");
        }
        const std::string inputLayout=argc>=7?argv[6]:"nchw";
        return benchmark(argv[1], core, warmup, runs, argc>=6?argv[5]:"", inputLayout);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
