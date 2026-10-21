// Execute a MiniGoCpp MGNET001-exported 3x3 RKNN on RK3588 Android.
// Neural network inference only. This is NOT a full Go-playing engine.
#include "rknn_api.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static void ok(int ret, const char* name) {
    if (ret != RKNN_SUCC)
        throw std::runtime_error(std::string(name) + " failed, code " + std::to_string(ret));
}
struct Context {
    rknn_context handle = 0;
    ~Context() { if (handle) rknn_destroy(handle); }
};
struct Prediction {
    std::array<float, 10> logits{};
    float value = 0.0f;
};
static double percentile(const std::vector<double>& sorted, double fraction) {
    const double idx = (sorted.size() - 1) * fraction;
    const size_t lo = static_cast<size_t>(idx);
    const size_t hi = std::min(lo + 1, sorted.size() - 1);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

int benchmark(const std::string& model_path, const std::string& core, int warmup, int runs) {
    std::ifstream f(model_path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open RKNN: " + model_path);
    std::vector<char> model{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
    if (model.empty()) throw std::runtime_error("empty RKNN file");
    Context context;
    ok(rknn_init(&context.handle, model.data(), static_cast<uint32_t>(model.size()), 0, nullptr),
       "rknn_init");

    rknn_sdk_version sdk{};
    if (rknn_query(context.handle, RKNN_QUERY_SDK_VERSION, &sdk, sizeof(sdk)) == RKNN_SUCC)
        std::printf("RKNN runtime=%s NPU driver=%s\n", sdk.api_version, sdk.drv_version);

    rknn_core_mask mask = core == "0" ? RKNN_NPU_CORE_0 :
                          core == "1" ? RKNN_NPU_CORE_1 :
                          core == "2" ? RKNN_NPU_CORE_2 : RKNN_NPU_CORE_0_1_2;
    ok(rknn_set_core_mask(context.handle, mask), "rknn_set_core_mask");

    rknn_input_output_num counts{};
    ok(rknn_query(context.handle, RKNN_QUERY_IN_OUT_NUM, &counts, sizeof(counts)),
       "RKNN_QUERY_IN_OUT_NUM");
    if (counts.n_input != 1 || counts.n_output != 2)
        throw std::runtime_error("expected MiniGo 1 input / 2 outputs, got " +
            std::to_string(counts.n_input) + "/" + std::to_string(counts.n_output));

    rknn_tensor_attr in_attr{};
    in_attr.index = 0;
    ok(rknn_query(context.handle, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr)),
       "RKNN_QUERY_INPUT_ATTR");
    if (in_attr.n_elems != 13)
        throw std::runtime_error("this sample expects 3x3 MiniGo input with 13 floats, got " +
            std::to_string(in_attr.n_elems));
    std::printf("Input: %s, elements=%u, core=%s\n", in_attr.name, in_attr.n_elems,
                core.c_str());

    int policy_index = -1, value_index = -1;
    std::array<rknn_tensor_attr, 2> attrs{};
    for (uint32_t i = 0; i < 2; ++i) {
        attrs[i].index = i;
        ok(rknn_query(context.handle, RKNN_QUERY_OUTPUT_ATTR, &attrs[i], sizeof(attrs[i])),
           "RKNN_QUERY_OUTPUT_ATTR");
        std::printf("Output[%u]: %s, elements=%u\n", i, attrs[i].name, attrs[i].n_elems);
        if (attrs[i].n_elems == 10) policy_index = int(i);
        if (attrs[i].n_elems == 1) value_index = int(i);
    }
    if (policy_index < 0 || value_index < 0)
        throw std::runtime_error("expected policy[10] and value[1] outputs");

    // Empty 3x3 board, Black to move, 0.5 komi, no previous PASS, zero moves.
    // Exactly mirrors Board::encode() in src/board.cpp.
    std::array<float, 13> features{};
    features[9] = 1.0f;             // toMove: Black
    features[10] = 0.5f / 9.0f;    // komi * toMove / (n*n)
    std::printf("Board: empty 3x3, Black to move, komi=0.5\n");

    rknn_input input{};
    input.index = 0;
    input.buf = features.data();
    input.size = static_cast<uint32_t>(features.size() * sizeof(float));
    input.type = RKNN_TENSOR_FLOAT32;
    // The N/C/H/W tensor is 1x13x1x1: NCHW and NHWC share identical flat order.
    input.fmt = RKNN_TENSOR_NHWC;
    input.pass_through = 0;

    auto predict = [&]() -> Prediction {
        ok(rknn_inputs_set(context.handle, 1, &input), "rknn_inputs_set");
        ok(rknn_run(context.handle, nullptr), "rknn_run");
        std::array<rknn_output, 2> outputs{};
        for (uint32_t i = 0; i < 2; ++i) {
            outputs[i].index = i;
            outputs[i].want_float = 1;
        }
        ok(rknn_outputs_get(context.handle, 2, outputs.data(), nullptr), "rknn_outputs_get");
        Prediction p;
        try {
            for (uint32_t i = 0; i < 2; ++i) {
                const uint32_t count = attrs[i].n_elems;
                if (!outputs[i].buf || outputs[i].size < count * sizeof(float))
                    throw std::runtime_error("missing or undersized RKNN output");
                const float* vals = static_cast<const float*>(outputs[i].buf);
                for (uint32_t j = 0; j < count; ++j)
                    if (!std::isfinite(vals[j]))
                        throw std::runtime_error("RKNN output contains NaN/Inf");
                if (int(i) == policy_index)
                    std::copy(vals, vals + 10, p.logits.begin());
                else if (int(i) == value_index)
                    p.value = vals[0];
            }
        } catch (...) {
            rknn_outputs_release(context.handle, 2, outputs.data());
            throw;
        }
        ok(rknn_outputs_release(context.handle, 2, outputs.data()), "rknn_outputs_release");
        return p;
    };

    for (int i = 0; i < warmup; ++i) predict();
    Prediction last{};
    std::vector<double> timings;
    timings.reserve(runs);
    for (int i = 0; i < runs; ++i) {
        auto start = Clock::now();
        last = predict();
        auto end = Clock::now();
        timings.push_back(std::chrono::duration<double, std::milli>(end-start).count());
    }
    double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
    std::sort(timings.begin(), timings.end());

    // On an empty board all 9 placements and PASS are legal.
    const float largest = *std::max_element(last.logits.begin(), last.logits.end());
    std::array<double, 10> probabilities{};
    double total = 0;
    for (size_t i = 0; i < 10; ++i) {
        probabilities[i] = std::exp(double(last.logits[i] - largest));
        total += probabilities[i];
    }
    std::puts("Move  policy_logit  probability");
    for (int i = 0; i < 10; ++i) {
        const std::string move = i == 9 ? "PASS" :
            std::string(1, char('A' + i%3)) + std::to_string(i/3+1);
        std::printf("%-5s %12.6f %10.2f%%\n", move.c_str(), last.logits[i],
                    100.0 * probabilities[i] / total);
    }
    std::printf("value=%.6f (current player, tanh)\n", last.value);
    std::printf("RESULT core=%s runs=%d mean=%.4f ms p50=%.4f ms p95=%.4f ms NN/s=%.1f\n",
                core.c_str(), runs, sum / runs, percentile(timings, 0.5),
                percentile(timings, 0.95), 1000.0 * runs / sum);
    std::puts("NN inference only; MCTS/search is not included.");
    return 0;
}
int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::fprintf(stderr,
                "Usage: %s my3.rknn [0|1|2|all] [warmup=20] [runs=100]\n", argv[0]);
            return 2;
        }
        const std::string core = argc > 2 ? argv[2] : "0";
        int warmup = argc > 3 ? std::stoi(argv[3]) : 20;
        int runs = argc > 4 ? std::stoi(argv[4]) : 100;
        if ((core!="0"&&core!="1"&&core!="2"&&core!="all") ||
             warmup < 0 || runs <= 0 || runs > 100000)
            throw std::runtime_error("invalid core, warmup or runs");
        return benchmark(argv[1], core, warmup, runs);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "ERROR: %s\n", ex.what());
        return 1;
    }
}
