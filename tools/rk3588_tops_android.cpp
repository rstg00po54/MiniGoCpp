// RK3588 Android: quantized 3x3 RKNN Conv sustained effective INT8 TOPS.
// This intentionally refuses non-INT8 RKNN tensors and validates varied outputs.
// Uses actual model-equivalent dense ops, not NPU hardware counters or rated TOPS.
#include "rknn_api.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static void check(int rc, const char* what) {
    if (rc != RKNN_SUCC)
        throw std::runtime_error(std::string(what) + " returned " + std::to_string(rc));
}
static double percentile(const std::vector<double>& sorted, double f) {
    const double i = (sorted.size() - 1) * f;
    size_t a = size_t(i), b = std::min(a + 1, sorted.size() - 1);
    return sorted[a] + (sorted[b] - sorted[a]) * (i - a);
}
struct Case { const char* name; int h,w,ci,co; };
static const Case cases[] = {
    {"rk64",64,64,128,128}, {"heavy",96,96,256,256},
    {"x128",128,128,256,256}, {"x256",256,256,256,256},
    {"x128c512",128,128,512,512}, {"x256c512",256,256,512,512},
};
static const Case& getCase(const std::string& name) {
    for (const auto& c : cases) if (name == c.name) return c;
    throw std::runtime_error("profile must be rk64/heavy/x128/x256/x128c512/x256c512");
}
struct RkContext {
    rknn_context handle = 0;
    ~RkContext() { if (handle) rknn_destroy(handle); }
};
static std::vector<float> makeInput(const Case& c, uint32_t sample) {
    std::vector<float> values(size_t(c.h) * c.w * c.ci);
    for (int ch = 0; ch < c.ci; ++ch)
        for (int y = 0; y < c.h; ++y)
            for (int x = 0; x < c.w; ++x) {
                uint32_t a = uint32_t(y * 29 + x * 13 + ch * 7) + sample * 23;
                uint32_t b = uint32_t(ch * 5 + y * 3 + x * 17);
                const int q = int((a ^ b) % 113) - 56;
                values[(size_t(ch) * c.h + y) * c.w + x] = float(q) * .005f;
            }
    return values;
}
static float refWeight(int oc, int ic, int ky, int kx) {
    return float(((oc * 13 + ic * 7 + ky * 11 + kx * 17) % 29) - 14) * .004f;
}
static float reference(const Case& c, const std::vector<float>& input,
                       int y, int x, int oc) {
    float sum = float((oc % 11) - 5) * .006f;
    for (int ky = 0; ky < 3; ++ky) for (int kx = 0; kx < 3; ++kx) {
        const int iy = y + ky - 1, ix = x + kx - 1;
        if (iy < 0 || ix < 0 || iy >= c.h || ix >= c.w) continue;
        for (int ic = 0; ic < c.ci; ++ic)
            sum += input[(size_t(ic) * c.h + iy) * c.w + ix] * refWeight(oc, ic, ky, kx);
    }
    return sum;
}
// Use model-native quantized bytes and exact model input layout instead of
// relying on SDK implicit NCHW float->NHWC INT8 conversion. This avoids an
// extra host copy/quantization inside the benchmark timing as well.
static std::vector<uint8_t> packInput(const Case& c, const rknn_tensor_attr& attr,
                                      const std::vector<float>& src) {
    if (src.size() != size_t(c.h) * c.w * c.ci)
        throw std::runtime_error("wrong source input shape");
    if (attr.fmt != RKNN_TENSOR_NHWC && attr.fmt != RKNN_TENSOR_NCHW)
        throw std::runtime_error("unknown quantized RKNN input format");
    const int lo = attr.type == RKNN_TENSOR_INT8 ? -128 : 0;
    const int hi = attr.type == RKNN_TENSOR_INT8 ? 127 : 255;
    std::vector<uint8_t> bytes(src.size());
    for (int ch = 0; ch < c.ci; ++ch)
        for (int y = 0; y < c.h; ++y)
            for (int x = 0; x < c.w; ++x) {
                const size_t srcIndex = (size_t(ch) * c.h + y) * c.w + x;
                const size_t dstIndex = attr.fmt == RKNN_TENSOR_NHWC
                    ? (size_t(y) * c.w + x) * c.ci + ch : srcIndex;
                int q = int(std::lround(src[srcIndex] / attr.scale)) + attr.zp;
                q = std::max(lo, std::min(hi, q));
                bytes[dstIndex] = static_cast<uint8_t>(q);
            }
    return bytes;
}
struct Samples { std::vector<float> values; };
static Samples inference(rknn_context context, const rknn_tensor_attr& inputAttr,
                         const rknn_tensor_attr& outputAttr, const Case& c,
                         const std::vector<uint8_t>& inputBytes, bool saveSamples) {
    rknn_input input{};
    input.index = 0;
    input.buf = const_cast<uint8_t*>(inputBytes.data());
    input.size = uint32_t(inputBytes.size());
    input.type = inputAttr.type;
    input.fmt = inputAttr.fmt;
    input.pass_through = 0;  // matched INT8/UINT8 model format, no host conversion.
    check(rknn_inputs_set(context, 1, &input), "rknn_inputs_set");
    check(rknn_run(context, nullptr), "rknn_run");
    rknn_output output{};
    output.index = 0;
    output.want_float = 0;   // avoid FP32 copies; verify actual INT8 tensor.
    check(rknn_outputs_get(context, 1, &output, nullptr), "rknn_outputs_get");
    Samples result;
    try {
        if (!output.buf || output.size < size_t(c.h) * c.w * c.co)
            throw std::runtime_error("missing/short RKNN quantized output");
        if (saveSamples) {
            const int ys[] = {0, 1, c.h/2, c.h-2, c.h-1};
            const int xs[] = {0, 1, c.w/2, c.w-2, c.w-1};
            const int channels[] = {0, c.co/2, c.co-1};
            result.values.reserve(75);
            for (int y : ys) for (int x : xs) for (int oc : channels) {
                const size_t index = outputAttr.fmt == RKNN_TENSOR_NHWC
                    ? (size_t(y) * c.w + x) * c.co + oc
                    : (size_t(oc) * c.h + y) * c.w + x;
                int quant = 0;
                if (outputAttr.type == RKNN_TENSOR_INT8)
                    quant = int(static_cast<const int8_t*>(output.buf)[index]);
                else if (outputAttr.type == RKNN_TENSOR_UINT8)
                    quant = int(static_cast<const uint8_t*>(output.buf)[index]);
                else throw std::runtime_error("output is not INT8/UINT8; refuse TOPS");
                result.values.push_back((quant - outputAttr.zp) * outputAttr.scale);
            }
        }
    } catch (...) {
        rknn_outputs_release(context, 1, &output);
        throw;
    }
    check(rknn_outputs_release(context, 1, &output), "rknn_outputs_release");
    return result;
}
static void verify(const Case& c, rknn_context ctx,
                   const rknn_tensor_attr& inputAttr, const rknn_tensor_attr& outputAttr,
                   const std::vector<float>& input,
                   const std::vector<uint8_t>& packedInput) {
    const auto first = inference(ctx, inputAttr, outputAttr, c, packedInput, true).values;
    if (first.size() != 75) throw std::runtime_error("verification missing 75 samples");
    const int ys[] = {0, 1, c.h/2, c.h-2, c.h-1};
    const int xs[] = {0, 1, c.w/2, c.w-2, c.w-1};
    const int ch[] = {0, c.co/2, c.co-1};
    double maxError = 0;
    size_t i = 0;
    float low = first.front(), high = first.front();
    for (int y : ys) for (int x : xs) for (int oc : ch) {
        const float expected = reference(c, input, y, x, oc);
        const float got = first[i++];
        if (!std::isfinite(got)) throw std::runtime_error("nonfinite RKNN output");
        const double error = std::abs(double(got) - expected);
        maxError = std::max(maxError, error);
        low = std::min(low, got);
        high = std::max(high, got);
        if (error > 0.45 + 0.20 * std::abs(expected))
            throw std::runtime_error("RKNN output differs from CPU conv: expected=" +
                                     std::to_string(expected) + " actual=" + std::to_string(got));
    }
    if (high - low < 0.01)
        throw std::runtime_error("RKNN sample outputs nearly constant");
    auto changedInput = makeInput(c, 1);
    const auto changedPacked = packInput(c, inputAttr, changedInput);
    size_t changedBytes = 0;
    for (size_t n = 0; n < packedInput.size(); ++n)
        if (packedInput[n] != changedPacked[n]) ++changedBytes;
    if (changedBytes == 0) throw std::runtime_error("both test inputs quantize to identical values");
    const auto second = inference(ctx, inputAttr, outputAttr, c, changedPacked, true).values;
    double sumDiff = 0, expectedDiff = 0;
    size_t sample = 0;
    for (int y : ys) for (int x : xs) for (int oc : ch) {
        const double refA = reference(c, input, y, x, oc);
        const double refB = reference(c, changedInput, y, x, oc);
        expectedDiff += std::abs(refB - refA);
        sumDiff += std::abs(first[sample] - second[sample]);
        ++sample;
    }
    const double response = sumDiff / first.size();
    const double referenceResponse = expectedDiff / first.size();
    std::printf("RK_INT8_RESPONSE_DIAG case=%s changed_input_bytes=%zu/%zu "
                "reference_mean_delta=%.5f npu_mean_delta=%.5f "
                "output_scale=%.6f first0=%.5f second0=%.5f\n",
                c.name, changedBytes, packedInput.size(), referenceResponse,
                response, outputAttr.scale, first[0], second[0]);
    std::fflush(stdout);
    if (referenceResponse > .04 && response < std::max(.005, double(outputAttr.scale) * .5))
        throw std::runtime_error(
            "RKNN output stayed unchanged despite clearly different quantized inputs");
    std::printf("RK_INT8_VERIFY_PASS case=%s samples=%zu max_abs_error=%.4f "
                "mean_input_response=%.4f\n",
                c.name, first.size(), maxError, response);
}
static void benchmark(const char* modelPath, const Case& c,
                      const std::string& core, int warmup, int runs, int seconds) {
    std::ifstream f(modelPath, std::ios::binary);
    if (!f) throw std::runtime_error(std::string("cannot read RKNN model: ") + modelPath);
    const std::vector<char> model{std::istreambuf_iterator<char>(f),
                                  std::istreambuf_iterator<char>()};
    if (model.empty()) throw std::runtime_error("empty RKNN file");
    RkContext context;
    check(rknn_init(&context.handle, (void*)model.data(),
                    uint32_t(model.size()), 0, nullptr), "rknn_init");
    const rknn_core_mask mask =
        core=="0" ? RKNN_NPU_CORE_0 :
        core=="1" ? RKNN_NPU_CORE_1 :
        core=="2" ? RKNN_NPU_CORE_2 :
        RKNN_NPU_CORE_0_1_2;
    check(rknn_set_core_mask(context.handle, mask), "rknn_set_core_mask");
    rknn_sdk_version version{};
    if (rknn_query(context.handle, RKNN_QUERY_SDK_VERSION, &version, sizeof(version)) == RKNN_SUCC)
        std::printf("RKNN runtime=%s driver=%s\n", version.api_version, version.drv_version);
    rknn_input_output_num counts{};
    check(rknn_query(context.handle, RKNN_QUERY_IN_OUT_NUM, &counts, sizeof(counts)),
          "RKNN_QUERY_IN_OUT_NUM");
    if (counts.n_input != 1 || counts.n_output != 1)
        throw std::runtime_error("wrong input/output count for synthetic conv");
    rknn_tensor_attr in{}, out{};
    in.index = 0; out.index = 0;
    check(rknn_query(context.handle, RKNN_QUERY_INPUT_ATTR, &in, sizeof(in)),
          "RKNN_QUERY_INPUT_ATTR");
    check(rknn_query(context.handle, RKNN_QUERY_OUTPUT_ATTR, &out, sizeof(out)),
          "RKNN_QUERY_OUTPUT_ATTR");
    std::printf("RKNN_INPUT type=%d fmt=%d qnt=%d scale=%g zp=%d elems=%u\n",
                int(in.type),int(in.fmt),int(in.qnt_type),in.scale,in.zp,in.n_elems);
    std::printf("RKNN_OUTPUT type=%d fmt=%d qnt=%d scale=%g zp=%d elems=%u\n",
                int(out.type),int(out.fmt),int(out.qnt_type),out.scale,out.zp,out.n_elems);
    const auto quantized = [](const rknn_tensor_attr& attr) {
        return (attr.type == RKNN_TENSOR_INT8 || attr.type == RKNN_TENSOR_UINT8) &&
               attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
               std::isfinite(attr.scale) && attr.scale > 0;
    };
    if (!quantized(in) || !quantized(out))
        throw std::runtime_error("RKNN model is not INT8-quantized at input/output: no TOPS claim");
    if (in.n_elems != size_t(c.h)*c.w*c.ci || out.n_elems != size_t(c.h)*c.w*c.co)
        throw std::runtime_error("RKNN model tensor shape does not match requested profile");
    const double ops = 2.0 * c.h * c.w * c.ci * c.co * 3 * 3;
    auto input = makeInput(c, 0);
    const auto packedInput = packInput(c, in, input);
    std::printf("RK_INT8_INPUT_PACKING source=NCHW_FP32 packed=%s_%s bytes=%zu\n",
                in.fmt == RKNN_TENSOR_NHWC ? "NHWC" : "NCHW",
                in.type == RKNN_TENSOR_INT8 ? "INT8" : "UINT8", packedInput.size());
    verify(c, context.handle, in, out, input, packedInput);
    std::printf("RK_INT8_START core=%s case=%s ops=%.3fG warmup=%d runs=%d seconds=%d\n",
                core.c_str(),c.name,ops/1e9,warmup,runs,seconds);
    std::fflush(stdout);
    for (int i=0;i<warmup;i++) inference(context.handle,in,out,c,packedInput,false);
    std::vector<double> timings;
    timings.reserve(seconds ? 50000 : size_t(runs));
    const auto start=Clock::now();
    auto windowStart=start;
    int windowN=0;size_t windowCount=0;double firstTops=0,lastTops=0;
    auto window=[&](Clock::time_point end,bool last){
        const double dt=std::chrono::duration<double>(end-windowStart).count();
        if (windowCount && dt>0) {
            const double tops=ops*windowCount/(dt*1e12);
            if (windowN==0) firstTops=tops;
            lastTops=tops;
            std::printf("RK_SUSTAIN_WINDOW case=%s core=%s window=%d elapsed_s=%.2f "
                        "samples=%zu effective_TOPS=%.4f%s\n",
                        c.name,core.c_str(),++windowN,
                        std::chrono::duration<double>(end-start).count(),
                        windowCount,tops,last?" final=yes":"");
            std::fflush(stdout);
        }
        windowStart=end;windowCount=0;
    };
    do {
        const auto t0=Clock::now();
        inference(context.handle,in,out,c,packedInput,false);
        const auto t1=Clock::now();
        timings.push_back(std::chrono::duration<double,std::milli>(t1-t0).count());
        ++windowCount;
        if (seconds && std::chrono::duration<double>(t1-windowStart).count()>=5.0)
            window(t1,false);
        if (!seconds && timings.size() >= size_t(runs)) break;
        if (seconds && std::chrono::duration<double>(t1-start).count() >= seconds) break;
    } while (true);
    const auto end=Clock::now();
    if (seconds && windowCount) window(end,true);
    const double elapsed=std::chrono::duration<double>(end-start).count();
    const double mean=std::accumulate(timings.begin(),timings.end(),0.0)/timings.size();
    std::sort(timings.begin(),timings.end());
    std::printf("RK_INT8_TOPS_PASS case=%s core=%s precision=int8 "
                "ops=%.3fG runs=%zu mean=%.4fms p50=%.4fms p95=%.4fms "
                "fastest=%.4fms TOPS_mean=%.4f TOPS_fastest=%.4f\n",
                c.name,core.c_str(),ops/1e9,timings.size(),mean,
                percentile(timings,.5),percentile(timings,.95),timings.front(),
                ops/(mean*1e9),ops/(timings.front()*1e9));
    if (seconds) {
        std::printf("RK_SUSTAIN_SUMMARY case=%s core=%s seconds=%.3f runs=%zu "
                    "wall_effective_TOPS=%.4f first_window_TOPS=%.4f "
                    "last_window_TOPS=%.4f last_over_first=%.4f\n",
                    c.name,core.c_str(),elapsed,timings.size(),
                    ops*timings.size()/(elapsed*1e12),firstTops,lastTops,
                    firstTops>0?lastTops/firstTops:0);
    }
    std::puts("WARNING: model-equivalent INT8 conv TOPS including host I/O; not 6 TOPS hardware peak.");
    std::fflush(stdout);
}
int main(int argc,char**argv) {
    try {
        if (argc!=7)
            throw std::runtime_error("Usage: rk3588_tops_android model.rknn profile core warmup runs seconds");
        const Case& c=getCase(argv[2]);
        const std::string core=argv[3];
        if (core!="0"&&core!="1"&&core!="2"&&core!="all")
            throw std::runtime_error("core must be 0/1/2/all");
        int warmup=std::stoi(argv[4]), runs=std::stoi(argv[5]), seconds=std::stoi(argv[6]);
        if(warmup<0||warmup>10000||runs<1||runs>10000||seconds<0||seconds>180)
            throw std::runtime_error("warmup 0..10000, runs 1..10000, seconds 0..180");
        benchmark(argv[1],c,core,warmup,runs,seconds);
        return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"RK_INT8_TOPS_FAIL: %s\n",e.what());
        return 1;
    }
}
