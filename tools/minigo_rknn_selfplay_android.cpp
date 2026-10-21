// MiniGoCpp 3x3 self-play using the ORIGINAL Board and MCTS, and RK3588 RKNN inference.
// Build as a native Android ARM64 executable. This does not train or alter any model.
#include "board.h"
#include "search.h"
#include "rknn_api.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static void check(int rc, const char* api) {
    if (rc != RKNN_SUCC)
        throw std::runtime_error(std::string(api) + " failed: " + std::to_string(rc));
}

class RknnEvaluator {
public:
    explicit RknnEvaluator(const std::string& path, const std::string& core) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot read RKNN file: " + path);
        std::vector<char> buffer{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
        if (buffer.empty()) throw std::runtime_error("RKNN model is empty");
        check(rknn_init(&handle_, buffer.data(), static_cast<uint32_t>(buffer.size()), 0, nullptr),
              "rknn_init");
        try {
            rknn_core_mask mask = core == "0" ? RKNN_NPU_CORE_0 :
                                  core == "1" ? RKNN_NPU_CORE_1 :
                                  core == "2" ? RKNN_NPU_CORE_2 : RKNN_NPU_CORE_0_1_2;
            check(rknn_set_core_mask(handle_, mask), "rknn_set_core_mask");
            rknn_input_output_num io{};
            check(rknn_query(handle_, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)), "RKNN_QUERY_IN_OUT_NUM");
            if (io.n_input != 1 || io.n_output != 2)
                throw std::runtime_error("expected one input and two outputs");
            rknn_tensor_attr in{};
            in.index = 0;
            check(rknn_query(handle_, RKNN_QUERY_INPUT_ATTR, &in, sizeof(in)),
                  "RKNN_QUERY_INPUT_ATTR");
            if (in.n_elems != 13)
                throw std::runtime_error("this 3x3 runner requires 13 input features");
            for (int i = 0; i < 2; ++i) {
                attrs_[i].index = i;
                check(rknn_query(handle_, RKNN_QUERY_OUTPUT_ATTR, &attrs_[i], sizeof(attrs_[i])),
                      "RKNN_QUERY_OUTPUT_ATTR");
                if (attrs_[i].n_elems == 10) policyIndex_ = i;
                if (attrs_[i].n_elems == 1) valueIndex_ = i;
            }
            if (policyIndex_ < 0 || valueIndex_ < 0)
                throw std::runtime_error("expected policy[10] and value[1]");
            rknn_sdk_version v{};
            if (rknn_query(handle_, RKNN_QUERY_SDK_VERSION, &v, sizeof(v)) == RKNN_SUCC)
                std::printf("RKNN Runtime: %s | Driver: %s\n", v.api_version, v.drv_version);
            std::printf("MODEL_STATUS input=13 policy=10 value=1 core=%s\n", core.c_str());
        } catch (...) {
            rknn_destroy(handle_);
            handle_ = 0;
            throw;
        }
    }
    ~RknnEvaluator() { if (handle_) rknn_destroy(handle_); }
    RknnEvaluator(const RknnEvaluator&) = delete;
    RknnEvaluator& operator=(const RknnEvaluator&) = delete;

    Prediction predict(const std::vector<float>& x) {
        if (x.size() != 13) throw std::runtime_error("bad Board::encode length");
        rknn_input input{};
        input.index = 0;
        input.buf = const_cast<float*>(x.data());
        input.size = static_cast<uint32_t>(x.size() * sizeof(float));
        input.type = RKNN_TENSOR_FLOAT32;
        // All spatial dimensions are 1, so flattened NCHW/NHWC order is identical.
        input.fmt = RKNN_TENSOR_NHWC;
        auto start = Clock::now();
        check(rknn_inputs_set(handle_, 1, &input), "rknn_inputs_set");
        check(rknn_run(handle_, nullptr), "rknn_run");
        rknn_output outputs[2]{};
        for (int i=0; i<2; ++i) {
            outputs[i].index = i;
            outputs[i].want_float = 1;
        }
        check(rknn_outputs_get(handle_, 2, outputs, nullptr), "rknn_outputs_get");
        Prediction pred;
        try {
            for (int i = 0; i < 2; ++i) {
                auto len = attrs_[i].n_elems;
                if (!outputs[i].buf || outputs[i].size < len * sizeof(float))
                    throw std::runtime_error("empty or malformed RKNN output");
                const auto* raw = static_cast<const float*>(outputs[i].buf);
                for (uint32_t j=0; j<len; ++j)
                    if (!std::isfinite(raw[j])) throw std::runtime_error("non-finite RKNN output");
                if (i == policyIndex_) pred.logits.assign(raw, raw + 10);
                else if (i == valueIndex_) pred.value = raw[0];
            }
        } catch (...) {
            rknn_outputs_release(handle_, 2, outputs);
            throw;
        }
        check(rknn_outputs_release(handle_, 2, outputs), "rknn_outputs_release");
        durationMs_ += std::chrono::duration<double,std::milli>(Clock::now()-start).count();
        ++evalCount_;
        return pred;
    }
    size_t evaluations() const { return evalCount_; }
    double totalMs() const { return durationMs_; }
private:
    rknn_context handle_ = 0;
    rknn_tensor_attr attrs_[2]{};
    int policyIndex_ = -1, valueIndex_ = -1;
    size_t evalCount_ = 0;
    double durationMs_ = 0;
};

static std::string sgfMove(int action) {
    if (action == 9) return "";
    std::string s;
    s.push_back(char('a' + action % 3));
    s.push_back(char('a' + action / 3));
    return s;
}

static std::string sgfResult(float score) {
    if (std::abs(score) < 1e-6f) return "0";
    std::ostringstream out;
    out << (score > 0 ? "B+" : "W+") << std::fixed << std::setprecision(1) << std::abs(score);
    return out.str();
}

static void writeSgf(const std::string& file, const std::vector<std::pair<int,int>>& moves,
                     float score) {
    std::ofstream out(file, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write SGF: " + file);
    out << "(;GM[1]FF[4]CA[UTF-8]SZ[3]KM[0.5]RU[Chinese]"
        << "PB[MiniGoCpp RKNN]PW[MiniGoCpp RKNN]RE[" << sgfResult(score) << "]";
    for (const auto& m : moves)
        out << ";" << (m.first == 1 ? 'B' : 'W') << "[" << sgfMove(m.second) << "]";
    out << ")\n";
    if (!out) throw std::runtime_error("failed to flush SGF: " + file);
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::fprintf(stderr, "Usage: %s my3.rknn [core=0] [sims=24] [games=1] [seed=42] [sgf_dir=sgf]\n", argv[0]);
            return 2;
        }
        const std::string model = argv[1], core = argc > 2 ? argv[2] : "0";
        const int sims = argc > 3 ? std::stoi(argv[3]) : 24;
        const int games = argc > 4 ? std::stoi(argv[4]) : 1;
        const int seed = argc > 5 ? std::stoi(argv[5]) : 42;
        const std::string sgfDir = argc > 6 ? argv[6] : "sgf";
        if ((core!="0" && core!="1" && core!="2" && core!="all") ||
            sims<2 || sims>10000 || games<1 || games>100 || sgfDir.empty())
            throw std::runtime_error("invalid core/sims/games/sgf dir");
        RknnEvaluator nn(model,core);
        NetworkEvaluator evaluate = [&nn](const std::vector<float>& input) { return nn.predict(input); };
        std::mt19937 rng(static_cast<uint32_t>(seed));
        int blacks = 0, whites = 0, draws = 0;
        std::printf("SELFPLAY_BEGIN board=3x3 sims=%d games=%d seed=%d core=%s\n",
                    sims, games, seed, core.c_str());
        for (int g=1; g<=games; ++g) {
            Board board(3, 0.5f);
            std::vector<std::pair<int,int>> history;
            auto gameStart = Clock::now();
            auto evalBefore = nn.evaluations();
            std::printf("\n=== GAME %d/%d ===\n%s", g, games, board.render().c_str());
            while (!board.terminal()) {
                int player = board.player();
                auto search = searchMCTS(board, evaluate, sims);
                // Same exploration policy as src/main.cpp::train:
                // sample during opening, greedy selection later.
                const bool explore = board.moveCount() < board.size()*board.size();
                int action = chooseMove(search.policy,rng,explore);
                if (!board.legal(action)) throw std::runtime_error("MCTS returned illegal action");
                std::printf("Move %2d %s %-4s rootValue=%+.4f MCTS=%d\n",
                            board.moveCount()+1, player==1?"B":"W",
                            board.moveName(action).c_str(), search.rootValue, sims);
                if (!board.play(action)) throw std::runtime_error("Board::play rejected action");
                history.emplace_back(player, action);
                std::printf("%s", board.render().c_str());
            }
            const float score = board.scoreBlack();
            if (score > 0) ++blacks; else if (score < 0) ++whites; else ++draws;
            char name[64];
            std::snprintf(name,sizeof(name),"/game_%03d.sgf",g);
            const std::string sgfPath = sgfDir + name;
            writeSgf(sgfPath,history,score);
            double gameMs = std::chrono::duration<double,std::milli>(Clock::now()-gameStart).count();
            std::printf("GAME_RESULT game=%d winner=%s scoreBlackMinusWhite=%.1f moves=%zu "
                        "nnEvals=%zu gameMs=%.3f sgf=%s\n",g,
                        score>0?"Black":score<0?"White":"Draw",score,history.size(),
                        nn.evaluations()-evalBefore,gameMs,sgfPath.c_str());
        }
        std::printf("SELFPLAY_DONE blackWins=%d whiteWins=%d draws=%d totalNNEvals=%zu "
                    "nnAvgMs=%.4f (includes RKNN input/output)\n", blacks,whites,draws,
                    nn.evaluations(), nn.evaluations()?nn.totalMs()/nn.evaluations():0.0);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr,"ERROR: %s\n",e.what());
        return 1;
    }
}
