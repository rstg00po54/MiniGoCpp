#pragma once
#include "board.h"
#include <cstdint>
#include <random>
#include <string>
#include <vector>

struct TrainingSample {
    std::vector<float> input, policy;
    std::vector<uint8_t> legal;
    int player = 1;
    float outcome = 0.0f;
};

struct Prediction {
    std::vector<float> logits;
    float value = 0.0f;
};

class Network {
public:
    Network(int size, int hidden, uint32_t seed);
    Prediction predict(const std::vector<float>& input) const;
    std::vector<float> probabilities(const std::vector<float>& logits, const std::vector<uint8_t>& legal) const;
    float train(const TrainingSample& sample, float learningRate);
    bool save(const std::string& path) const;
    static Network load(const std::string& path);
    int size() const { return n_; }
    int hidden() const { return h_; }
    int inputCount() const { return in_; }
    int outputCount() const { return out_; }
    size_t parameterCount() const;
private:
    struct Parameter { std::vector<float> data, m, v; };
    struct ForwardCache {
        std::vector<float> z0, a0, z1, a1, z2, t, h, logits;
        float value = 0.0f;
    };
    int n_, h_, in_, out_;
    uint64_t step_ = 0;
    Parameter w0_, b0_, w1_, b1_, w2_, b2_, wp_, bp_, wv_, bv_;
    void initialize(uint32_t seed);
    ForwardCache forward(const std::vector<float>& input) const;
    std::vector<Parameter*> parameters();
    std::vector<const Parameter*> parameters() const;
};
