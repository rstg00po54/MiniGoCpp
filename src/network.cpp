#include "network.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace {
void affine(const std::vector<float>& w, const std::vector<float>& b, const std::vector<float>& x, std::vector<float>& y) {
    int rows = int(b.size()), cols = int(x.size()); y.resize(rows);
    for (int r = 0; r < rows; ++r) {
        float s = b[r]; int offset = r * cols;
        for (int c = 0; c < cols; ++c) s += w[offset+c] * x[c];
        y[r] = s;
    }
}
void relu(const std::vector<float>& z, std::vector<float>& a) {
    a.resize(z.size());
    for (size_t i = 0; i < z.size(); ++i) a[i] = std::max(0.0f, z[i]);
}
void matrixGrad(const std::vector<float>& dy, const std::vector<float>& x, std::vector<float>& dw, std::vector<float>& db) {
    int cols = int(x.size());
    for (size_t r = 0; r < dy.size(); ++r) {
        db[r] += dy[r];
        for (int c = 0; c < cols; ++c) dw[r*cols+c] += dy[r] * x[c];
    }
}
void transposedMultiply(const std::vector<float>& w, const std::vector<float>& dy, std::vector<float>& dx) {
    int cols = int(dx.size());
    for (size_t r = 0; r < dy.size(); ++r) for (int c = 0; c < cols; ++c) dx[c] += w[r*cols+c] * dy[r];
}
}

Network::Network(int size, int hidden, uint32_t seed) : n_(size), h_(hidden), in_(size*size+4), out_(size*size+1) {
    if (size < 2 || size > 19 || hidden < 2 || hidden > 1024) throw std::invalid_argument("invalid network dimensions");
    auto init = [](Parameter& p, int count) { p.data.assign(count, 0.0f); p.m.assign(count, 0.0f); p.v.assign(count, 0.0f); };
    init(w0_, h_*in_); init(b0_, h_); init(w1_, h_*h_); init(b1_, h_);
    init(w2_, h_*h_); init(b2_, h_); init(wp_, out_*h_); init(bp_, out_);
    init(wv_, h_); init(bv_, 1);
    initialize(seed);
}

std::vector<Network::Parameter*> Network::parameters() { return {&w0_,&b0_,&w1_,&b1_,&w2_,&b2_,&wp_,&bp_,&wv_,&bv_}; }
std::vector<const Network::Parameter*> Network::parameters() const { return {&w0_,&b0_,&w1_,&b1_,&w2_,&b2_,&wp_,&bp_,&wv_,&bv_}; }
size_t Network::parameterCount() const { size_t total = 0; for (auto p : parameters()) total += p->data.size(); return total; }

void Network::initialize(uint32_t seed) {
    std::mt19937 rng(seed);
    auto randomMatrix = [&](Parameter& p, int in, int out) {
        float range = std::sqrt(6.0f / float(in + out));
        std::uniform_real_distribution<float> dist(-range, range);
        for (float& v : p.data) v = dist(rng);
    };
    randomMatrix(w0_, in_, h_); randomMatrix(w1_, h_, h_); randomMatrix(w2_, h_, h_);
    randomMatrix(wp_, h_, out_); randomMatrix(wv_, h_, 1);
}

Network::ForwardCache Network::forward(const std::vector<float>& input) const {
    if (int(input.size()) != in_) throw std::invalid_argument("input size mismatch");
    ForwardCache c;
    affine(w0_.data, b0_.data, input, c.z0); relu(c.z0, c.a0);
    affine(w1_.data, b1_.data, c.a0, c.z1); relu(c.z1, c.a1);
    affine(w2_.data, b2_.data, c.a1, c.z2);
    c.t.resize(h_);
    for (int i = 0; i < h_; ++i) c.t[i] = c.a0[i] + c.z2[i]; // handcrafted residual skip
    relu(c.t, c.h);
    affine(wp_.data, bp_.data, c.h, c.logits);
    float v = bv_.data[0]; for (int i = 0; i < h_; ++i) v += wv_.data[i] * c.h[i];
    c.value = std::tanh(v);
    return c;
}

Prediction Network::predict(const std::vector<float>& input) const {
    auto c = forward(input); return {std::move(c.logits), c.value};
}

std::vector<float> Network::probabilities(const std::vector<float>& logits, const std::vector<uint8_t>& legal) const {
    if (int(logits.size()) != out_ || legal.size() != logits.size()) throw std::invalid_argument("policy size mismatch");
    float mx = -1e30f;
    for (int i = 0; i < out_; ++i) if (legal[i]) mx = std::max(mx, logits[i]);
    if (mx < -1e29f) throw std::invalid_argument("no legal moves");
    std::vector<float> p(out_, 0.0f); float z = 0;
    for (int i = 0; i < out_; ++i) if (legal[i]) { p[i] = std::exp(logits[i] - mx); z += p[i]; }
    for (float& v : p) v /= z;
    return p;
}

float Network::train(const TrainingSample& s, float lr) {
    if (s.policy.size() != size_t(out_) || s.legal.size() != size_t(out_)) throw std::invalid_argument("invalid training sample");
    auto c = forward(s.input);
    auto prob = probabilities(c.logits, s.legal);
    float loss = 0.5f * (c.value - s.outcome) * (c.value - s.outcome);
    std::vector<float> dPolicy(out_);
    for (int i = 0; i < out_; ++i) {
        if (s.policy[i] > 0) loss -= s.policy[i] * std::log(std::max(prob[i], 1e-12f));
        dPolicy[i] = prob[i] - s.policy[i];
    }
    auto params = parameters();
    std::vector<std::vector<float>> g; g.reserve(params.size());
    for (auto p : params) g.emplace_back(p->data.size(), 0.0f);
    auto& gw0=g[0]; auto& gb0=g[1]; auto& gw1=g[2]; auto& gb1=g[3]; auto& gw2=g[4];
    auto& gb2=g[5]; auto& gwp=g[6]; auto& gbp=g[7]; auto& gwv=g[8]; auto& gbv=g[9];

    // Policy head: cross-entropy gradient (masked softmax).
    matrixGrad(dPolicy, c.h, gwp, gbp);
    std::vector<float> dh(h_, 0.0f);
    transposedMultiply(wp_.data, dPolicy, dh);

    // Value head: 0.5*(tanh(z) - outcome)^2.
    float dv = (c.value - s.outcome) * (1.0f - c.value*c.value);
    gbv[0] = dv;
    for (int i = 0; i < h_; ++i) { gwv[i] = dv * c.h[i]; dh[i] += wv_.data[i] * dv; }

    // Residual block: h = relu(a0 + W2*relu(W1*a0 + b1) + b2).
    std::vector<float> dt(h_), da0(h_), da1(h_,0.0f), dz1(h_);
    for (int i = 0; i < h_; ++i) { dt[i] = c.t[i] > 0 ? dh[i] : 0.0f; da0[i] = dt[i]; }
    matrixGrad(dt, c.a1, gw2, gb2);
    transposedMultiply(w2_.data, dt, da1);
    for (int i = 0; i < h_; ++i) dz1[i] = c.z1[i] > 0 ? da1[i] : 0.0f;
    matrixGrad(dz1, c.a0, gw1, gb1);
    transposedMultiply(w1_.data, dz1, da0);
    std::vector<float> dz0(h_);
    for (int i = 0; i < h_; ++i) dz0[i] = c.z0[i] > 0 ? da0[i] : 0.0f;
    matrixGrad(dz0, s.input, gw0, gb0);

    // Handwritten Adam optimizer, updating all tensors AFTER backprop.
    ++step_;
    float corr1 = 1.0f - std::pow(0.9f, float(step_));
    float corr2 = 1.0f - std::pow(0.999f, float(step_));
    for (size_t j = 0; j < params.size(); ++j) {
        Parameter& p = *params[j];
        for (size_t i = 0; i < p.data.size(); ++i) {
            float grad = std::clamp(g[j][i], -5.0f, 5.0f);
            p.m[i] = 0.9f * p.m[i] + 0.1f * grad;
            p.v[i] = 0.999f * p.v[i] + 0.001f * grad * grad;
            p.data[i] -= lr * (p.m[i] / corr1) / (std::sqrt(p.v[i] / corr2) + 1e-8f);
        }
    }
    return loss;
}

bool Network::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary); if (!f) return false;
    f.write("MGNET001", 8);
    uint32_t dims[2] = {uint32_t(n_), uint32_t(h_)};
    f.write(reinterpret_cast<const char*>(dims), sizeof(dims));
    f.write(reinterpret_cast<const char*>(&step_), sizeof(step_));
    for (auto p : parameters()) {
        for (const auto* v : {&p->data, &p->m, &p->v}) f.write(reinterpret_cast<const char*>(v->data()), std::streamsize(v->size()*sizeof(float)));
    }
    return bool(f);
}

Network Network::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary); if (!f) throw std::runtime_error("cannot open model: " + path);
    char magic[8]; f.read(magic, 8);
    if (!f || std::string(magic,8) != "MGNET001") throw std::runtime_error("invalid model header");
    uint32_t dims[2]{}; uint64_t step = 0;
    f.read(reinterpret_cast<char*>(dims), sizeof(dims)); f.read(reinterpret_cast<char*>(&step), sizeof(step));
    if (!f || dims[0] < 2 || dims[0] > 19 || dims[1] < 2 || dims[1] > 1024) throw std::runtime_error("invalid model dimensions");
    Network net{static_cast<int>(dims[0]), static_cast<int>(dims[1]), 1u}; net.step_ = step;
    for (auto p : net.parameters()) {
        for (auto* v : {&p->data, &p->m, &p->v}) f.read(reinterpret_cast<char*>(v->data()), std::streamsize(v->size()*sizeof(float)));
    }
    if (!f) throw std::runtime_error("model truncated");
    return net;
}
