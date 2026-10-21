#pragma once
#include "board.h"
#include "network.h"
#include <functional>
#include <random>
#include <vector>

struct SearchResult {
    std::vector<float> policy;
    float rootValue = 0.0f;
};

SearchResult searchMCTS(const Board& board, const Network& net, int simulations);
int chooseMove(const std::vector<float>& policy, std::mt19937& rng, bool sample);

// Device-neutral policy/value callback: permits RKNN inference without changing MCTS.
using NetworkEvaluator = std::function<Prediction(const std::vector<float>&)>;
SearchResult searchMCTS(const Board& board, const NetworkEvaluator& evaluator, int simulations);
