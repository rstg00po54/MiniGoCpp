#pragma once
#include "board.h"
#include "network.h"
#include <random>
#include <vector>

struct SearchResult {
    std::vector<float> policy;
    float rootValue = 0.0f;
};

SearchResult searchMCTS(const Board& board, const Network& net, int simulations);
int chooseMove(const std::vector<float>& policy, std::mt19937& rng, bool sample);
