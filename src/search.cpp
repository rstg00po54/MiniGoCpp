#include "search.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace {
struct Node;
struct Child {
    int action;
    float prior;
    std::unique_ptr<Node> node;
};
struct Node {
    explicit Node(const Board& b) : board(b) {}
    Board board;
    int visits = 0;
    double valueSum = 0.0;
    bool expanded = false;
    std::vector<Child> children;
};

double simulate(Node& node, const Network& net) {
    if (node.board.terminal()) {
        double result = node.board.resultFor(node.board.player());
        node.visits++; node.valueSum += result;
        return result;
    }
    if (!node.expanded) {
        auto pred = net.predict(node.board.encode());
        std::vector<uint8_t> legal(node.board.actionCount(), 0);
        auto actions = node.board.legalActions();
        for (int a : actions) legal[a] = 1;
        auto priors = net.probabilities(pred.logits, legal);
        for (int a : actions) node.children.push_back({a, priors[a], nullptr});
        node.expanded = true; node.visits++; node.valueSum += pred.value;
        return pred.value;
    }
    Child* best = nullptr; double bestScore = -1e100;
    for (auto& child : node.children) {
        int visited = child.node ? child.node->visits : 0;
        double q = visited ? -child.node->valueSum / visited : 0.0;
        double u = 1.5 * child.prior * std::sqrt(double(node.visits)) / (visited+1);
        if (q + u > bestScore) { bestScore = q+u; best = &child; }
    }
    if (!best) throw std::runtime_error("MCTS node has no children");
    if (!best->node) {
        Board next = node.board;
        if (!next.play(best->action)) throw std::runtime_error("MCTS illegal child move");
        best->node = std::make_unique<Node>(next);
    }
    double value = -simulate(*best->node, net);
    node.visits++; node.valueSum += value;
    return value;
}
}

SearchResult searchMCTS(const Board& board, const Network& net, int simulations) {
    if (board.terminal()) throw std::invalid_argument("cannot search terminal board");
    if (net.size() != board.size()) throw std::invalid_argument("model/board mismatch");
    Node root(board);
    for (int i = 0; i < std::max(simulations, 2); ++i) simulate(root, net);
    SearchResult r; r.policy.assign(board.actionCount(), 0.0f);
    int sum = 0;
    for (auto& c : root.children) if (c.node) sum += c.node->visits;
    if (sum > 0) {
        for (auto& c : root.children) if (c.node) r.policy[c.action] = float(c.node->visits) / sum;
    } else {
        for (auto& c : root.children) r.policy[c.action] = c.prior;
    }
    r.rootValue = float(root.valueSum / root.visits);
    return r;
}

int chooseMove(const std::vector<float>& policy, std::mt19937& rng, bool sample) {
    if (sample) {
        std::discrete_distribution<int> dist(policy.begin(), policy.end());
        return dist(rng);
    }
    return int(std::max_element(policy.begin(), policy.end()) - policy.begin());
}
