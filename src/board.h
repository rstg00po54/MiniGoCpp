#pragma once
#include <cstdint>
#include <string>
#include <vector>

class Board {
public:
    explicit Board(int size = 3, float komi = 0.5f);
    int size() const { return n_; }
    int player() const { return toMove_; }
    int moveCount() const { return moves_; }
    int passCount() const { return passes_; }
    int maxMoves() const { return 4 * n_ * n_; }
    int actionCount() const { return n_ * n_ + 1; }
    int passAction() const { return n_ * n_; }
    float komi() const { return komi_; }
    bool terminal() const { return passes_ >= 2 || moves_ >= maxMoves(); }
    bool play(int action);
    bool legal(int action) const;
    std::vector<int> legalActions() const;
    std::vector<float> encode() const;
    float scoreBlack() const;
    int winner() const;
    float resultFor(int player) const;
    std::string moveName(int action) const;
    int parseMove(const std::string& text) const;
    std::string render() const;
    int stoneAt(int x, int y) const { return cells_[y*n_ + x]; }
private:
    int n_, toMove_ = 1, passes_ = 0, moves_ = 0;
    float komi_;
    std::vector<int8_t> cells_;
    std::vector<std::string> history_;
    std::string key() const;
    std::vector<int> neighbors(int pos) const;
    void group(int pos, std::vector<int>& stones, int& liberties) const;
};
