#include "board.h"
#include <algorithm>
#include <cctype>
#include <iomanip>
#include <sstream>
#include <stdexcept>

Board::Board(int size, float komi) : n_(size), komi_(komi), cells_(size*size, 0) {
    if (size < 2 || size > 19) throw std::invalid_argument("board size must be 2..19");
    history_.push_back(key());
}

std::string Board::key() const {
    std::string s; s.reserve(cells_.size());
    for (auto v : cells_) s += char('1' + v);
    return s;
}

std::vector<int> Board::neighbors(int p) const {
    std::vector<int> r; int x = p % n_, y = p / n_;
    if (x > 0) r.push_back(p-1);
    if (x+1 < n_) r.push_back(p+1);
    if (y > 0) r.push_back(p-n_);
    if (y+1 < n_) r.push_back(p+n_);
    return r;
}

void Board::group(int pos, std::vector<int>& stones, int& liberties) const {
    stones.clear(); liberties = 0;
    if (cells_[pos] == 0) return;
    std::vector<uint8_t> seen(n_*n_, 0), seenLib(n_*n_, 0);
    stones.push_back(pos); seen[pos] = 1;
    for (size_t i = 0; i < stones.size(); ++i) {
        for (int q : neighbors(stones[i])) {
            if (cells_[q] == 0 && !seenLib[q]) { seenLib[q] = 1; ++liberties; }
            else if (cells_[q] == cells_[pos] && !seen[q]) { seen[q] = 1; stones.push_back(q); }
        }
    }
}

bool Board::play(int action) {
    if (terminal() || action < 0 || action > passAction()) return false;
    if (action == passAction()) { ++passes_; ++moves_; toMove_ = -toMove_; return true; }
    if (cells_[action] != 0) return false;
    auto old = cells_;
    cells_[action] = static_cast<int8_t>(toMove_);
    for (int q : neighbors(action)) {
        if (cells_[q] != -toMove_) continue;
        std::vector<int> stones; int libs;
        group(q, stones, libs);
        if (libs == 0) for (int p : stones) cells_[p] = 0;
    }
    std::vector<int> stones; int libs;
    group(action, stones, libs);
    if (libs == 0 || std::find(history_.begin(), history_.end(), key()) != history_.end()) {
        cells_ = std::move(old); return false;
    }
    history_.push_back(key());
    passes_ = 0; ++moves_; toMove_ = -toMove_;
    return true;
}

bool Board::legal(int action) const { Board b = *this; return b.play(action); }

std::vector<int> Board::legalActions() const {
    std::vector<int> result;
    if (terminal()) return result;
    for (int a = 0; a <= passAction(); ++a) if (legal(a)) result.push_back(a);
    return result;
}

std::vector<float> Board::encode() const {
    std::vector<float> x; x.reserve(n_*n_+4);
    for (int8_t cell : cells_) x.push_back(float(cell * toMove_));
    x.push_back(float(toMove_));
    x.push_back(komi_ * toMove_ / float(n_*n_));
    x.push_back(float(passes_));
    x.push_back(float(moves_) / float(maxMoves()));
    return x;
}

float Board::scoreBlack() const {
    int black = 0, white = 0;
    std::vector<uint8_t> seen(n_*n_, 0);
    for (int p = 0; p < n_*n_; ++p) {
        if (cells_[p] == 1) { ++black; continue; }
        if (cells_[p] == -1) { ++white; continue; }
        if (seen[p]) continue;
        std::vector<int> region{p}; seen[p] = 1; bool touchesBlack = false, touchesWhite = false;
        for (size_t i = 0; i < region.size(); ++i) {
            for (int q : neighbors(region[i])) {
                if (cells_[q] == 1) touchesBlack = true;
                else if (cells_[q] == -1) touchesWhite = true;
                else if (!seen[q]) { seen[q] = 1; region.push_back(q); }
            }
        }
        if (touchesBlack && !touchesWhite) black += int(region.size());
        if (touchesWhite && !touchesBlack) white += int(region.size());
    }
    return float(black - white) - komi_;
}

int Board::winner() const { float s = scoreBlack(); return s > 0 ? 1 : (s < 0 ? -1 : 0); }
float Board::resultFor(int player) const { return float(winner() * player); }

std::string Board::moveName(int action) const {
    if (action == passAction()) return "pass";
    if (action < 0 || action >= passAction()) return "?";
    return std::string(1, char('A' + action % n_)) + std::to_string(action / n_ + 1);
}

int Board::parseMove(const std::string& text) const {
    std::string s;
    for (char ch : text) if (!std::isspace(static_cast<unsigned char>(ch))) s += char(std::toupper(static_cast<unsigned char>(ch)));
    if (s == "PASS" || s == "P") return passAction();
    if (s.size() < 2) return -1;
    int x = s[0] - 'A', row = -1;
    try { row = std::stoi(s.substr(1)); } catch (...) { return -1; }
    if (x < 0 || x >= n_ || row < 1 || row > n_) return -1;
    return (row-1)*n_ + x;
}

std::string Board::render() const {
    std::ostringstream out;
    out << "   "; for (int x = 0; x < n_; ++x) out << char('A'+x) << ' ';
    out << '\n';
    for (int y = 0; y < n_; ++y) {
        out << std::setw(2) << (y+1) << ' ';
        for (int x = 0; x < n_; ++x) {
            int c = stoneAt(x,y); out << (c == 1 ? 'X' : (c == -1 ? 'O' : '.')) << ' ';
        }
        out << '\n';
    }
    out << "X=black O=white | next=" << (toMove_ == 1 ? "black" : "white") << " | passes=" << passes_ << '\n';
    return out.str();
}
