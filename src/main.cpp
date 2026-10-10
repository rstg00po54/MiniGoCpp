#include "board.h"
#include "network.h"
#include "search.h"
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

namespace {
struct Options {
    std::map<std::string, std::string> args;
    Options(int argc, char** argv) {
        for (int i = 2; i < argc; ++i) {
            std::string key = argv[i];
            if (key.rfind("--",0) != 0 || i+1 >= argc) throw std::invalid_argument("expected --name value");
            args[key] = argv[++i];
        }
    }
    std::string get(const std::string& key, const std::string& def) const { auto it=args.find(key); return it==args.end()?def:it->second; }
    int number(const std::string& key, int def) const { return std::stoi(get(key,std::to_string(def))); }
    float real(const std::string& key, float def) const { return std::stof(get(key,std::to_string(def))); }
};

void usage() {
    std::cout << "MiniGoCpp: handcrafted C++ Go engine + residual neural network + backprop + MCTS\n\n"
              << "  ./minigo train --size 3 --games 100 --sims 24 --hidden 32 --epochs 2 --out models/go3.bin\n"
              << "  ./minigo train --size 3 --games 100 --sims 24 --model models/go3.bin --out models/go3.bin\n"
              << "  ./minigo play  --model models/go3.bin --sims 80\n"
              << "  ./minigo eval  --model models/go3.bin --games 100 --sims 40\n"
              << "  ./minigo inspect --model models/go3.bin\n\n"
              << "No model? Use --size 3 --hidden 32 for a random (untrained) network.\n"
              << "Coordinates: A1 at top left; 'pass' to skip; 'quit' to exit.\n"
              << "Uses Chinese-style AREA scoring, 0.5 komi by default, positional superko, no suicide.\n"
              << "Training uses MCTS visit policy and final win/loss target, not human examples.\n";
}

Network makeNetwork(const Options& opts) {
    if (opts.args.count("--model")) return Network::load(opts.get("--model", ""));
    return Network(opts.number("--size",3), opts.number("--hidden",32), uint32_t(opts.number("--seed",42)));
}

TrainingSample record(const Board& b, const SearchResult& result) {
    TrainingSample s; s.player = b.player(); s.input = b.encode(); s.policy = result.policy;
    s.legal.assign(b.actionCount(), 0);
    for (int a : b.legalActions()) s.legal[a] = 1;
    return s;
}

void train(const Options& o) {
    Network net = makeNetwork(o);
    int games = o.number("--games",100), sims=o.number("--sims",24), epochs=o.number("--epochs",2);
    float komi = o.real("--komi",0.5f), lr=o.real("--lr",0.001f);
    std::string file = o.get("--out", "models/go" + std::to_string(net.size()) + ".bin");
    if (games<=0 || sims<2 || epochs<1 || lr<=0) throw std::invalid_argument("invalid train arguments");
    std::mt19937 rng(uint32_t(o.number("--seed",42)));
    std::cout << "Train " << net.size() << "x" << net.size() << ", hidden=" << net.hidden()
              << ", params=" << net.parameterCount() << ", games=" << games << ", sims=" << sims << '\n';
    std::vector<TrainingSample> samples;
    int black=0, white=0, draw=0, positions=0;
    for (int game=1; game<=games; ++game) {
        Board board(net.size(),komi);
        std::vector<TrainingSample> episode;
        while (!board.terminal()) {
            auto result = searchMCTS(board, net, sims);
            episode.push_back(record(board,result));
            bool explore = board.moveCount() < net.size()*net.size();
            int action = chooseMove(result.policy,rng,explore);
            if (!board.play(action)) throw std::runtime_error("chosen move was illegal");
        }
        int winner=board.winner();
        if (winner==1) ++black; else if (winner==-1) ++white; else ++draw;
        for (auto& s : episode) { s.outcome=float(winner*s.player); samples.push_back(std::move(s)); }
        positions += int(episode.size());
        if (game%10==0 || game==games) {
            double loss = 0.0; long updates=0;
            for (int epoch=0; epoch<epochs; ++epoch) {
                std::shuffle(samples.begin(),samples.end(),rng);
                for (const auto& s : samples) { loss += net.train(s,lr); ++updates; }
            }
            std::filesystem::path path(file);
            if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
            if (!net.save(file)) throw std::runtime_error("unable to save model: " + file);
            std::cout << "games=" << std::setw(4) << game << '/' << games
                      << "  positions=" << positions << "  recent_loss=" << std::fixed << std::setprecision(3) << (loss/updates)
                      << "  B/W/D=" << black << '/' << white << '/' << draw << "  saved=" << file << '\n';
            samples.clear();
        }
    }
    std::cout << "Done. Model saved to " << file << '\n';
}

int randomMove(const Board& b, std::mt19937& rng) {
    auto actions = b.legalActions();
    std::uniform_int_distribution<size_t> dist(0,actions.size()-1);
    return actions[dist(rng)];
}

void evaluate(const Options& o) {
    Network net=makeNetwork(o);
    int games=o.number("--games",100), sims=o.number("--sims",40);
    if (games<=0 || sims<2) throw std::invalid_argument("invalid eval arguments");
    std::mt19937 rng(uint32_t(o.number("--seed",1234)));
    int wins=0, losses=0, draws=0;
    for (int i=0; i<games; ++i) {
        Board b(net.size(),o.real("--komi",0.5f));
        int ai = i%2==0 ? 1 : -1;
        while (!b.terminal()) {
            int a = b.player()==ai ? chooseMove(searchMCTS(b,net,sims).policy,rng,false) : randomMove(b,rng);
            if (!b.play(a)) throw std::runtime_error("evaluation move illegal");
        }
        int result=b.winner()*ai;
        if (result>0) ++wins; else if (result<0) ++losses; else ++draws;
    }
    std::cout << "vs random, both colors: " << wins << " wins / " << losses << " losses / " << draws
              << " draws (" << std::fixed << std::setprecision(1) << (100.0*(wins+0.5*draws)/games) << "% score)\n";
}

void inspect(const Options& o) {
    Network net=makeNetwork(o);
    Board b(net.size(),o.real("--komi",0.5f));
    auto prediction=net.predict(b.encode());
    std::vector<uint8_t> mask(b.actionCount(), 1);
    auto probs=net.probabilities(prediction.logits,mask);
    std::vector<int> actions(b.actionCount());
    std::iota(actions.begin(), actions.end(), 0);
    std::sort(actions.begin(),actions.end(),[&](int a,int b){return probs[a]>probs[b];});
    int h=net.hidden(), ni=net.inputCount(), no=net.outputCount();
    std::cout << "Network " << net.size() << "x" << net.size() << " | input=" << ni
              << " | hidden=" << h << " | policy=" << no << " | value=1\n"
              << "Architecture: input -> dense+ReLU -> [dense+ReLU -> dense -> add(skip) -> ReLU] -> policy/value\n"
              << "Parameter count: " << net.parameterCount() << " = "
              << (h*ni+h) << " input + " << (2*h*h+2*h) << " residual + "
              << (no*h+no) << " policy + " << (h+1) << " value\n"
              << "Empty board network value (before search): " << prediction.value << "\n"
              << "Highest raw policy probabilities (before search):\n";
    for (int i=0; i<std::min(no,10); ++i) {
        int a=actions[i];
        std::cout << "  " << std::setw(5) << b.moveName(a) << ": "
                  << std::fixed << std::setprecision(2) << probs[a]*100 << "%\n";
    }
}

void interactive(const Options& o) {
    Network net=makeNetwork(o);
    int sims=o.number("--sims",80), human=o.number("--human",1);
    if (sims<2 || (human!=1 && human!=-1)) throw std::invalid_argument("--human must be 1 for black or -1 for white");
    Board b(net.size(),o.real("--komi",0.5f));
    std::mt19937 rng(uint32_t(o.number("--seed",42)));
    std::cout << "You play " << (human==1?"black (X)":"white (O)") << ". Type A1, B2, pass, or quit.\n";
    while (!b.terminal()) {
        std::cout << b.render();
        int a=-1;
        if (b.player()==human) {
            std::string text;
            std::cout << "Your move: ";
            if (!std::getline(std::cin,text) || text=="quit" || text=="q") return;
            a=b.parseMove(text);
            if (!b.legal(a)) { std::cout << "Illegal move. Try again.\n"; continue; }
        } else {
            auto r=searchMCTS(b,net,sims);
            a=chooseMove(r.policy,rng,false);
            std::cout << "AI chooses " << b.moveName(a) << " (search value " << std::setprecision(2) << r.rootValue << ")\n";
        }
        if (!b.play(a)) throw std::runtime_error("unexpected illegal move");
    }
    std::cout << b.render() << "End: black-minus-white-minus-komi=" << b.scoreBlack()
              << ", winner=" << (b.winner()==1?"black":b.winner()==-1?"white":"draw") << '\n';
}
}

int main(int argc, char** argv) {
    try {
        if (argc<2) { usage(); return 0; }
        std::string command=argv[1];
        if (command=="help" || command=="--help") { usage(); return 0; }
        Options o(argc,argv);
        if (command=="train") train(o);
        else if (command=="play") interactive(o);
        else if (command=="eval") evaluate(o);
        else if (command=="inspect") inspect(o);
        else { usage(); return 1; }
    } catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << '\n'; return 1; }
    return 0;
}
