#include "board.h"
#include "network.h"
#include "search.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    {
        Board b(3,0.5f);
        assert(b.legal(b.parseMove("B2")));
        assert(!b.legal(-1));
        assert(b.play(b.parseMove("B2")));
        assert(!b.play(b.parseMove("B2")));
        assert(b.stoneAt(1,1)==1);
        assert(b.play(b.passAction()));
        assert(b.play(b.passAction()));
        assert(b.terminal());
        assert(!b.play(b.passAction()));
        assert(std::abs(b.scoreBlack()-8.5f)<0.001f);
    }
    {
        Board b(3);
        assert(b.play(b.parseMove("A2"))); // black
        assert(b.play(b.parseMove("B2"))); // white
        assert(b.play(b.parseMove("B1")));
        assert(b.play(b.passAction()));
        assert(b.play(b.parseMove("C2")));
        assert(b.play(b.passAction()));
        assert(b.play(b.parseMove("B3"))); // captures B2
        assert(b.stoneAt(1,1)==0);
    }
    {
        Board b(3);
        assert(b.play(b.parseMove("B1")));
        assert(b.play(b.parseMove("C3")));
        assert(b.play(b.parseMove("A2")));
        assert(!b.play(b.parseMove("A1"))); // suicide
        assert(b.stoneAt(0,0)==0);
    }
    {
        Board b(5);
        assert(b.play(b.parseMove("B3"))); // black surround
        assert(b.play(b.parseMove("D2"))); // white surround
        assert(b.play(b.parseMove("C2")));
        assert(b.play(b.parseMove("D4")));
        assert(b.play(b.parseMove("C4")));
        assert(b.play(b.parseMove("E3")));
        assert(b.play(b.parseMove("A1")));
        assert(b.play(b.parseMove("C3"))); // white has one liberty at D3
        assert(b.play(b.parseMove("D3"))); // black captures C3, makes a ko
        assert(b.stoneAt(2,2)==0);
        assert(!b.play(b.parseMove("C3"))); // positional superko forbids immediate retake
        assert(b.stoneAt(3,2)==1);
    }
    {
        Network n(3,16,42); Board b(3);
        assert(n.parameterCount()>800);
        auto r=searchMCTS(b,n,16);
        float s=0; for(float x:r.policy) s+=x;
        assert(std::abs(s-1.0f)<0.001f);
        TrainingSample t; t.input=b.encode(); t.policy.assign(10,0); t.policy[4]=1;
        t.legal.assign(10,1); t.outcome=1;
        auto before=n.predict(t.input);
        float pBefore=n.probabilities(before.logits,t.legal)[4];
        for (int i=0; i<120; ++i) n.train(t,0.003f);
        auto after=n.predict(t.input);
        float pAfter=n.probabilities(after.logits,t.legal)[4];
        assert(pAfter>pBefore);
        assert(after.value>before.value);
        assert(n.save("/tmp/minigo_test.bin"));
        auto restored=Network::load("/tmp/minigo_test.bin");
        auto same=restored.predict(t.input);
        assert(std::abs(same.value-after.value)<0.000001f);
        assert(std::abs(same.logits[4]-after.logits[4])<0.000001f);
    }
    std::cout << "PASS: moves, capture, suicide, superko, double pass, scoring, MCTS, backprop, save/load\n";
}
