// Headless test of the GE core (no Qt). Build:
//   g++ -std=c++20 -O2 -Isrc tests/ge_core_test.cpp src/core/ge/*.cpp -o ge_core_test
#include "core/ge/GEEngine.h"
#include "core/ge/NetSpec.h"
#include <cassert>
#include <cstdio>
using namespace NeuralStudio::ge;

static int fails = 0;
#define CHECK(c) do { if(!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while(0)

int main() {
    // 1. parser: errors
    for (const char* bad : {"", "<a> ::= <b>", "<a> ::= <a>", "junk line", "<a> ::= x\n<a> ::= y"}) {
        bool threw = false;
        try { Grammar::parse(bad); } catch (const std::exception&) { threw = true; }
        CHECK(threw);
    }
    // 2. parser: default grammar, round trip
    Grammar g = Grammar::parse(defaultUnifiedGrammar(5));
    CHECK(g.ntIndex("layer") >= 0 && g.ntIndex("feats") >= 0);
    CHECK(g.rules()[g.ntIndex("feats")].prods.size() == 3 && g.ntIndex("feat") >= 0);
    CHECK(g.rules()[g.ntIndex("feats")].prods[0].syms.empty());   // empty alternative
    CHECK(g.rules()[g.ntIndex("var")].prods.size() == 5);

    // 3. classic mapping example from the BNF paper style: deterministic
    Grammar g2 = Grammar::parse("<S> ::= <a><b>\n<a> ::= x | y | z\n<b> ::= 0 | 1");
    std::vector<int> genes = {4, 3};            // 4%3=1 -> y ; 3%2=1 -> 1
    CodonSource cs(genes, 0);
    Node t; Grammar::DeriveOptions o;
    CHECK(g2.derive(cs, t, o));
    CHECK(Grammar::text(t) == "y1");
    std::vector<int> few = {4};                 // second codon missing, no wraps
    CodonSource cs2(few, 0);
    CHECK(!g2.derive(cs2, t, o));
    CodonSource cs3(few, 1);                    // wrapping: 4%3=1, then 4%2=0
    CHECK(g2.derive(cs3, t, o));
    CHECK(Grammar::text(t) == "y0");

    // 4. interpreter
    InterpretOptions io; io.nInputs = 5; io.maxLayers = 3; io.maxFeatures = 2;
    Grammar g3 = Grammar::parse(
        "<S> ::= <arch> <feats>\n<arch> ::= <layer><layer><layer><layer><out>\n"
        "<layer> ::= L(<size>,<act>,<cmd>,<cmd>)\n<out> ::= O(<cmd>,<cmd>)\n"
        "<size> ::= 16\n<act> ::= tanh\n<cmd> ::= SHRINK_SHRINK | EXPAND\n"
        "<feats> ::= <feat><feat><feat>\n<feat> ::= F(<expr>)\n<expr> ::= (x1*x2) | sin(x9) | log(x3)\n");
    std::vector<int> gz(60, 0);
    CodonSource cz(gz, 0);
    CHECK(g3.derive(cz, t, o));
    NetSpec sp = interpret(g3, t, io);
    CHECK(sp.hidden.size() == 3);                  // 4 layers derived, capped to 3
    CHECK(sp.hidden[0].lbFactor == 0.25 && sp.hidden[0].size == 16 && sp.hidden[0].act == "tanh");
    CHECK(sp.outLbFactor == 0.25);
    CHECK(sp.features.size() == 1);                // (x1*x2) kept; sin(x9) invalid (only 5 inputs); duplicates dropped
    std::printf("spec: %s\n", sp.describe().c_str());

    CHECK(Expr::parse("cos(3)", 5) && !Expr::parse("cos(3)", 5)->usesVariable());
    // constant folding and size cap
    { auto f = Expr::parse("log(exp(log(sin(3))))+x1", 2); CHECK(f && f->size() == 3 && f->usesVariable());
      auto big = Expr::parse("((x1+x2)*(x1-x2))+((x1*x2)+(x1+x1))", 2); CHECK(big && big->size() > 8);
      double v[2] = {1.5, 0.5};
      auto g1 = Expr::parse("(2*3)+x1", 2); CHECK(g1 && g1->size() == 3 && std::fabs(g1->eval(v) - 7.5) < 1e-12); }
    // 5. expressions
    auto e = Expr::parse("(x1*x2)+sin(x3)/2", 3);
    CHECK(e != nullptr);
    double x[3] = {2, 3, 0};
    CHECK(std::fabs(e->eval(x) - 6.0) < 1e-12);
    CHECK(Expr::parse("x4", 3) == nullptr);
    CHECK(Expr::parse("(x1+", 3) == nullptr);
    double z[3] = {1, 0, 0};
    auto dz = Expr::parse("x1/x2", 3);
    CHECK(std::isfinite(dz->eval(z)));

    // 6. engine: both genotypes should reduce a synthetic fitness
    //    target: 2 layers, sizes 64 and 32, activation tanh, no features
    Grammar gd = Grammar::parse(defaultUnifiedGrammar(5));
    InterpretOptions io2; io2.nInputs = 5;
    auto fit = [&](const Node& tree) {
        NetSpec s = interpret(gd, tree, io2);
        if (!s.valid()) return std::numeric_limits<double>::infinity();
        double f = 0;
        f += std::fabs(static_cast<double>(s.hidden.size()) - 2.0) * 10;
        for (size_t i = 0; i < s.hidden.size(); ++i) {
            f += std::fabs(s.hidden[i].size - (i == 0 ? 64.0 : 32.0)) * 0.1;
            f += s.hidden[i].act == "tanh" ? 0.0 : 3.0;
            f += std::fabs(s.hidden[i].lbFactor - 0.5) + std::fabs(s.hidden[i].ubFactor - 1.5);
        }
        f += s.features.size() * 2.0;
        return f;
    };
    for (auto kind : {Genotype::Codons, Genotype::Probabilistic}) {
        double sumFirst = 0, sumLast = 0; const int R = 5;
        for (int r = 0; r < R; ++r) {
            GEConfig c; c.genotype = kind; c.popSize = 40; c.seed = 100 + r;
            c.replaceRate = 0.3;
            GEEngine eng(gd, c, fit);
            eng.initPopulation();
            sumFirst += eng.best().fitness;
            for (int k = 0; k < 60; ++k) eng.step();
            sumLast += eng.best().fitness;
            if (r == 0) {
                Node tr; CHECK(eng.derive(eng.best(), tr));
                std::printf("%s best: f=%.3f  %s  (evals=%ld, diversity=%.2f)\n",
                    kind == Genotype::Codons ? "GE " : "PGE", eng.best().fitness,
                    interpret(gd, tr, io2).describe().c_str(), eng.evaluations(), eng.diversity());
            }
        }
        std::printf("%s mean best: init %.3f -> final %.3f\n",
            kind == Genotype::Codons ? "GE " : "PGE", sumFirst / R, sumLast / R);
        CHECK(sumLast < sumFirst);
    }
    // 7. local crossover runs
    { GEConfig c; c.crossover = Crossover::Local; c.localRate = 0.3; c.popSize = 20; c.seed = 7;
      GEEngine eng(gd, c, fit); eng.initPopulation(); for (int k = 0; k < 10; ++k) eng.step();
      CHECK(std::isfinite(eng.best().fitness)); }

    std::printf(fails ? "\n%d FAILURES\n" : "\nALL OK\n", fails);
    return fails ? 1 : 0;
}
