#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  init.h  —  Population initialiser shim.
//
//  The optimsolution optimizer .cpp files do:
//    Initializer initSampler;
//    initSampler.configure(initopt_);
//    X_ = initSampler.samplePopulation(*prob_, rng_, pop_);
//
//  In NeuralStudio we always use uniform random in the problem's bounds.
// ─────────────────────────────────────────────────────────────────────────────
#include "optimizer.h"

namespace optimsolution {

class Initializer {
public:
    void configure(const InitOptions&) {}   // no-op

    std::vector<std::vector<double>>
    samplePopulation(Problem& prob, std::mt19937& rng, int n) {
        const int D       = prob.dimension();
        const auto& lb    = prob.lb();
        const auto& ub    = prob.ub();

        std::vector<std::vector<double>> pop(n, std::vector<double>(D));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < D; ++j) {
                std::uniform_real_distribution<double> u(lb[j], ub[j]);
                pop[i][j] = u(rng);
            }
        return pop;
    }
};

} // namespace optimsolution
