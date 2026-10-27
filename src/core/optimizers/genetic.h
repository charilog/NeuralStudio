#pragma once
#include "optimizer.h"

namespace optimsolution {

// ─── GeneticAlgorithm ─────────────────────────────────────────────────────────
//  Classic elitist genetic algorithm for continuous optimisation.
//
//  Operators:
//    Selection  : tournament  (k = 3 by default)
//    Crossover  : uniform      (each gene from P1 or P2 with p=0.5)
//    Mutation   : Gaussian     (N(0,σ) added with probability m_mutRate)
//    Replacement: generational with elitism
//
//  Configuration (via MethodConfig):
//    "crossover_rate"  double  probability of crossover vs cloning (default 0.8)
//    "mutation_rate"   double  per-gene mutation probability       (default 0.05)
//    "mutation_sigma"  double  std-dev of Gaussian mutation noise  (default 0.2)
//    "elite"           int     number of elites preserved          (default 2)
//    "tournament_k"    int     tournament size                      (default 3)
// ─────────────────────────────────────────────────────────────────────────────
class GeneticAlgorithm : public Optimizer {
public:
    std::string methodShortName() const override { return "GA"; }
    std::string methodFullName()  const override { return "Genetic Algorithm"; }
    void configure(const MethodConfig& mc) override;

    void init()          override;
    void one_iteration() override;

private:
    // ── Hyper-parameters ─────────────────────────────────────────────────────
    double m_crossRate  = 0.80;
    double m_mutRate    = 0.05;
    double m_mutSigma   = 0.20;
    int    m_elite      = 2;
    int    m_tournK     = 3;

    // ── Population ───────────────────────────────────────────────────────────
    std::vector<Vec>    m_pop;      // [pop_][dim]
    std::vector<double> m_fit;      // fitness (loss) — lower is better
    std::vector<int>    m_rank;     // sorted indices (best first)

    // ── Genetic operators ─────────────────────────────────────────────────────
    int   tournamentSelect();
    Vec   uniformCrossover(const Vec& p1, const Vec& p2);
    void  gaussianMutate(Vec& x);
    void  clampBounds(Vec& x);
    void  updateRank();
};

} // namespace optimsolution
