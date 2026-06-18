#include "genetic.h"
#include "init.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace optimsolution {

void GeneticAlgorithm::configure(const MethodConfig& mc) {
    m_crossRate = mc.getDbl("crossover_rate", 0.80);
    m_mutRate   = mc.getDbl("mutation_rate",  0.05);
    m_mutSigma  = mc.getDbl("mutation_sigma", 0.20);
    m_elite     = mc.getInt("elite",          2);
    m_tournK    = mc.getInt("tournament_k",   3);
}

// ─── Genetic operators ────────────────────────────────────────────────────────
void GeneticAlgorithm::updateRank() {
    m_rank.resize(m_pop.size());
    std::iota(m_rank.begin(), m_rank.end(), 0);
    std::sort(m_rank.begin(), m_rank.end(),
              [&](int a, int b){ return m_fit[a] < m_fit[b]; });
}

int GeneticAlgorithm::tournamentSelect() {
    std::uniform_int_distribution<int> pick(0, static_cast<int>(m_pop.size())-1);
    int best = pick(rng_);
    for (int i = 1; i < m_tournK; ++i) {
        int c = pick(rng_);
        if (m_fit[c] < m_fit[best]) best = c;
    }
    return best;
}

Vec GeneticAlgorithm::uniformCrossover(const Vec& p1, const Vec& p2) {
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    Vec child(p1.size());
    for (size_t i = 0; i < p1.size(); ++i)
        child[i] = (coin(rng_) < 0.5) ? p1[i] : p2[i];
    return child;
}

void GeneticAlgorithm::gaussianMutate(Vec& x) {
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    std::normal_distribution<double>       noise(0.0, m_mutSigma);
    for (auto& v : x)
        if (coin(rng_) < m_mutRate) v += noise(rng_);
}

void GeneticAlgorithm::clampBounds(Vec& x) {
    const auto& lb = prob_->lb();
    const auto& ub = prob_->ub();
    for (int j = 0; j < static_cast<int>(x.size()); ++j)
        x[j] = std::max(lb[j], std::min(ub[j], x[j]));
}

// ─── init ─────────────────────────────────────────────────────────────────────
void GeneticAlgorithm::init() {
    Initializer initSampler;
    m_pop = initSampler.samplePopulation(*prob_, rng_, pop_);
    m_fit.resize(pop_);

    // Evaluate initial population
    for (int i = 0; i < pop_; ++i) {
        m_fit[i] = prob_->evaluate(m_pop[i]);
        if (m_fit[i] < best_f_) {
            best_f_ = m_fit[i]; best_x_ = m_pop[i];
        }
    }
    updateRank();
}

// ─── one_iteration ────────────────────────────────────────────────────────────
void GeneticAlgorithm::one_iteration() {
    const int N = pop_;
    std::vector<Vec>    newPop(N);
    std::vector<double> newFit(N);

    // ── Elitism: copy best individuals unchanged ──────────────────────────────
    const int elite = std::min(m_elite, N);
    for (int i = 0; i < elite; ++i) {
        newPop[i] = m_pop[m_rank[i]];
        newFit[i] = m_fit[m_rank[i]];
    }

    // ── Generate offspring ────────────────────────────────────────────────────
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    for (int i = elite; i < N && !done(); ++i) {
        Vec child;
        if (coin(rng_) < m_crossRate) {
            // Crossover
            int p1 = tournamentSelect();
            int p2 = tournamentSelect();
            child = uniformCrossover(m_pop[p1], m_pop[p2]);
        } else {
            // Clone
            child = m_pop[tournamentSelect()];
        }
        gaussianMutate(child);
        clampBounds(child);

        newFit[i] = prob_->evaluate(child);
        newPop[i] = child;

        if (newFit[i] < best_f_) {
            best_f_ = newFit[i]; best_x_ = child;
        }
    }

    m_pop = std::move(newPop);
    m_fit = std::move(newFit);
    updateRank();
    updateStop(m_fit);
}

} // namespace optimsolution
