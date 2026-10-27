#include "simann.h"
#include <cmath>
#include <algorithm>
namespace optimsolution {
void SimulatedAnnealing::init() {
    m_x = best_x_;
    m_f = prob_->evaluate(m_x);
    best_f_ = m_f;
    m_fScale = std::max(1e-12, std::fabs(m_f));
    m_t = 1.0;
    if (m_cool > 0.0 && m_cool < 1.0) m_coolUsed = m_cool;
    else {
        const double n = std::max(1, max_evals_ - prob_->calls());
        m_coolUsed = std::pow(1e-3, 1.0 / n);   // t: 1 -> 1e-3 over the remaining budget
    }
}
void SimulatedAnnealing::one_iteration() {
    // One iteration = pop_ moves (a "generation", like the population methods, which spend
    // pop_ evaluations per iteration). It made ONE move per iteration, so the drivers, which
    // run `iterations` iterations of a budget iterations x population, used only 1/pop_ of the
    // budget, and the cooling schedule (derived from that budget) never got below t ~ 0.7.
    const int moves = std::max(1, population());
    for (int k = 0; k < moves && prob_->calls() < max_evals_; ++k) move();
}
void SimulatedAnnealing::move() {
    const int D=(int)prob_->dimension();
    const auto& lb=prob_->lb(); const auto& ub=prob_->ub();
    std::normal_distribution<double> noise(0.0, m_sigma * m_t);
    Vec xNew(D);
    for(int j=0;j<D;++j)
        xNew[j]=std::max(lb[j],std::min(ub[j], m_x[j]+noise(rng_)));
    const double fNew = prob_->evaluate(xNew);
    // Metropolis rule against the CURRENT point (it compared with the best point,
    // which made the walk stick to the incumbent)
    const double dE   = fNew - m_f;
    const double T    = m_T0 * m_t * m_fScale;
    std::uniform_real_distribution<double> uni(0.0,1.0);
    if(std::isfinite(fNew) && (dE < 0.0 || (T > 0.0 && uni(rng_) < std::exp(-dE / T)))){
        m_x = xNew; m_f = fNew;
        if(fNew < best_f_){best_f_=fNew; best_x_=xNew;}
    }
    m_t *= m_coolUsed;
}
} // namespace optimsolution
