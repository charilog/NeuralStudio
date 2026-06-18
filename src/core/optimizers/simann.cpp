#include "simann.h"
#include <cmath>
namespace optimsolution {
void SimulatedAnnealing::init() {
    m_x = best_x_;
    best_f_ = prob_->evaluate(m_x);
    m_T = m_T0;
}
void SimulatedAnnealing::one_iteration() {
    const int D=(int)prob_->dimension();
    const auto& lb=prob_->lb(); const auto& ub=prob_->ub();
    std::normal_distribution<double> noise(0.0, m_sigma * m_T);
    Vec xNew(D);
    for(int j=0;j<D;++j)
        xNew[j]=std::max(lb[j],std::min(ub[j], m_x[j]+noise(rng_)));
    double fNew = prob_->evaluate(xNew);
    double dE   = fNew - best_f_;
    // Accept if better, or with Boltzmann probability if worse
    std::uniform_real_distribution<double> uni(0.0,1.0);
    if(dE < 0.0 || (m_T > 1e-12 && uni(rng_) < std::exp(-dE / m_T))){
        m_x = xNew;
        if(fNew < best_f_){best_f_=fNew; best_x_=xNew;}
    }
    m_T *= m_cool;
}
} // namespace optimsolution
