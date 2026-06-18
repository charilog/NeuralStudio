#include "levenberg.h"
#include <cmath>
#include <algorithm>
namespace optimsolution {
void LevenbergMarquardt::init() {
    m_x = best_x_;
    auto [f,g] = prob_->evalAndGrad(m_x);
    best_f_=f;
}
void LevenbergMarquardt::one_iteration() {
    const int D=(int)prob_->dimension();
    auto [f,g] = prob_->evalAndGrad(m_x);
    // Diagonal Gauss-Newton approximation: H_diag[i] = g[i]²
    // LM step: Δw[i] = -g[i] / (|g[i]| + λ)
    // (equivalent to (J^T J + λI)^{-1} J^T r with diagonal J^T J = g²)
    Vec step(D);
    double predReduction = 0;
    for(int j=0;j<D;++j){
        double hd = std::abs(g[j]) + m_lambda;
        step[j]   = -g[j] / hd;
        predReduction += g[j]*g[j] / hd;
    }
    const auto& lb=prob_->lb(); const auto& ub=prob_->ub();
    Vec xNew(D);
    for(int j=0;j<D;++j) xNew[j]=std::max(lb[j],std::min(ub[j],m_x[j]+step[j]));
    double fNew = prob_->evaluate(xNew);
    // Gain ratio ρ = actual / predicted
    double rho = (predReduction > 1e-14) ? (f - fNew) / predReduction : -1.0;
    if(rho > 0.0){
        m_x = xNew;
        if(fNew < best_f_){best_f_=fNew; best_x_=xNew;}
        if(rho > 0.75) m_lambda = std::max(1e-9, m_lambda/3.0);
    } else {
        m_lambda = std::min(1e6, m_lambda*10.0);
    }
    updateStop(g);
}
} // namespace optimsolution
