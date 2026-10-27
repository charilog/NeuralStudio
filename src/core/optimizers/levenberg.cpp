#include "levenberg.h"
#include <cmath>
#include <algorithm>
namespace optimsolution {
void LevenbergMarquardt::init() {
    m_x = best_x_;
    auto [f,g] = prob_->evalAndGrad(m_x);
    best_f_=f; m_f=f; m_g=g;
}
void LevenbergMarquardt::one_iteration() {
    const int D=(int)prob_->dimension();
    // f and g at the current point are kept from the last accepted step (they were
    // recomputed every iteration, also after a rejected step, wasting an evaluation)
    const double f = m_f; const Vec& g = m_g;
    // Damped, diagonally scaled gradient step: Δw[i] = -g[i] / (|g[i]| + λ).
    // NOTE: this is a diagonal trust-region heuristic, not the full Levenberg-Marquardt
    // method (that needs the Jacobian of the residuals, which the problem does not give);
    // λ is adapted with the gain ratio as in LM.
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
    auto [fNew, gNew] = prob_->evalAndGrad(xNew);
    // Gain ratio ρ = actual / predicted
    double rho = (predReduction > 1e-14) ? (f - fNew) / predReduction : -1.0;
    if(rho > 0.0){
        m_x = xNew; m_f = fNew; m_g = gNew;
        if(fNew < best_f_){best_f_=fNew; best_x_=xNew;}
        if(rho > 0.75) m_lambda = std::max(1e-9, m_lambda/3.0);
    } else {
        m_lambda = std::min(1e6, m_lambda*10.0);
    }
    updateStop(m_g);
}
} // namespace optimsolution
