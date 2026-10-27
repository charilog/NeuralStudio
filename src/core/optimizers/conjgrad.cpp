#include "conjgrad.h"
#include <cmath>
namespace optimsolution {
double ConjugateGradient::dot(const Vec& a, const Vec& b) {
    double s=0; for(size_t i=0;i<a.size();++i) s+=a[i]*b[i]; return s;
}
double ConjugateGradient::lineSearch(const Vec& x, double f, const Vec& g, const Vec& d) const {
    const double gd = dot(g,d);
    if (gd >= 0) return 0;
    double a=1.0;
    const auto& lb=prob_->lb(); const auto& ub=prob_->ub();
    const int D=(int)x.size();
    for(int i=0;i<25;++i){
        Vec xn(D); for(int j=0;j<D;++j) xn[j]=std::max(lb[j],std::min(ub[j],x[j]+a*d[j]));
        if(const_cast<Problem*>(prob_)->evaluate(xn) <= f+1e-4*a*gd) return a;
        a*=0.5;
    }
    return 0.0;   // no Armijo step found (it returned the last tiny trial step, which was then taken
                  // although it did not decrease f, so the iterate could move to a worse point)
}
void ConjugateGradient::init() {
    m_x = best_x_;
    auto [f,g] = prob_->evalAndGrad(m_x);
    best_f_=f; m_f=f; m_g=g; m_d=g; for(auto& v:m_d) v=-v;
    m_restart=0; m_steepest=true; m_stalled=false;
}
void ConjugateGradient::one_iteration() {
    if(m_stalled) return;
    const int D=(int)prob_->dimension();
    // Armijo test against f at the current iterate (it used the best value)
    double a = lineSearch(m_x, m_f, m_g, m_d);
    if(a<=0.0){
        // the conjugate direction failed: restart from steepest descent; when that fails too,
        // no descent step exists (converged / flat), and further iterations evaluate nothing
        if(m_steepest) { m_stalled=true; return; }
        m_d=m_g; for(auto& v:m_d)v=-v; m_restart=0; m_steepest=true; return;
    }
    const auto& lb=prob_->lb(); const auto& ub=prob_->ub();
    Vec xNew(D); for(int j=0;j<D;++j) xNew[j]=std::max(lb[j],std::min(ub[j],m_x[j]+a*m_d[j]));
    auto [fNew,gNew] = prob_->evalAndGrad(xNew);
    if(fNew<best_f_){best_f_=fNew; best_x_=xNew;}
    // Polak-Ribière β (clamped to ≥0)
    double num=0,den=0;
    for(int j=0;j<D;++j){num+=gNew[j]*(gNew[j]-m_g[j]); den+=m_g[j]*m_g[j];}
    double beta = (den>1e-14) ? std::max(0.0, num/den) : 0.0;
    // Restart every D iterations or if β is large (loss of conjugacy)
    if(++m_restart >= D) {beta=0; m_restart=0;}
    for(int j=0;j<D;++j) m_d[j] = -gNew[j] + beta*m_d[j];
    m_steepest = (beta==0.0);
    if(dot(gNew,m_d)>=0){ for(int j=0;j<D;++j) m_d[j]=-gNew[j]; m_steepest=true; m_restart=0; }   // keep a descent direction
    m_x=xNew; m_f=fNew; m_g=gNew;
}
} // namespace optimsolution
