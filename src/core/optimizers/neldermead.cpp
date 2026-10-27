#include "neldermead.h"
#include "init.h"
#include <algorithm>
#include <numeric>
namespace optimsolution {
void NelderMead::sortSimplex(){
    std::vector<int> idx(m_simplex.size());
    std::iota(idx.begin(),idx.end(),0);
    std::sort(idx.begin(),idx.end(),[&](int a,int b){return m_fval[a]<m_fval[b];});
    std::vector<Vec> sv; std::vector<double> fv;
    for(int i:idx){sv.push_back(m_simplex[i]); fv.push_back(m_fval[i]);}
    m_simplex=sv; m_fval=fv;
}
Vec NelderMead::centroid() const {
    const int D=(int)m_simplex[0].size();
    const int N=(int)m_simplex.size()-1; // exclude worst
    Vec c(D,0.0);
    for(int i=0;i<N;++i) for(int j=0;j<D;++j) c[j]+=m_simplex[i][j]/N;
    return c;
}
void NelderMead::init() {
    const int D=(int)prob_->dimension();
    const auto& lb=prob_->lb(); const auto& ub=prob_->ub();
    // D+1 initial vertices: x0 = best_x_, rest = x0 ± small perturbation
    m_simplex.clear(); m_fval.clear();
    for(int j=0;j<D && j<(int)best_x_.size();++j) best_x_[j]=std::max(lb[j],std::min(ub[j],best_x_[j]));
    m_simplex.push_back(best_x_);
    m_fval.push_back(prob_->evaluate(best_x_));
    best_f_ = m_fval.back();   // the start point is a candidate too (it was ignored, and a
                               // worse perturbed vertex could replace a better warm start)
    double delta = 0.0;
    for(int j=0;j<D;++j) delta += (ub[j]-lb[j]);
    delta = std::max(0.05, delta/D*0.05); // 5% of average range
    const Vec x0 = best_x_;   // every vertex is x0 moved along ONE axis (best_x_ may change below)
    for(int i=0;i<D;++i){
        Vec v=x0;
        // step inward when x0 sits on the upper bound (clamping v[i] back to x0[i] gave a
        // degenerate, flat simplex that can never move along that coordinate)
        v[i] = (x0[i]+delta <= ub[i]) ? x0[i]+delta : std::max(lb[i], x0[i]-delta);
        m_simplex.push_back(v);
        m_fval.push_back(prob_->evaluate(v));
        if(m_fval.back()<best_f_){best_f_=m_fval.back(); best_x_=v;}
    }
    sortSimplex();
}
void NelderMead::one_iteration() {
    // One iteration = steps until pop_ evaluations are spent (as the population methods);
    // a single reflection step per iteration used a small fraction of the budget the drivers
    // assign (iterations x population).
    const int target = prob_->calls() + std::max(1, population());
    while (prob_->calls() < target && prob_->calls() < max_evals_) step();
}
void NelderMead::step() {
    const int D=(int)prob_->dimension();
    const auto& lb=prob_->lb(); const auto& ub=prob_->ub();
    auto clamp=[&](Vec v){for(int j=0;j<D;++j)v[j]=std::max(lb[j],std::min(ub[j],v[j]));return v;};
    auto eval=[&](Vec v){double f=prob_->evaluate(v);if(f<best_f_){best_f_=f;best_x_=v;}return f;};
    Vec c=centroid();
    Vec xw=m_simplex.back(); double fw=m_fval.back();
    // Reflection
    // every trial point is projected onto the box AFTER it is computed (the clamp was
    // applied to the centroid and then overwritten, so trial points left the bounds)
    Vec xr(D); for(int j=0;j<D;++j) xr[j]=c[j]+ALPHA*(c[j]-xw[j]); xr=clamp(xr);
    double fr=eval(xr);
    if(fr<m_fval[0]){
        // Expansion
        Vec xe(D); for(int j=0;j<D;++j) xe[j]=c[j]+GAMMA*(xr[j]-c[j]); xe=clamp(xe);
        double fe=eval(xe);
        m_simplex.back()=(fe<fr)?xe:xr; m_fval.back()=(fe<fr)?fe:fr;
    } else if(fr<m_fval[m_simplex.size()-2]){
        m_simplex.back()=xr; m_fval.back()=fr;
    } else {
        // Contraction
        Vec xc(D); for(int j=0;j<D;++j) xc[j]=c[j]+RHO*(xw[j]-c[j]); xc=clamp(xc);
        double fc=eval(xc);
        if(fc<fw){ m_simplex.back()=xc; m_fval.back()=fc; }
        else {
            // Shrink
            for(int i=1;i<(int)m_simplex.size();++i){
                for(int j=0;j<D;++j)
                    m_simplex[i][j]=m_simplex[0][j]+SIGMA*(m_simplex[i][j]-m_simplex[0][j]);
                m_fval[i]=eval(m_simplex[i]);
            }
        }
    }
    sortSimplex();
}
} // namespace optimsolution
