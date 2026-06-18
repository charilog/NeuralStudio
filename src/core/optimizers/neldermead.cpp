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
    m_simplex.push_back(best_x_);
    m_fval.push_back(prob_->evaluate(best_x_));
    double delta = 0.0;
    for(int j=0;j<D;++j) delta += (ub[j]-lb[j]);
    delta = std::max(0.05, delta/D*0.05); // 5% of average range
    for(int i=0;i<D;++i){
        Vec v=best_x_; v[i]=std::max(lb[i],std::min(ub[i],v[i]+delta));
        m_simplex.push_back(v);
        m_fval.push_back(prob_->evaluate(v));
        if(m_fval.back()<best_f_){best_f_=m_fval.back(); best_x_=v;}
    }
    sortSimplex();
}
void NelderMead::one_iteration() {
    const int D=(int)prob_->dimension();
    const auto& lb=prob_->lb(); const auto& ub=prob_->ub();
    auto clamp=[&](Vec v){for(int j=0;j<D;++j)v[j]=std::max(lb[j],std::min(ub[j],v[j]));return v;};
    auto eval=[&](Vec v){double f=prob_->evaluate(v);if(f<best_f_){best_f_=f;best_x_=v;}return f;};
    Vec c=centroid();
    Vec xw=m_simplex.back(); double fw=m_fval.back();
    // Reflection
    Vec xr=clamp(c); for(int j=0;j<D;++j) xr[j]=c[j]+ALPHA*(c[j]-xw[j]);
    double fr=eval(xr);
    if(fr<m_fval[0]){
        // Expansion
        Vec xe=clamp(c); for(int j=0;j<D;++j) xe[j]=c[j]+GAMMA*(xr[j]-c[j]);
        double fe=eval(xe);
        m_simplex.back()=(fe<fr)?xe:xr; m_fval.back()=(fe<fr)?fe:fr;
    } else if(fr<m_fval[m_simplex.size()-2]){
        m_simplex.back()=xr; m_fval.back()=fr;
    } else {
        // Contraction
        Vec xc=clamp(c); for(int j=0;j<D;++j) xc[j]=c[j]+RHO*(xw[j]-c[j]);
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
