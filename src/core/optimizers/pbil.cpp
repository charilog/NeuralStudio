#include "pbil.h"
#include "init.h"
#include <algorithm>
#include <numeric>
#include <cmath>
namespace optimsolution {
void PBIL::init() {
    const int D=(int)prob_->dimension();
    const auto& lb=prob_->lb(); const auto& ub=prob_->ub();
    m_mu.resize(D); m_sigma.resize(D);
    for(int j=0;j<D;++j){ m_mu[j]=(lb[j]+ub[j])*0.5; m_sigma[j]=(ub[j]-lb[j])/6.0; }
    // Evaluate initial population
    Initializer init;
    auto pop = init.samplePopulation(*prob_, rng_, pop_);
    for(auto& x:pop){
        double f=prob_->evaluate(x);
        if(f<best_f_){best_f_=f; best_x_=x;}
    }
}
void PBIL::one_iteration() {
    const int D=(int)prob_->dimension();
    const auto& lb=prob_->lb(); const auto& ub=prob_->ub();
    // Sample population from current distribution
    std::vector<Vec> pop(pop_);
    std::vector<double> fit(pop_);
    for(int i=0;i<pop_;++i){
        pop[i].resize(D);
        for(int j=0;j<D;++j){
            std::normal_distribution<double> nd(m_mu[j], m_sigma[j]);
            pop[i][j]=std::max(lb[j],std::min(ub[j],nd(rng_)));
        }
        fit[i]=prob_->evaluate(pop[i]);
        if(fit[i]<best_f_){best_f_=fit[i]; best_x_=pop[i];}
    }
    // Truncation selection: keep top selPct
    std::vector<int> idx(pop_); std::iota(idx.begin(),idx.end(),0);
    std::sort(idx.begin(),idx.end(),[&](int a,int b){return fit[a]<fit[b];});
    int k=std::max(1,(int)(pop_*m_selPct));
    // Update μ and σ toward the selected mean/std
    Vec selMu(D,0.0), selSig(D,0.0);
    for(int i=0;i<k;++i) for(int j=0;j<D;++j) selMu[j]+=pop[idx[i]][j]/k;
    for(int i=0;i<k;++i) for(int j=0;j<D;++j){ double d=pop[idx[i]][j]-selMu[j]; selSig[j]+=d*d; }
    for(int j=0;j<D;++j){
        selSig[j]=std::sqrt(selSig[j]/k)+1e-6;
        m_mu[j]   = (1-m_lr)*m_mu[j]    + m_lr*selMu[j];
        m_sigma[j] = (1-m_lr)*m_sigma[j] + m_lr*selSig[j];
    }
    updateStop(fit);
}
} // namespace optimsolution
