#include "umda.h"
#include "init.h"
#include <algorithm>
#include <numeric>
#include <cmath>
namespace optimsolution {
void UMDA::init() {
    const int D=(int)prob_->dimension();
    const auto& lb=prob_->lb(); const auto& ub=prob_->ub();
    m_mu.resize(D); m_sigma.resize(D);
    for(int j=0;j<D;++j){ m_mu[j]=(lb[j]+ub[j])*0.5; m_sigma[j]=(ub[j]-lb[j])/6.0; }
    Initializer ini; auto pop=ini.samplePopulation(*prob_,rng_,pop_);
    for(auto& x:pop){ double f=prob_->evaluate(x); if(f<best_f_){best_f_=f;best_x_=x;} }
}
void UMDA::one_iteration() {
    const int D=(int)prob_->dimension();
    const auto& lb=prob_->lb(); const auto& ub=prob_->ub();
    // Sample from current marginals
    std::vector<Vec> pop(pop_); std::vector<double> fit(pop_);
    for(int i=0;i<pop_;++i){
        pop[i].resize(D);
        for(int j=0;j<D;++j){
            std::normal_distribution<double> nd(m_mu[j],m_sigma[j]);
            pop[i][j]=std::max(lb[j],std::min(ub[j],nd(rng_)));
        }
        fit[i]=prob_->evaluate(pop[i]);
        if(fit[i]<best_f_){best_f_=fit[i];best_x_=pop[i];}
    }
    // Truncation selection: best selPct
    std::vector<int> idx(pop_); std::iota(idx.begin(),idx.end(),0);
    std::sort(idx.begin(),idx.end(),[&](int a,int b){return fit[a]<fit[b];});
    int k=std::max(2,(int)(pop_*m_selPct));
    // Estimate new marginals from selected
    Vec newMu(D,0.0), newSig(D,0.0);
    for(int i=0;i<k;++i) for(int j=0;j<D;++j) newMu[j]+=pop[idx[i]][j]/k;
    for(int i=0;i<k;++i) for(int j=0;j<D;++j){ double d=pop[idx[i]][j]-newMu[j]; newSig[j]+=d*d; }
    for(int j=0;j<D;++j){
        m_mu[j]   = newMu[j];
        m_sigma[j] = std::max(1e-6, std::sqrt(newSig[j]/k)); // prevent collapse
    }
    updateStop(fit);
}
} // namespace optimsolution
