#pragma once
#include "optimizer.h"
namespace optimsolution {
// Univariate Marginal Distribution Algorithm (continuous, Gaussian marginals).
// Estimates independent per-dimension marginals from the selected population.
class UMDA : public Optimizer {
public:
    std::string methodShortName() const override { return "UMDA"; }
    std::string methodFullName()  const override { return "Univariate Marginal Distribution Algorithm"; }
    void configure(const MethodConfig& mc) override {
        m_selPct = mc.getDbl("selection", 0.5);
    }
    void init() override;
    void one_iteration() override;
private:
    double m_selPct = 0.5;
    Vec m_mu, m_sigma;
};
} // namespace optimsolution
