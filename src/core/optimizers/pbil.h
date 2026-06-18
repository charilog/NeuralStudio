#pragma once
#include "optimizer.h"
namespace optimsolution {
// Population-Based Incremental Learning (Gaussian/continuous variant).
// Maintains a probability model N(μ,σ²) per dimension, updated toward the best solutions.
class PBIL : public Optimizer {
public:
    std::string methodShortName() const override { return "PBIL"; }
    std::string methodFullName()  const override { return "Population-Based Incremental Learning"; }
    void configure(const MethodConfig& mc) override {
        m_lr    = mc.getDbl("lr", 0.1);
        m_selPct= mc.getDbl("selection", 0.3);
    }
    void init() override;
    void one_iteration() override;
private:
    double m_lr     = 0.1;
    double m_selPct = 0.3;
    Vec m_mu, m_sigma;
};
} // namespace optimsolution
