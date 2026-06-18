#pragma once
#include "optimizer.h"
namespace optimsolution {
// Levenberg-Marquardt with diagonal Gauss-Newton Hessian approximation.
// Gain ratio ρ drives λ adaptation (LM damping parameter).
class LevenbergMarquardt : public Optimizer {
public:
    std::string methodShortName() const override { return "LM"; }
    std::string methodFullName()  const override { return "Levenberg-Marquardt"; }
    void configure(const MethodConfig& mc) override { m_lambda = mc.getDbl("lambda",1e-3); }
    void init() override;
    void one_iteration() override;
private:
    double m_lambda = 1e-3;
    Vec    m_x;
};
} // namespace optimsolution
