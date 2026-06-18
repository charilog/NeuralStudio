#pragma once
#include "optimizer.h"
namespace optimsolution {
// Polak-Ribière Conjugate Gradient — uses evalAndGrad (backprop gradient).
// Restarts every D iterations or when the conjugate direction loses descent property.
class ConjugateGradient : public Optimizer {
public:
    std::string methodShortName() const override { return "CG"; }
    std::string methodFullName()  const override { return "Conjugate Gradient (Polak-Ribière)"; }
    void init() override;
    void one_iteration() override;
private:
    Vec    m_g;      // current gradient
    Vec    m_d;      // conjugate direction
    Vec    m_x;      // current iterate
    int    m_restart = 0;
    static double dot(const Vec& a, const Vec& b);
    double lineSearch(const Vec& x, double f, const Vec& g, const Vec& d) const;
};
} // namespace optimsolution
