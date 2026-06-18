#pragma once
#include "optimizer.h"
namespace optimsolution {
// Nelder-Mead Downhill Simplex — gradient-free, single-point.
// Maintains D+1 vertices; suitable for low-dimensional problems.
// For large D: use a "reduced" simplex with fewer vertices (subset of dims).
class NelderMead : public Optimizer {
public:
    std::string methodShortName() const override { return "NM"; }
    std::string methodFullName()  const override { return "Nelder-Mead Simplex"; }
    void configure(const MethodConfig&) override {}
    void init() override;
    void one_iteration() override;
private:
    static constexpr double ALPHA=1.0, GAMMA=2.0, RHO=0.5, SIGMA=0.5;
    std::vector<Vec>    m_simplex;
    std::vector<double> m_fval;
    void sortSimplex();
    Vec  centroid() const;
};
} // namespace optimsolution
