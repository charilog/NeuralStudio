#pragma once
#include "optimizer.h"
namespace optimsolution {
// Simulated Annealing — single-point, stochastic, gradient-free.
// Temperature anneals geometrically; accepts worse solutions probabilistically.
class SimulatedAnnealing : public Optimizer {
public:
    std::string methodShortName() const override { return "SA"; }
    std::string methodFullName()  const override { return "Simulated Annealing"; }
    void configure(const MethodConfig& mc) override {
        m_T0   = mc.getDbl("T0", 1.0);
        m_cool = mc.getDbl("cooling", 0.97);
        m_sigma= mc.getDbl("sigma", 0.1);
    }
    void init() override;
    void one_iteration() override;
private:
    double m_T0   = 1.0;
    double m_cool = 0.97;
    double m_sigma= 0.1;
    double m_T;
    Vec    m_x;
};
} // namespace optimsolution
