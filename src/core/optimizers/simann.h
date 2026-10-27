#pragma once
#include "optimizer.h"
namespace optimsolution {
// Simulated Annealing — single-point, stochastic, gradient-free.
// Temperature anneals geometrically; accepts worse solutions probabilistically.
//
// The temperature is a fraction t in (0, 1] of the scale |f(x0)| of the
// objective (so acceptance does not depend on the units of the loss), the
// Gaussian step is sigma * t, and unless "cooling" is given the cooling factor
// is chosen so that t falls from 1 to 1e-3 over the evaluation budget. Moves
// are judged against the CURRENT point (Metropolis rule).
class SimulatedAnnealing : public Optimizer {
public:
    std::string methodShortName() const override { return "SA"; }
    std::string methodFullName()  const override { return "Simulated Annealing"; }
    void configure(const MethodConfig& mc) override {
        m_T0   = mc.getDbl("T0", 0.01);
        m_cool = mc.getDbl("cooling", 0.0);
        m_sigma= mc.getDbl("sigma", 0.1);
    }
    void init() override;
    void one_iteration() override;
private:
    void move();                 // one Metropolis move (one evaluation)
    double m_T0   = 0.01;    // initial temperature, relative to |f(x0)|
    double m_cool = 0.0;    // per-evaluation factor; 0 = derived from the budget
    double m_sigma= 0.1;    // step at t = 1
    double m_t    = 1.0;    // current relative temperature
    double m_coolUsed = 0.97;
    double m_fScale = 1.0;
    double m_f    = 0.0;    // f(m_x)
    Vec    m_x;
};
} // namespace optimsolution
