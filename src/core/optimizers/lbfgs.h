#pragma once
#include "optimizer.h"
#include <deque>

namespace optimsolution {

// ─── LBFGS ────────────────────────────────────────────────────────────────────
//  Limited-Memory BFGS quasi-Newton optimizer.
//
//  Uses the two-loop recursion to compute H_k * g without storing the full
//  inverse Hessian.  Requires the problem to provide analytical gradients
//  via evalAndGrad() — in NeuralStudio this is satisfied by NSProblem which
//  computes gradients via backpropagation.
//
//  Configuration (via MethodConfig):
//    "memory"   int    number of (s,y) curvature pairs to retain  (default 10)
//
//  Used for both "lbfgs" (m=7) and "bfgs" (m=20) entries in the UI.
// ─────────────────────────────────────────────────────────────────────────────
class LBFGS : public Optimizer {
public:
    std::string methodShortName() const override { return "LBFGS"; }
    std::string methodFullName()  const override {
        return "Limited-Memory BFGS (quasi-Newton)";
    }
    void configure(const MethodConfig& mc) override {
        m_ = mc.getInt("memory", 10);
    }

    void init()          override;
    void one_iteration() override;

    // Allow external code to set memory size directly
    void setMemory(int m) { m_ = m; }

private:
    int m_ = 10;   // number of curvature-pair vectors to keep

    Vec    m_g;         // gradient at current iterate
    Vec    m_currentX;  // current iterate (L-BFGS always advances; ≠ best_x_)

    // L-BFGS history (oldest first)
    std::deque<Vec>    m_sHist;   // s_k = x_{k+1} - x_k
    std::deque<Vec>    m_yHist;   // y_k = g_{k+1} - g_k
    std::deque<double> m_rho;     // 1 / (y_k . s_k)

    // Two-loop recursion: returns H_k * q (negative search direction)
    Vec  applyHessian(const Vec& q) const;

    // Armijo backtracking line search; returns step length alpha
    double lineSearch(const Vec& x, double f, const Vec& g, const Vec& d,
                      double c1 = 1e-4, int maxIter = 25) const;

    static double dot(const Vec& a, const Vec& b);
};

} // namespace optimsolution
