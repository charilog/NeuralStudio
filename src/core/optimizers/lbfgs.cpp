#include "lbfgs.h"
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace optimsolution {

double LBFGS::dot(const Vec& a, const Vec& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

// Two-loop recursion: compute H_k * q
Vec LBFGS::applyHessian(const Vec& q_in) const {
    Vec q = q_in;
    const int k = (int)m_sHist.size();
    std::vector<double> alpha(k);
    for (int i = k-1; i >= 0; --i) {
        alpha[i] = m_rho[i] * dot(m_sHist[i], q);
        for (size_t j = 0; j < q.size(); ++j) q[j] -= alpha[i] * m_yHist[i][j];
    }
    Vec r = q;
    if (k > 0) {
        const double yy = dot(m_yHist[k-1], m_yHist[k-1]);
        if (yy > 1e-14) {
            const double gamma = dot(m_sHist[k-1], m_yHist[k-1]) / yy;
            for (auto& v : r) v *= gamma;
        }
    }
    for (int i = 0; i < k; ++i) {
        const double beta = m_rho[i] * dot(m_yHist[i], r);
        for (size_t j = 0; j < r.size(); ++j)
            r[j] += m_sHist[i][j] * (alpha[i] - beta);
    }
    return r;
}

// Armijo backtracking line search
double LBFGS::lineSearch(const Vec& x, double f,
                          const Vec& g, const Vec& d,
                          double c1, int maxIter) const {
    const double gd = dot(g, d);
    if (gd >= 0.0) return 0.0;
    double alpha = 1.0;
    const int D = (int)x.size();
    const auto& lb = prob_->lb();
    const auto& ub = prob_->ub();
    for (int i = 0; i < maxIter; ++i) {
        Vec xn(D);
        for (int j = 0; j < D; ++j)
            xn[j] = std::max(lb[j], std::min(ub[j], x[j] + alpha * d[j]));
        const double fn = const_cast<Problem*>(prob_)->evaluate(xn);
        if (fn <= f + c1 * alpha * gd) return alpha;
        alpha *= 0.5;
    }
    // No Armijo step: report failure. (It returned the last trial step, ~3e-8, which the
    // caller then TOOK although it did not decrease f; the iterate drifted to worse points
    // and the same 25 evaluations were repeated in every following iteration.)
    return 0.0;
}

void LBFGS::init() {
    m_sHist.clear(); m_yHist.clear(); m_rho.clear();
    m_stalled = false;
    // best_x_ may be warm-started by MetaTrainer with current network weights
    auto [f0, g0] = prob_->evalAndGrad(best_x_);
    best_f_    = f0;
    m_g        = g0;
    m_currentX = best_x_;
    m_currentF = f0;
}

void LBFGS::one_iteration() {
    if (m_stalled) return;   // converged: callers detect an iteration that evaluates nothing
    const int D = (int)prob_->dimension();

    // Compute search direction d = -H * g
    Vec d = applyHessian(m_g);
    for (auto& v : d) v = -v;

    // Ensure descent direction; reset history if not
    if (dot(m_g, d) >= 0.0) {
        d = m_g; for (auto& v : d) v = -v;
        m_sHist.clear(); m_yHist.clear(); m_rho.clear();
    }

    if (m_activeSet) {
        const auto& lb0 = prob_->lb(); const auto& ub0 = prob_->ub();
        bool any = false;
        for (int j = 0; j < D; ++j) {
            const double tol = 1e-12 * std::max(1.0, std::fabs(ub0[j] - lb0[j]));
            if ((m_currentX[j] <= lb0[j] + tol && d[j] < 0.0) || (m_currentX[j] >= ub0[j] - tol && d[j] > 0.0)) d[j] = 0.0;
            else if (d[j] != 0.0) any = true;
        }
        if (!any) return;   // projected gradient is zero: a stationary point of the box-constrained problem
        if (dot(m_g, d) >= 0.0) {   // the reduced direction is no longer a descent direction: projected steepest descent
            for (int j = 0; j < D; ++j) {
                const double tol = 1e-12 * std::max(1.0, std::fabs(ub0[j] - lb0[j]));
                const double g = -m_g[j];
                d[j] = ((m_currentX[j] <= lb0[j] + tol && g < 0.0) || (m_currentX[j] >= ub0[j] - tol && g > 0.0)) ? 0.0 : g;
            }
            m_sHist.clear(); m_yHist.clear(); m_rho.clear();
        }
    }
    // Armijo condition relative to f at the CURRENT iterate (it used best_f_, which
    // differs once the iterate has moved to a worse point after a projected step)
    const double alpha = lineSearch(m_currentX, m_currentF, m_g, d);
    if (alpha <= 0.0) {
        // the quasi-Newton direction gave no decrease: drop the curvature memory and retry
        // with steepest descent; if that fails as well, no descent step exists any more
        if (!m_sHist.empty()) { m_sHist.clear(); m_yHist.clear(); m_rho.clear(); }
        else m_stalled = true;
        return;
    }

    // New iterate x_{k+1}
    const auto& lb = prob_->lb(); const auto& ub = prob_->ub();
    Vec xNew(D);
    for (int j = 0; j < D; ++j)
        xNew[j] = std::max(lb[j], std::min(ub[j], m_currentX[j] + alpha * d[j]));

    auto [fNew, gNew] = prob_->evalAndGrad(xNew);

    // Update best seen
    if (fNew < best_f_) { best_f_ = fNew; best_x_ = xNew; }

    // L-BFGS memory update (s = x_{k+1} - x_k,  y = g_{k+1} - g_k)
    Vec s(D), y(D);
    for (int j = 0; j < D; ++j) {
        s[j] = xNew[j] - m_currentX[j];
        y[j] = gNew[j] - m_g[j];
    }
    const double sy = dot(s, y);
    if (sy > 1e-14) {
        if ((int)m_sHist.size() >= m_)
            { m_sHist.pop_front(); m_yHist.pop_front(); m_rho.pop_front(); }
        m_sHist.push_back(s); m_yHist.push_back(y);
        m_rho.push_back(1.0 / sy);
    }

    m_currentX = xNew;
    m_currentF = fNew;
    m_g        = gNew;
}

} // namespace optimsolution
