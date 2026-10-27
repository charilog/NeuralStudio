#include "RBFNetwork.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace NeuralStudio {

void RBFNetwork::init(int nIn, int nOut, TaskType task, const RBFConfig& cfg) {
    m_nIn  = nIn;
    m_nOut = nOut;
    m_task = task;
    m_cfg  = cfg;
    m_built = false;
    m_centres.clear(); m_widths.clear(); m_W.clear();
}

void RBFNetwork::fitPreprocessing(const Dataset& ds) {
    m_inMin.assign(m_nIn, 0.0); m_inRange.assign(m_nIn, 1.0); m_inFill.assign(m_nIn, 0.5);
    for (int d = 0; d < m_nIn && d < static_cast<int>(ds.inputStats.size()); ++d) {
        const auto& st = ds.inputStats[d];
        m_inMin[d]   = st.min;
        m_inRange[d] = (st.max - st.min) > 1e-10 ? st.max - st.min : 1.0;
        m_inFill[d]  = (st.mean - st.min) / m_inRange[d];
    }
    m_classValues = ds.uniqueOutputValues(0);
}

RBFNetwork::Vec RBFNetwork::normalize(const Vec& raw) const {
    if (m_inMin.empty()) return raw;
    Vec x(raw.size());
    for (size_t d = 0; d < raw.size(); ++d) {
        if (d >= m_inMin.size()) { x[d] = raw[d]; continue; }
        x[d] = std::isfinite(raw[d]) ? (raw[d] - m_inMin[d]) / m_inRange[d] : m_inFill[d];
    }
    return x;
}

int RBFNetwork::classIndex(double v) const {
    if (m_classValues.empty()) return 0;
    int best = 0; double bd = std::fabs(v - m_classValues[0]);
    for (int i = 1; i < static_cast<int>(m_classValues.size()); ++i) {
        const double d = std::fabs(v - m_classValues[i]);
        if (d < bd) { bd = d; best = i; }
    }
    return best;
}

RBFNetwork::Vec RBFNetwork::encodeTarget(const Vec& y) const {
    if (m_task == TaskType::Regression) return y;
    const int c = classIndex(y.empty() ? 0.0 : y[0]);
    if (m_task == TaskType::BinaryClassification) return Vec{ c == 1 ? 1.0 : 0.0 };
    Vec t(m_nOut, 0.0);
    if (c >= 0 && c < m_nOut) t[c] = 1.0;
    return t;
}

void RBFNetwork::setCenters(Mat centres, Vec widths) {
    m_centres = std::move(centres);
    m_widths  = std::move(widths);
    // Apply widthScale
    for (double& s : m_widths) s *= m_cfg.widthScale;
}

void RBFNetwork::setOutputWeights(Mat W) {
    m_W    = std::move(W);
    m_built = true;
}

// ─── kernelValue ──────────────────────────────────────────────────────────────
double RBFNetwork::kernelValue(double r2, double sigma) const {
    const double s2 = sigma * sigma;
    if (s2 < 1e-18) return 0.0;
    switch (m_cfg.kernel) {
    case RBFKernel::Multiquadric:
        return std::sqrt(r2 + s2);
    case RBFKernel::InvMultiquadric:
        return 1.0 / std::sqrt(r2 + s2);
    default: // Gaussian
        return std::exp(-r2 / (2.0 * s2));
    }
}

// ─── activations ──────────────────────────────────────────────────────────────
RBFNetwork::Vec RBFNetwork::activations(const Vec& raw) const {
    const Vec x = normalize(raw);
    const int k = int(m_centres.size());
    Vec phi(k + 1);
    for (int i = 0; i < k; ++i) {
        double r2 = 0;
        for (int d = 0; d < m_nIn; ++d) {
            double diff = x[d] - m_centres[i][d];
            r2 += diff * diff;
        }
        phi[i] = kernelValue(r2, m_widths[i]);
    }
    phi[k] = 1.0;   // bias unit
    return phi;
}

// ─── designMatrix ─────────────────────────────────────────────────────────────
RBFNetwork::Mat RBFNetwork::designMatrix(const Mat& X) const {
    const int N = int(X.size());
    const int k = int(m_centres.size());
    Mat Phi(N, Vec(k + 1));
    for (int n = 0; n < N; ++n)
        Phi[n] = activations(X[n]);
    return Phi;
}

// ─── predict ──────────────────────────────────────────────────────────────────
RBFNetwork::Vec RBFNetwork::predict(const Vec& x) const {
    if (!m_built) return Vec(m_nOut, 0.0);
    Vec phi = activations(x);
    const int km1 = int(phi.size());  // k+1

    Vec out(m_nOut, 0.0);
    for (int o = 0; o < m_nOut; ++o)
        for (int j = 0; j < km1; ++j)
            out[o] += m_W[j][o] * phi[j];

    // Output: the weights are a least-squares fit to 0/1 targets, so the score itself is
    // compared with 0.5 (a sigmoid on top, as before, mapped a score of 0 to 0.5 and so
    // predicted class 1 for almost every sample)
    if (m_task == TaskType::BinaryClassification) {
        out[0] = std::min(1.0, std::max(0.0, out[0]));
    } else if (m_task == TaskType::MultiClassClassification) {
        // Softmax
        double maxV = *std::max_element(out.begin(), out.end());
        double sum = 0;
        for (double& v : out) { v = std::exp(v - maxV); sum += v; }
        for (double& v : out) v /= sum;
    }
    return out;
}

} // namespace NeuralStudio
