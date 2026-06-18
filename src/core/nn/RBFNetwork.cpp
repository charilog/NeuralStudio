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
RBFNetwork::Vec RBFNetwork::activations(const Vec& x) const {
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

    // Apply output activation
    if (m_task == TaskType::BinaryClassification) {
        // Sigmoid
        out[0] = 1.0 / (1.0 + std::exp(-out[0]));
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
