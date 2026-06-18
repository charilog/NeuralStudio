#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <QString>

namespace NeuralStudio {

enum class LossType { MSE, BinaryCrossEntropy, CategoricalCrossEntropy };

inline QString lossName(LossType l) {
    switch(l) {
        case LossType::MSE:                      return "MSE";
        case LossType::BinaryCrossEntropy:       return "Binary Cross-Entropy";
        case LossType::CategoricalCrossEntropy:  return "Categorical Cross-Entropy";
    }
    return "Unknown";
}

// ── Softmax (numerical stable) ────────────────────────────────────────────────
inline std::vector<double> softmax(const std::vector<double>& x) {
    double maxVal = *std::max_element(x.begin(), x.end());
    std::vector<double> out(x.size());
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = std::exp(x[i] - maxVal);
        sum += out[i];
    }
    for (auto& v : out) v /= (sum + 1e-10);
    return out;
}

// ── CCE loss from softmax probabilities ───────────────────────────────────────
inline double cceLoss(const std::vector<double>& probs,
                      const std::vector<double>& oneHot) {
    const double eps = 1e-7;
    double loss = 0.0;
    for (size_t i = 0; i < probs.size(); ++i)
        loss -= oneHot[i] * std::log(std::max(eps, probs[i]));
    return loss;
}

// ── Combined gradient of softmax + CCE = probs - one_hot ─────────────────────
inline std::vector<double> softmaxCCEGrad(const std::vector<double>& probs,
                                           const std::vector<double>& oneHot) {
    std::vector<double> grad(probs.size());
    for (size_t i = 0; i < probs.size(); ++i)
        grad[i] = probs[i] - oneHot[i];
    return grad;
}

// ── Generic loss (for MSE / BCE) ──────────────────────────────────────────────
inline double computeLoss(LossType loss,
                          const std::vector<double>& predicted,
                          const std::vector<double>& target) {
    const int n = static_cast<int>(predicted.size());
    if (n == 0) return 0.0;
    double sum = 0.0;
    if (loss == LossType::MSE) {
        for (int i = 0; i < n; ++i) { double d = predicted[i]-target[i]; sum += d*d; }
    } else {
        const double eps = 1e-7;
        for (int i = 0; i < n; ++i) {
            double p = std::max(eps, std::min(1.0-eps, predicted[i]));
            sum += -(target[i]*std::log(p) + (1.0-target[i])*std::log(1.0-p));
        }
    }
    return sum / n;
}

inline std::vector<double> lossGradient(LossType loss,
                                        const std::vector<double>& predicted,
                                        const std::vector<double>& target) {
    const int n = static_cast<int>(predicted.size());
    std::vector<double> grad(n, 0.0);
    if (loss == LossType::MSE) {
        for (int i = 0; i < n; ++i) grad[i] = 2.0*(predicted[i]-target[i])/n;
    } else {
        const double eps = 1e-7;
        for (int i = 0; i < n; ++i) {
            double p = std::max(eps, std::min(1.0-eps, predicted[i]));
            grad[i] = (-(target[i]/p) + (1.0-target[i])/(1.0-p)) / n;
        }
    }
    return grad;
}

} // namespace NeuralStudio
