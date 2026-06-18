#include "Layer.h"
#include <cmath>
#include <random>
#include <algorithm>

namespace NeuralStudio {

namespace {
// Thread-safe RNG for dropout mask (one per thread).
thread_local std::mt19937 g_dropRng{ std::random_device{}() };
}

Layer::Layer(int inputSize, int outputSize, Activation activation)
    : m_in(inputSize), m_out(outputSize), m_act(activation)
{
    // Xavier / Glorot uniform init
    std::mt19937 rng{ std::random_device{}() };
    const double limit = std::sqrt(6.0 / static_cast<double>(inputSize + outputSize));
    std::uniform_real_distribution<double> dist(-limit, limit);

    weights.assign(outputSize, std::vector<double>(inputSize));
    biases .assign(outputSize, 0.0);
    for (auto& row : weights) for (auto& w : row) w = dist(rng);

    m_lastInput  .resize(inputSize,  0.0);
    m_lastZ      .resize(outputSize, 0.0);
    m_lastOutput .resize(outputSize, 0.0);
    m_dropoutMask.assign(outputSize, 1.0);

    m_dW.assign(outputSize, std::vector<double>(inputSize, 0.0));
    m_dB.assign(outputSize, 0.0);
}

void Layer::zeroGradients() {
    for (auto& row : m_dW) std::fill(row.begin(), row.end(), 0.0);
    std::fill(m_dB.begin(), m_dB.end(), 0.0);
}

std::vector<double> Layer::forward(const std::vector<double>& input) {
    m_lastInput = input;

    // Generate dropout mask only when training AND rate > 0
    const bool useDrop = m_training && m_dropoutRate > 0.0;
    if (useDrop) {
        std::bernoulli_distribution coin(1.0 - m_dropoutRate);
        const double scale = 1.0 / (1.0 - m_dropoutRate);
        for (int o = 0; o < m_out; ++o)
            m_dropoutMask[o] = coin(g_dropRng) ? scale : 0.0;
    } else {
        std::fill(m_dropoutMask.begin(), m_dropoutMask.end(), 1.0);
    }

    for (int o = 0; o < m_out; ++o) {
        double z = biases[o];
        for (int i = 0; i < m_in; ++i)
            z += weights[o][i] * input[i];
        m_lastZ[o]      = z;
        m_lastOutput[o] = applyActivation(m_act, z) * m_dropoutMask[o];
    }
    return m_lastOutput;
}

std::vector<double> Layer::backward(const std::vector<double>& gradOutput) {
    std::vector<double> delta(m_out);
    for (int o = 0; o < m_out; ++o) {
        // Re-apply mask so dropped neurons have zero gradient
        const double act = applyActivation(m_act, m_lastZ[o]);
        delta[o] = gradOutput[o] * activationDerivative(m_act, m_lastZ[o], act)
                                  * m_dropoutMask[o];
    }
    for (int o = 0; o < m_out; ++o) {
        m_dB[o] += delta[o];
        for (int i = 0; i < m_in; ++i)
            m_dW[o][i] += delta[o] * m_lastInput[i];
    }
    std::vector<double> gradIn(m_in, 0.0);
    for (int i = 0; i < m_in; ++i)
        for (int o = 0; o < m_out; ++o)
            gradIn[i] += weights[o][i] * delta[o];
    return gradIn;
}

void Layer::updateSGD(double lr, int batchSize) {
    const double scale = lr / static_cast<double>(batchSize);
    for (int o = 0; o < m_out; ++o) {
        biases[o] -= scale * m_dB[o];
        for (int i = 0; i < m_in; ++i)
            weights[o][i] -= scale * m_dW[o][i];
    }
    zeroGradients();
}

void Layer::initAdam() {
    m_mW.assign(m_out, std::vector<double>(m_in, 0.0));
    m_vW.assign(m_out, std::vector<double>(m_in, 0.0));
    m_mB.assign(m_out, 0.0);
    m_vB.assign(m_out, 0.0);
    m_adamInit = true;
}

void Layer::updateAdam(double lr, double beta1, double beta2,
                       double eps, int t, int batchSize) {
    if (!m_adamInit) initAdam();
    const double scale = 1.0 / static_cast<double>(batchSize);
    const double bc1   = 1.0 - std::pow(beta1, t);
    const double bc2   = 1.0 - std::pow(beta2, t);

    for (int o = 0; o < m_out; ++o) {
        const double gb = m_dB[o] * scale;
        m_mB[o] = beta1 * m_mB[o] + (1.0 - beta1) * gb;
        m_vB[o] = beta2 * m_vB[o] + (1.0 - beta2) * gb * gb;
        biases[o] -= lr * (m_mB[o] / bc1) / (std::sqrt(m_vB[o] / bc2) + eps);

        for (int i = 0; i < m_in; ++i) {
            const double gw = m_dW[o][i] * scale;
            m_mW[o][i] = beta1 * m_mW[o][i] + (1.0 - beta1) * gw;
            m_vW[o][i] = beta2 * m_vW[o][i] + (1.0 - beta2) * gw * gw;
            weights[o][i] -= lr * (m_mW[o][i] / bc1) / (std::sqrt(m_vW[o][i] / bc2) + eps);
        }
    }
    zeroGradients();
}

// ─── initAdaGrad ─────────────────────────────────────────────────────────────
void Layer::initAdaGrad() {
    m_gW.assign(m_out, std::vector<double>(m_in, 0.0));
    m_gB.assign(m_out, 0.0);
    m_adaGradInit = true;
}

// ─── updateAdamW ─────────────────────────────────────────────────────────────
//  AdamW: Adam with DECOUPLED weight decay (applied directly to weights,
//  not via the gradient). This gives better regularisation than L2 in Adam.
void Layer::updateAdamW(double lr, double b1, double b2, double eps,
                        int t, int bs, double wd) {
    if (!m_adamInit) initAdam();
    const double sc  = 1.0 / static_cast<double>(bs);
    const double bc1 = 1.0 - std::pow(b1, t);
    const double bc2 = 1.0 - std::pow(b2, t);
    for (int o = 0; o < m_out; ++o) {
        const double gb = m_dB[o] * sc;
        m_mB[o] = b1 * m_mB[o] + (1.0-b1) * gb;
        m_vB[o] = b2 * m_vB[o] + (1.0-b2) * gb*gb;
        biases[o] -= lr * (m_mB[o]/bc1) / (std::sqrt(m_vB[o]/bc2) + eps)
                   + lr * wd * biases[o];
        for (int i = 0; i < m_in; ++i) {
            const double gw = m_dW[o][i] * sc;
            m_mW[o][i] = b1 * m_mW[o][i] + (1.0-b1) * gw;
            m_vW[o][i] = b2 * m_vW[o][i] + (1.0-b2) * gw*gw;
            weights[o][i] -= lr * (m_mW[o][i]/bc1) / (std::sqrt(m_vW[o][i]/bc2) + eps)
                           + lr * wd * weights[o][i];
        }
    }
    zeroGradients();
}

// ─── updateRMSProp ────────────────────────────────────────────────────────────
//  RMSProp: exponential moving average of squared gradients.
//  No bias correction (unlike Adam). Default ρ = 0.9.
void Layer::updateRMSProp(double lr, double rho, double eps, int bs) {
    if (!m_adamInit) initAdam();
    const double sc = 1.0 / static_cast<double>(bs);
    for (int o = 0; o < m_out; ++o) {
        const double gb = m_dB[o] * sc;
        m_vB[o] = rho * m_vB[o] + (1.0-rho) * gb*gb;
        biases[o] -= lr * gb / (std::sqrt(m_vB[o]) + eps);
        for (int i = 0; i < m_in; ++i) {
            const double gw = m_dW[o][i] * sc;
            m_vW[o][i] = rho * m_vW[o][i] + (1.0-rho) * gw*gw;
            weights[o][i] -= lr * gw / (std::sqrt(m_vW[o][i]) + eps);
        }
    }
    zeroGradients();
}

// ─── updateNadam ─────────────────────────────────────────────────────────────
//  Nadam: Nesterov-accelerated Adam. Uses a lookahead gradient estimate so
//  the momentum correction is applied before the step, giving faster convergence.
void Layer::updateNadam(double lr, double b1, double b2, double eps, int t, int bs) {
    if (!m_adamInit) initAdam();
    const double sc   = 1.0 / static_cast<double>(bs);
    const double bc1t = 1.0 - std::pow(b1, t);
    const double bc1n = 1.0 - std::pow(b1, t+1);  // next step's correction
    const double bc2  = 1.0 - std::pow(b2, t);
    for (int o = 0; o < m_out; ++o) {
        const double gb = m_dB[o] * sc;
        m_mB[o] = b1 * m_mB[o] + (1.0-b1) * gb;
        m_vB[o] = b2 * m_vB[o] + (1.0-b2) * gb*gb;
        // Nesterov: blend current gradient + next-step momentum estimate
        const double mHatN = (b1 * m_mB[o]/bc1n) + ((1.0-b1) * gb/bc1t);
        biases[o] -= lr * mHatN / (std::sqrt(m_vB[o]/bc2) + eps);
        for (int i = 0; i < m_in; ++i) {
            const double gw = m_dW[o][i] * sc;
            m_mW[o][i] = b1 * m_mW[o][i] + (1.0-b1) * gw;
            m_vW[o][i] = b2 * m_vW[o][i] + (1.0-b2) * gw*gw;
            const double mwHatN = (b1 * m_mW[o][i]/bc1n) + ((1.0-b1) * gw/bc1t);
            weights[o][i] -= lr * mwHatN / (std::sqrt(m_vW[o][i]/bc2) + eps);
        }
    }
    zeroGradients();
}

// ─── updateAdaGrad ────────────────────────────────────────────────────────────
//  AdaGrad: accumulates ALL squared gradients (no decay → LR effectively
//  decreases over time). Works well for sparse features; may stagnate later.
void Layer::updateAdaGrad(double lr, double eps, int bs) {
    if (!m_adaGradInit) initAdaGrad();
    const double sc = 1.0 / static_cast<double>(bs);
    for (int o = 0; o < m_out; ++o) {
        const double gb = m_dB[o] * sc;
        m_gB[o] += gb * gb;
        biases[o] -= lr * gb / (std::sqrt(m_gB[o]) + eps);
        for (int i = 0; i < m_in; ++i) {
            const double gw = m_dW[o][i] * sc;
            m_gW[o][i] += gw * gw;
            weights[o][i] -= lr * gw / (std::sqrt(m_gW[o][i]) + eps);
        }
    }
    zeroGradients();
}

} // namespace NeuralStudio
