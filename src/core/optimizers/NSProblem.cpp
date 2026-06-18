#include "NSProblem.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>

namespace NeuralStudio {

// ─── Constructor ─────────────────────────────────────────────────────────────
NSProblem::NSProblem(NeuralNetwork* net, const Dataset* ds,
                     double weightBound, double valSplit)
    : m_net(net), m_ds(ds)
{
    m_dim = net->totalParams();

    // Bounds: symmetric [-weightBound, +weightBound]
    m_lb.assign(m_dim, -weightBound);
    m_ub.assign(m_dim, +weightBound);

    // Task info for target encoding
    m_isCls  = (net->taskType() != TaskType::Regression);
    m_midOut = 0.5 * (net->outputMin() + net->outputMax());

    buildCache(valSplit);
}

// ─── buildCache ──────────────────────────────────────────────────────────────
void NSProblem::buildCache(double valSplit) {
    const int n = m_ds->sampleCount;
    m_normInputs.resize(n);
    m_targets.resize(n);

    // Fixed partition (seed 42 for reproducibility across runs)
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(42);
    std::shuffle(idx.begin(), idx.end(), rng);

    const int nVal = static_cast<int>(std::round(n * valSplit));
    m_trainIdx.assign(idx.begin(), idx.begin() + (n - nVal));
    m_valIdx  .assign(idx.begin() + (n - nVal), idx.end());

    // Pre-normalise inputs and encode targets for every sample
    for (int r = 0; r < n; ++r) {
        m_normInputs[r] = m_net->normalize(m_ds->inputRow(r));

        double rawOut = m_ds->outputAt(r, 0);
        if (m_net->taskType() == TaskType::MultiClassClassification) {
            // One-hot vector
            m_targets[r] = m_net->oneHot(m_net->classIndex(rawOut));
        } else if (m_net->taskType() == TaskType::BinaryClassification) {
            m_targets[r] = { (rawOut > m_midOut) ? 1.0 : 0.0 };
        } else {
            // Regression: all output columns
            const int nOut = m_ds->outputCount;
            m_targets[r].resize(nOut);
            for (int o = 0; o < nOut; ++o)
                m_targets[r][o] = m_ds->outputAt(r, o);
        }
    }
}

// ─── evaluate ────────────────────────────────────────────────────────────────
//  Load candidate weights x into the network, compute training loss, return it.
//  (The optimizer calls this many times per iteration.)
double NSProblem::evaluate(const std::vector<double>& x) {
    unflatten(*m_net, x);
    ++m_calls;

    const double loss = lossOnIdx(m_trainIdx);

    // Track the best solution seen so far
    if (loss < m_bestLoss) {
        m_bestLoss = loss;
    }

    return loss;
}

// ─── lossOnIdx ───────────────────────────────────────────────────────────────
double NSProblem::lossOnIdx(const std::vector<int>& idx) const {
    if (idx.empty()) return 0.0;

    const bool isMulti = (m_net->taskType() == TaskType::MultiClassClassification);
    const bool isBin   = (m_net->taskType() == TaskType::BinaryClassification);
    const double eps   = 1e-7;
    double total       = 0.0;

    for (int k : idx) {
        // Forward pass (uses pre-normalised input)
        std::vector<double> x = m_normInputs[k];
        for (auto& layer : const_cast<NeuralNetwork*>(m_net)->layers())
            x = layer.forward(x);
        // x = logits (or sigmoid output for binary)

        const auto& t = m_targets[k];

        if (isMulti) {
            // Softmax + CCE
            auto probs = softmax(x);
            for (size_t i = 0; i < probs.size(); ++i)
                total -= t[i] * std::log(std::max(eps, probs[i]));
        } else if (isBin) {
            // Binary cross-entropy (output layer already applies sigmoid)
            double p = std::max(eps, std::min(1.0 - eps, x[0]));
            total -= t[0] * std::log(p) + (1.0 - t[0]) * std::log(1.0 - p);
        } else {
            // MSE
            for (size_t i = 0; i < x.size(); ++i) {
                double d = x[i] - t[i];
                total += d * d;
            }
        }
    }
    return total / static_cast<double>(idx.size());
}

// ─── accOnIdx ────────────────────────────────────────────────────────────────
double NSProblem::accOnIdx(const std::vector<int>& idx) const {
    if (idx.empty() || !m_isCls) return -1.0;
    int correct = 0;
    const bool isMulti = (m_net->taskType() == TaskType::MultiClassClassification);

    for (int k : idx) {
        auto pred = const_cast<NeuralNetwork*>(m_net)->predictNorm(m_normInputs[k]);
        int tc, pc;
        if (isMulti) {
            tc = static_cast<int>(
                std::max_element(m_targets[k].begin(), m_targets[k].end()) - m_targets[k].begin());
            pc = static_cast<int>(
                std::max_element(pred.begin(), pred.end()) - pred.begin());
        } else {
            tc = (m_targets[k][0] >= 0.5) ? 1 : 0;
            pc = (pred[0] >= 0.5) ? 1 : 0;
        }
        if (tc == pc) ++correct;
    }
    return static_cast<double>(correct) / static_cast<double>(idx.size());
}

// ─── Monitoring helpers (use CURRENT network weights) ─────────────────────────
double NSProblem::valLoss()  const { return lossOnIdx(m_valIdx); }
double NSProblem::valAcc()   const { return accOnIdx(m_valIdx);  }
double NSProblem::trainAcc() const { return accOnIdx(m_trainIdx); }

// ─── evalAndGrad ─────────────────────────────────────────────────────────────
//  Computes loss AND gradient via backpropagation in one pass.
//  Used by L-BFGS. Does NOT count as an extra call beyond the normal evaluate.
std::pair<double, std::vector<double>>
NSProblem::evalAndGrad(const std::vector<double>& x) {
    unflatten(*m_net, x);
    ++m_calls;

    m_net->setTrainingMode(true);

    // Zero all layer gradients
    for (auto& layer : m_net->layers()) layer.zeroGradients();

    const int  n      = static_cast<int>(m_trainIdx.size());
    const bool isMulti = (m_net->taskType() == TaskType::MultiClassClassification);
    const bool isBin   = (m_net->taskType() == TaskType::BinaryClassification);
    const double eps   = 1e-7;
    double totalLoss   = 0.0;

    for (int k : m_trainIdx) {
        // Forward pass
        std::vector<double> act = m_normInputs[k];
        for (auto& layer : m_net->layers()) act = layer.forward(act);
        // act = raw logits / sigmoid output

        // Compute output-layer gradient
        std::vector<double> grad;
        if (isMulti) {
            auto probs = softmax(act);
            totalLoss += cceLoss(probs, m_targets[k]);
            grad = softmaxCCEGrad(probs, m_targets[k]);
            for (auto& g : grad) g /= n;
        } else {
            totalLoss += computeLoss(m_net->lossType(), act, m_targets[k]);
            grad = lossGradient(m_net->lossType(), act, m_targets[k]);
        }

        // Backward pass (accumulates into layer.m_dW/m_dB)
        for (int li = static_cast<int>(m_net->layers().size())-1; li >= 0; --li)
            grad = m_net->layers()[li].backward(grad);
    }

    m_net->setTrainingMode(false);

    // Collect flat gradient from layer accumulators (average over batch)
    std::vector<double> flatGrad;
    flatGrad.reserve(m_dim);
    for (const auto& layer : m_net->layers()) {
        for (const auto& row : layer.dW())
            for (double g : row) flatGrad.push_back(g / n);
        for (double g : layer.dB())  flatGrad.push_back(g / n);
    }

    const double loss = totalLoss / n;
    if (loss < m_bestLoss) m_bestLoss = loss;
    return {loss, flatGrad};
}


std::vector<double> NSProblem::flatten(const NeuralNetwork& net) {
    std::vector<double> flat;
    for (const auto& layer : const_cast<NeuralNetwork&>(net).layers()) {
        for (const auto& row : layer.weights)
            flat.insert(flat.end(), row.begin(), row.end());
        flat.insert(flat.end(), layer.biases.begin(), layer.biases.end());
    }
    return flat;
}

void NSProblem::unflatten(NeuralNetwork& net, const std::vector<double>& w) {
    int idx = 0;
    for (auto& layer : net.layers()) {
        for (auto& row : layer.weights)
            for (double& v : row) v = w[idx++];
        for (double& b : layer.biases) b = w[idx++];
    }
}

} // namespace NeuralStudio
