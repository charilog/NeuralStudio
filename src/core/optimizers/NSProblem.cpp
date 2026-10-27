#include "NSProblem.h"
#include "core/nn/ActivationFunction.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace NeuralStudio {

// ─── Constructor ─────────────────────────────────────────────────────────────
NSProblem::NSProblem(NeuralNetwork* net, const Dataset* ds,
                     double weightBound, double valSplit, unsigned splitSeed)
    : m_net(net), m_ds(ds), m_splitSeed(splitSeed)
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
    std::mt19937 rng(m_splitSeed);
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

// ─── pureForward ─────────────────────────────────────────────────────────────
//  Thread-safe forward pass: reads only the layers' public weights/biases and
//  the pure applyActivation() function; NEVER touches Layer's mutable state
//  (m_lastInput, dropout mask, ...) — that state is only needed for backprop,
//  which the evolutionary path never performs.
//
//  Semantics note: Layer::forward with training-mode OFF (MetaTrainer sets
//  setTrainingMode(false) before evolution; worker-created nets default to
//  false) fills the dropout mask with 1.0 and computes exactly
//      out[o] = act( biases[o] + Σ_i weights[o][i]·in[i] )
//  which is precisely what this function computes — so per-sample results are
//  bit-identical to the previous implementation.
static void pureForward(const NeuralNetwork& net,
                        const std::vector<double>& input,
                        std::vector<double>& cur,
                        std::vector<double>& nxt,
                        double* zOut0 = nullptr) {   // pre-activation of output 0 (binary BCE from the logit)
    cur = input;
    auto& layers = const_cast<NeuralNetwork&>(net).layers();
    const size_t nL = layers.size();
    for (size_t li = 0; li < nL; ++li) {
        const auto& L = layers[li];
        const int nOut = L.outputSize(), nIn = L.inputSize();
        nxt.resize(nOut);
        for (int o = 0; o < nOut; ++o) {
            double z = L.biases[o];
            const auto& w = L.weights[o];
            for (int i = 0; i < nIn; ++i) z += w[i] * cur[i];
            if (zOut0 && o == 0 && li + 1 == nL) *zOut0 = z;
            nxt[o] = applyActivation(L.activation(), z);
        }
        cur.swap(nxt);
    }
    net.applyOutputScaling(cur);   // regression: back to the data scale (identity otherwise)
}

// Binary cross-entropy of one sample. With a sigmoid output it is computed from the logit z
// (exact). The former clipped form  -log(max(1e-7, p))  is flat for |z| > ~16, so a saturated,
// confidently WRONG prediction cost at most 16.1 and contributed no gradient at all.
static inline double binaryLoss(const NeuralNetwork& net, double z, double p, double t) {
    if (const_cast<NeuralNetwork&>(net).layers().back().activation() == Activation::Sigmoid)
        return bceFromLogit(z, t);
    const double eps = 1e-7;
    const double q = std::max(eps, std::min(1.0 - eps, p));
    return -(t * std::log(q) + (1.0 - t) * std::log(1.0 - q));
}

// ─── lossOnIdx ───────────────────────────────────────────────────────────────
//  Mean loss over the given sample indices. This is THE hot path of every
//  evolutionary optimizer (called population-size × generations times per
//  training), so the per-sample loop is parallelised with OpenMP.
//
//  Correctness design:
//   • Weights are written (by unflatten in evaluate()) strictly BEFORE the
//     parallel region and are read-only inside it — no data race.
//   • Each thread uses its own scratch vectors (cur/nxt) — no shared state.
//   • Determinism: instead of an OpenMP `reduction` (whose summation order is
//     unspecified), each thread accumulates a contiguous chunk into its own
//     partial slot, and the partials are then summed serially in fixed order.
//     For a fixed thread count, results are therefore reproducible run-to-run.
//     (Relative to the old strictly-serial order, the chunked order can differ
//     in the last ~1 ulp of the double sum — far below any meaningful level.)
double NSProblem::lossOnIdx(const std::vector<int>& idx) const {
    if (idx.empty()) return 0.0;

    const bool isMulti = (m_net->taskType() == TaskType::MultiClassClassification);
    const bool isBin   = (m_net->taskType() == TaskType::BinaryClassification);
    const int n = static_cast<int>(idx.size());

    auto sampleLoss = [&](int k, std::vector<double>& cur, std::vector<double>& nxt) -> double {
        double z0 = 0.0;
        pureForward(*m_net, m_normInputs[k], cur, nxt, &z0);
        const auto& x = cur;             // network output after pureForward
        const auto& t = m_targets[k];
        double s = 0.0;
        if (isMulti) {
            s = cceFromLogits(x, t);     // exact (the softmax probabilities were clipped at 1e-7)
        } else if (isBin) {
            s = binaryLoss(*m_net, z0, x[0], t[0]);
        } else {
            // mean over the outputs, exactly as computeLoss / lossGradient (MSE)
            for (size_t i = 0; i < x.size(); ++i) {
                double d = x[i] - t[i];
                s += d * d;
            }
            if (!x.empty()) s /= static_cast<double>(x.size());
        }
        return s;
    };

#ifdef _OPENMP
    const int nth = std::max(1, omp_get_max_threads());
    std::vector<double> partial(nth, 0.0);
    #pragma omp parallel num_threads(nth)
    {
        const int tid = omp_get_thread_num();
        std::vector<double> cur, nxt;    // per-thread scratch
        double local = 0.0;
        // static schedule → each thread owns one contiguous chunk of idx,
        // which keeps the final summation order deterministic.
        #pragma omp for schedule(static)
        for (int j = 0; j < n; ++j)
            local += sampleLoss(idx[j], cur, nxt);
        partial[tid] = local;
    }
    double total = 0.0;
    for (int t = 0; t < nth; ++t) total += partial[t];   // fixed order
#else
    std::vector<double> cur, nxt;
    double total = 0.0;
    for (int j = 0; j < n; ++j)
        total += sampleLoss(idx[j], cur, nxt);
#endif
    return total / static_cast<double>(n);
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

// ─── setFoldSplit ────────────────────────────────────────────────────────────
void NSProblem::setFoldSplit(int k, int K) {
    const int n = m_ds->sampleCount;
    if (K < 2 || k < 0 || k >= K || n < K) return;
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(m_splitSeed);
    std::shuffle(idx.begin(), idx.end(), rng);
    const int a = static_cast<int>(static_cast<long long>(n) * k / K);
    const int b = static_cast<int>(static_cast<long long>(n) * (k + 1) / K);
    m_trainIdx.clear(); m_valIdx.clear();
    for (int i = 0; i < n; ++i) (i >= a && i < b ? m_valIdx : m_trainIdx).push_back(idx[i]);
}

// ─── setExternalValDataset ───────────────────────────────────────────────────
void NSProblem::setExternalValDataset(const Dataset* valDs) {
    m_externalValDs = valDs;
    if (!valDs) return;
    // When using external val, ALL training samples go to m_trainIdx
    m_trainIdx.resize(m_ds->sampleCount);
    std::iota(m_trainIdx.begin(), m_trainIdx.end(), 0);
    m_valIdx.clear();   // not used; valLoss/valAcc use m_externalValDs directly
}

// ─── lossOnDataset / accOnDataset ────────────────────────────────────────────
//  Compute loss / accuracy on an external dataset (not pre-cached).
//  Called only for monitoring (~once per optimizer iteration), so on-the-fly
//  normalization is acceptable.
double NSProblem::lossOnDataset(const Dataset* ds) const {
    if (!ds || ds->sampleCount == 0) return -1.0;
    const bool isMulti = (m_net->taskType() == TaskType::MultiClassClassification);
    const bool isBin   = (m_net->taskType() == TaskType::BinaryClassification);
    double total = 0.0;
    const int n = ds->sampleCount;
    std::vector<double> x, nxt;
    for (int r = 0; r < n; ++r) {
        double z0 = 0.0;
        pureForward(*m_net, m_net->normalize(ds->inputRow(r)), x, nxt, &z0);
        const double rawOut = ds->outputAt(r, 0);
        if (isMulti) {
            total += cceFromLogits(x, m_net->oneHot(m_net->classIndex(rawOut)));
        } else if (isBin) {
            const double t = (rawOut > m_midOut) ? 1.0 : 0.0;
            total += binaryLoss(*m_net, z0, x[0], t);
        } else {
            const int nOut = ds->outputCount;
            double s = 0.0;
            for (int o = 0; o < nOut; ++o) {
                double d = x[o] - ds->outputAt(r, o);
                s += d * d;
            }
            total += nOut > 0 ? s / nOut : 0.0;
        }
    }
    return total / n;
}

double NSProblem::accOnDataset(const Dataset* ds) const {
    if (!ds || ds->sampleCount == 0 || !m_isCls) return -1.0;
    const bool isMulti = (m_net->taskType() == TaskType::MultiClassClassification);
    int correct = 0;
    const int n = ds->sampleCount;
    for (int r = 0; r < n; ++r) {
        auto normIn = m_net->normalize(ds->inputRow(r));
        auto pred   = const_cast<NeuralNetwork*>(m_net)->predictNorm(normIn);
        const double rawOut = ds->outputAt(r, 0);
        int tc, pc;
        if (isMulti) {
            tc = m_net->classIndex(rawOut);
            pc = static_cast<int>(
                std::max_element(pred.begin(), pred.end()) - pred.begin());
        } else {
            tc = (rawOut > m_midOut) ? 1 : 0;
            pc = (pred[0] >= 0.5) ? 1 : 0;
        }
        if (tc == pc) ++correct;
    }
    return static_cast<double>(correct) / n;
}

// ─── Monitoring helpers (use CURRENT network weights) ─────────────────────────
double NSProblem::valLoss()  const {
    if (m_externalValDs) return lossOnDataset(m_externalValDs);
    // no validation rows: -1 ("none", as valAcc). lossOnIdx returned 0 for an empty set, a
    // perfect validation loss that early stopping and the GE fitness then took at face value.
    if (m_valIdx.empty()) return -1.0;
    return lossOnIdx(m_valIdx);
}
double NSProblem::valAcc()   const {
    if (m_externalValDs) return accOnDataset(m_externalValDs);
    return accOnIdx(m_valIdx);
}
double NSProblem::trainAcc() const { return accOnIdx(m_trainIdx); }

// ─── evalAndGrad ─────────────────────────────────────────────────────────────
//  Computes loss AND gradient via backpropagation in one pass.
//  Used by L-BFGS. Does NOT count as an extra call beyond the normal evaluate.
std::pair<double, std::vector<double>>
NSProblem::evalAndGrad(const std::vector<double>& x) {
    unflatten(*m_net, x);
    ++m_calls;

    // Dropout OFF: the gradient must belong to the same deterministic loss that evaluate()
    // returns (it switched training mode ON, so with dropout > 0 L-BFGS / CG / LM got the
    // gradient of a randomly thinned network next to the loss of the full one, and their
    // line searches compared incompatible quantities).
    m_net->setTrainingMode(false);

    // Zero all layer gradients
    for (auto& layer : m_net->layers()) layer.zeroGradients();

    const int  n      = static_cast<int>(m_trainIdx.size());
    const bool isMulti = (m_net->taskType() == TaskType::MultiClassClassification);
    const bool isBin   = (m_net->taskType() == TaskType::BinaryClassification);
    const bool sigOut  = isBin && m_net->layers().back().activation() == Activation::Sigmoid;
    double totalLoss   = 0.0;
    if (n == 0) return {0.0, std::vector<double>(m_dim, 0.0)};

    for (int k : m_trainIdx) {
        // Forward pass
        std::vector<double> act = m_normInputs[k];
        for (auto& layer : m_net->layers()) act = layer.forward(act);
        // act = raw logits / sigmoid output / linear regression output

        // Compute output-layer gradient (per sample; the batch average is taken once,
        // when the gradient is collected below — the multi-class branch used to divide
        // by n here as well, which shrank its gradient by a further factor n)
        std::vector<double> grad;
        int firstLayer = static_cast<int>(m_net->layers().size()) - 1;
        if (isMulti) {
            auto probs = softmax(act);
            totalLoss += cceFromLogits(act, m_targets[k]);
            grad = softmaxCCEGrad(probs, m_targets[k]);
        } else if (sigOut) {
            // sigmoid + BCE: loss from the logit, dL/dz = p - t (exact for saturated outputs)
            const double z = m_net->layers().back().lastZ()[0];
            totalLoss += bceFromLogit(z, m_targets[k][0]);
            grad = m_net->layers().back().backwardFromZ({act[0] - m_targets[k][0]});
            --firstLayer;
        } else {
            m_net->applyOutputScaling(act);
            totalLoss += computeLoss(m_net->lossType(), act, m_targets[k]);
            grad = lossGradient(m_net->lossType(), act, m_targets[k]);
            if (m_net->taskType() == TaskType::Regression)
                for (size_t o = 0; o < grad.size(); ++o) grad[o] *= m_net->outScale(static_cast<int>(o));
        }

        // Backward pass (accumulates into layer.m_dW/m_dB)
        for (int li = firstLayer; li >= 0; --li)
            grad = m_net->layers()[li].backward(grad);
    }
    (void)isBin;

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


// ─── predictions (ensembles, noise-aware selection) ──────────────────────────
void NSProblem::predictions(bool validation, std::vector<std::vector<double>>& pred,
                            std::vector<std::vector<double>>& target) const {
    pred.clear(); target.clear();
    const bool isMulti = (m_net->taskType() == TaskType::MultiClassClassification);
    const bool isBin   = (m_net->taskType() == TaskType::BinaryClassification);
    auto toProb = [&](const std::vector<double>& x) {
        if (isMulti) return softmax(x);
        return x;   // sigmoid output (binary) or regression outputs
    };
    std::vector<double> cur, nxt;
    if (validation && m_externalValDs) {
        const Dataset* ds = m_externalValDs;
        for (int r = 0; r < ds->sampleCount; ++r) {
            pureForward(*m_net, m_net->normalize(ds->inputRow(r)), cur, nxt);
            pred.push_back(toProb(cur));
            const double rawOut = ds->outputAt(r, 0);
            if (isMulti) target.push_back(m_net->oneHot(m_net->classIndex(rawOut)));
            else if (isBin) target.push_back({rawOut > m_midOut ? 1.0 : 0.0});
            else {
                std::vector<double> t(ds->outputCount);
                for (int o = 0; o < ds->outputCount; ++o) t[o] = ds->outputAt(r, o);
                target.push_back(t);
            }
        }
        return;
    }
    const std::vector<int>& idx = validation ? m_valIdx : m_trainIdx;
    for (int k : idx) {
        pureForward(*m_net, m_normInputs[k], cur, nxt);
        pred.push_back(toProb(cur));
        target.push_back(m_targets[k]);
    }
}

double NSProblem::predLoss(const std::vector<double>& p, const std::vector<double>& t) const {
    const bool isMulti = (m_net->taskType() == TaskType::MultiClassClassification);
    const bool isBin   = (m_net->taskType() == TaskType::BinaryClassification);
    const double eps = 1e-7;
    double s = 0.0;
    if (isMulti) {
        for (size_t i = 0; i < p.size() && i < t.size(); ++i) s -= t[i] * std::log(std::max(eps, p[i]));
    } else if (isBin) {
        const double q = std::max(eps, std::min(1.0 - eps, p[0]));
        s -= t[0] * std::log(q) + (1.0 - t[0]) * std::log(1.0 - q);
    } else {
        size_t m = 0;
        for (size_t i = 0; i < p.size() && i < t.size(); ++i, ++m) { const double d = p[i] - t[i]; s += d * d; }
        if (m > 0) s /= static_cast<double>(m);
    }
    return s;
}

bool NSProblem::predCorrect(const std::vector<double>& p, const std::vector<double>& t) const {
    if (!m_isCls || p.empty() || t.empty()) return false;
    if (m_net->taskType() == TaskType::MultiClassClassification)
        return std::max_element(p.begin(), p.end()) - p.begin() == std::max_element(t.begin(), t.end()) - t.begin();
    return (p[0] >= 0.5) == (t[0] >= 0.5);
}

// ─── evalAndGradParallel ────────────────────────────────────────────────────
std::pair<double, std::vector<double>>
NSProblem::evalAndGradParallel(const std::vector<double>& x) {
    unflatten(*m_net, x);
    ++m_calls;
    auto& layers = m_net->layers();
    const int L = static_cast<int>(layers.size());
    std::vector<int> offset(L + 1, 0);   // flatten order: weights (row-major) then biases, layer by layer
    for (int l = 0; l < L; ++l) offset[l + 1] = offset[l] + layers[l].outputSize() * (layers[l].inputSize() + 1);
    const bool isMulti = (m_net->taskType() == TaskType::MultiClassClassification);
    const bool sigOut  = m_net->taskType() == TaskType::BinaryClassification &&
                         layers.back().activation() == Activation::Sigmoid;
    const int n = static_cast<int>(m_trainIdx.size());
    if (n == 0) return {0.0, std::vector<double>(m_dim, 0.0)};

    auto oneSample = [&](int k, std::vector<double>& g, std::vector<std::vector<double>>& in,
                         std::vector<std::vector<double>>& z, std::vector<std::vector<double>>& a) -> double {
        // forward, keeping the input, pre-activation and activation of every layer
        const std::vector<double>* cur = &m_normInputs[k];
        for (int l = 0; l < L; ++l) {
            const Layer& Ly = layers[l];
            in[l] = *cur;
            const int nO = Ly.outputSize(), nI = Ly.inputSize();
            z[l].resize(nO); a[l].resize(nO);
            for (int o = 0; o < nO; ++o) {
                double s = Ly.biases[o];
                const auto& w = Ly.weights[o];
                for (int i = 0; i < nI; ++i) s += w[i] * in[l][i];
                z[l][o] = s; a[l][o] = applyActivation(Ly.activation(), s);
            }
            cur = &a[l];
        }
        const std::vector<double>& out = a[L - 1];
        std::vector<double> grad;
        double loss;
        bool gradIsZ = false;            // grad already holds dL/dz of the output layer
        if (isMulti) {
            const auto probs = softmax(out);
            loss = cceFromLogits(out, m_targets[k]);
            grad = softmaxCCEGrad(probs, m_targets[k]);
        } else if (sigOut) {
            loss = bceFromLogit(z[L - 1][0], m_targets[k][0]);
            grad = {out[0] - m_targets[k][0]};
            gradIsZ = true;
        } else if (m_net->taskType() == TaskType::Regression) {
            std::vector<double> y = out;
            m_net->applyOutputScaling(y);
            loss = computeLoss(m_net->lossType(), y, m_targets[k]);
            grad = lossGradient(m_net->lossType(), y, m_targets[k]);
            for (size_t o = 0; o < grad.size(); ++o) grad[o] *= m_net->outScale(static_cast<int>(o));
        } else {
            loss = computeLoss(m_net->lossType(), out, m_targets[k]);
            grad = lossGradient(m_net->lossType(), out, m_targets[k]);
        }
        for (int l = L - 1; l >= 0; --l) {
            const Layer& Ly = layers[l];
            const int nO = Ly.outputSize(), nI = Ly.inputSize();
            std::vector<double> delta(nO);
            for (int o = 0; o < nO; ++o)
                delta[o] = (gradIsZ && l == L - 1) ? grad[o]
                                                   : grad[o] * activationDerivative(Ly.activation(), z[l][o], a[l][o]);
            double* gw = g.data() + offset[l];
            for (int o = 0; o < nO; ++o) {
                const double d = delta[o];
                if (d == 0.0) continue;
                double* row = gw + o * nI;
                for (int i = 0; i < nI; ++i) row[i] += d * in[l][i];
            }
            double* gb = gw + nO * nI;
            for (int o = 0; o < nO; ++o) gb[o] += delta[o];
            if (l > 0) {
                std::vector<double> gi(nI, 0.0);
                for (int o = 0; o < nO; ++o) {
                    const double d = delta[o];
                    if (d == 0.0) continue;
                    const auto& w = Ly.weights[o];
                    for (int i = 0; i < nI; ++i) gi[i] += w[i] * d;
                }
                grad.swap(gi);
            }
        }
        return loss;
    };

    double total = 0.0;
    std::vector<double> flat(m_dim, 0.0);
#ifdef _OPENMP
    const int nth = std::max(1, std::min(omp_get_max_threads(), n / 16 + 1));
    std::vector<std::vector<double>> partG(nth, std::vector<double>(m_dim, 0.0));
    std::vector<double> partL(nth, 0.0);
    #pragma omp parallel num_threads(nth)
    {
        const int tid = omp_get_thread_num();
        std::vector<std::vector<double>> in(L), z(L), a(L);
        #pragma omp for schedule(static)
        for (int j = 0; j < n; ++j) partL[tid] += oneSample(m_trainIdx[j], partG[tid], in, z, a);
    }
    for (int t = 0; t < nth; ++t) {
        total += partL[t];
        for (int i = 0; i < m_dim; ++i) flat[i] += partG[t][i];
    }
#else
    {
        std::vector<std::vector<double>> in(L), z(L), a(L);
        for (int j = 0; j < n; ++j) total += oneSample(m_trainIdx[j], flat, in, z, a);
    }
#endif
    for (double& g : flat) g /= n;
    const double loss = total / n;
    if (loss < m_bestLoss) m_bestLoss = loss;
    return {loss, flat};
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
