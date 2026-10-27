#include "Trainer.h"
#include "utils/Logger.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <cmath>

namespace NeuralStudio {

Trainer::Trainer(QObject* parent) : QObject(parent) {}

// ─── buildCache ──────────────────────────────────────────────────────────────
void Trainer::buildCache() {
    const int n = m_ds->sampleCount;
    m_normInputs.resize(n);
    m_targets.resize(n);

    const TaskType task   = m_net->taskType();
    const double   midOut = 0.5 * (m_net->outputMin() + m_net->outputMax());

    for (int r = 0; r < n; ++r) {
        m_normInputs[r] = m_net->normalize(m_ds->inputRow(r));

        double rawOut = m_ds->outputAt(r, 0);

        if (task == TaskType::MultiClassClassification) {
            // One-hot encode: map raw value → class index → one-hot vector
            int idx = m_net->classIndex(rawOut);
            m_targets[r] = m_net->oneHot(idx);

        } else if (task == TaskType::BinaryClassification) {
            m_targets[r] = { (rawOut > midOut) ? 1.0 : 0.0 };

        } else { // Regression: pass through raw values for all outputs
            const int nOut = m_ds->outputCount;
            m_targets[r].resize(nOut);
            for (int o = 0; o < nOut; ++o)
                m_targets[r][o] = m_ds->outputAt(r, o);
        }
    }
}

// ─── trainBatch ──────────────────────────────────────────────────────────────
void Trainer::trainBatch(const std::vector<int>& batchIdx, int adamStep) {
    auto& layers = m_net->layers();
    const int bs = static_cast<int>(batchIdx.size());
    for (auto& l : layers) l.zeroGradients();

    const bool isMultiClass = (m_net->taskType() == TaskType::MultiClassClassification);
    const bool isRegression = (m_net->taskType() == TaskType::Regression);

    for (int b : batchIdx) {
        // Forward pass through all layers
        std::vector<double> x = m_normInputs[b];
        for (auto& l : layers) x = l.forward(x);
        // x = raw output of last layer (logits for multi-class, sigmoid for binary)

        std::vector<double> grad;
        int firstLayer = static_cast<int>(layers.size()) - 1;   // first layer of the generic backward loop
        if (isMultiClass) {
            // Apply softmax → compute combined softmax+CCE gradient (per sample; the
            // 1/batchSize average is applied once, by the Layer update — dividing here
            // too, as before, shrank the multi-class gradient by another factor bs)
            std::vector<double> probs = softmax(x);
            grad = softmaxCCEGrad(probs, m_targets[b]);
        } else if (isRegression) {
            // gradient of the STANDARDISED squared error w.r.t. the linear output layer:
            // y = shift + scale*x, loss/scale^2  ->  d/dx = dL/dy / scale. Learning rates
            // then behave the same whatever the units of the target.
            m_net->applyOutputScaling(x);
            grad = lossGradient(m_net->lossType(), x, m_targets[b]);
            for (size_t o = 0; o < grad.size(); ++o) grad[o] /= m_net->outScale(static_cast<int>(o));
        } else if (layers.back().activation() == Activation::Sigmoid) {
            // sigmoid + BCE: dL/dz = p - t, exact. (The gradient of the CLIPPED probability-space
            // BCE times sigmoid'(z) vanished for saturated outputs, i.e. exactly for the samples
            // the network gets confidently wrong, so they stopped being corrected.)
            std::vector<double> gz(x.size());
            for (size_t o = 0; o < x.size(); ++o) gz[o] = x[o] - m_targets[b][o];
            grad = layers.back().backwardFromZ(gz);
            --firstLayer;
        } else {
            grad = lossGradient(m_net->lossType(), x, m_targets[b]);
        }

        // Backward pass (reverse)
        for (int li = firstLayer; li >= 0; --li)
            grad = layers[li].backward(grad);
    }

    // Parameter update — route to the correct optimizer
    for (auto& l : layers) {
        switch (m_cfg.optimizer) {
        case OptimizerType::Adam:
            l.updateAdam(m_cfg.learningRate, m_cfg.adamBeta1, m_cfg.adamBeta2,
                         m_cfg.adamEps, adamStep, bs);
            break;
        case OptimizerType::AdamW:
            l.updateAdamW(m_cfg.learningRate, m_cfg.adamBeta1, m_cfg.adamBeta2,
                          m_cfg.adamEps, adamStep, bs, m_cfg.weightDecay);
            break;
        case OptimizerType::RMSProp:
            l.updateRMSProp(m_cfg.learningRate, m_cfg.rmsRho, m_cfg.adamEps, bs);
            break;
        case OptimizerType::Nadam:
            l.updateNadam(m_cfg.learningRate, m_cfg.adamBeta1, m_cfg.adamBeta2,
                          m_cfg.adamEps, adamStep, bs);
            break;
        case OptimizerType::AdaGrad:
            l.updateAdaGrad(m_cfg.learningRate, m_cfg.adamEps, bs);
            break;
        default: // SGD
            l.updateSGD(m_cfg.learningRate, bs);
            break;
        }
    }
}

// ─── lossOnSet ───────────────────────────────────────────────────────────────
double Trainer::lossOnSet(const std::vector<int>& idx) const {
    if (idx.empty()) return 0.0;
    double total = 0.0;
    const bool isMultiClass = (m_net->taskType() == TaskType::MultiClassClassification);

    for (int k : idx) {
        // Forward: get raw output then apply softmax if needed
        std::vector<double> x = m_normInputs[k];
        for (auto& l : m_net->layers()) x = const_cast<Layer&>(l).forward(x);
        m_net->applyOutputScaling(x);   // regression: data scale

        total += sampleLoss(x, m_targets[k]);
    }
    return total / static_cast<double>(idx.size());
}

// Loss of one sample from the network output x (after the output scaling) and its encoded
// target. Classification losses are computed from the logits (exact, no clipping), the same
// definition as NSProblem, so the curves of the gradient and the evolutionary trainers agree.
double Trainer::sampleLoss(const std::vector<double>& x, const std::vector<double>& t) const {
    const TaskType task = m_net->taskType();
    if (task == TaskType::MultiClassClassification) return cceFromLogits(x, t);
    if (task == TaskType::BinaryClassification) {
        const Layer& out = m_net->layers().back();
        if (out.activation() == Activation::Sigmoid && !out.lastZ().empty() && !t.empty())
            return bceFromLogit(out.lastZ()[0], t[0]);
    }
    return computeLoss(m_net->lossType(), x, t);
}

// ─── accOnSet ────────────────────────────────────────────────────────────────
double Trainer::accOnSet(const std::vector<int>& idx) const {
    if (idx.empty()) return 0.0;
    int correct = 0;
    const bool isMultiClass = (m_net->taskType() == TaskType::MultiClassClassification);

    for (int k : idx) {
        std::vector<double> x = m_normInputs[k];
        for (auto& l : m_net->layers()) x = const_cast<Layer&>(l).forward(x);

        if (isMultiClass) {
            std::vector<double> probs = softmax(x);
            int predClass = static_cast<int>(
                std::max_element(probs.begin(), probs.end()) - probs.begin());
            int trueClass = static_cast<int>(
                std::max_element(m_targets[k].begin(), m_targets[k].end()) - m_targets[k].begin());
            if (predClass == trueClass) ++correct;
        } else {
            int predClass   = (x[0]              >= 0.5) ? 1 : 0;
            int targetClass = (m_targets[k][0]   >= 0.5) ? 1 : 0;
            if (predClass == targetClass) ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(idx.size());
}

// ─── lossOnExtDataset / accOnExtDataset ───────────────────────────────────────
//  Evaluate loss / accuracy on an external dataset (not in the internal cache).
//  Called only for monitoring (once per epoch), so on-the-fly normalization is fine.
double Trainer::lossOnExtDataset(const Dataset* ds) const {
    if (!ds || ds->sampleCount == 0) return -1.0;
    const bool isMultiClass = (m_net->taskType() == TaskType::MultiClassClassification);
    double total = 0.0;
    const int n = ds->sampleCount;
    for (int r = 0; r < n; ++r) {
        auto x = m_net->normalize(ds->inputRow(r));
        for (auto& l : m_net->layers()) x = const_cast<Layer&>(l).forward(x);
        // Build target
        const double rawOut = ds->outputAt(r, 0);
        if (isMultiClass) {
            std::vector<double> t(x.size(), 0.0);
            const int ci = m_net->classIndex(rawOut);
            if (ci >= 0 && ci < static_cast<int>(t.size())) t[ci] = 1.0;
            total += sampleLoss(x, t);
        } else if (m_net->taskType() == TaskType::BinaryClassification) {
            double t0 = (rawOut > 0.5 * (m_net->outputMin() + m_net->outputMax())) ? 1.0 : 0.0;
            std::vector<double> t = {t0};
            total += sampleLoss(x, t);
        } else {
            m_net->applyOutputScaling(x);   // regression: data scale
            const int nOut = ds->outputCount;
            std::vector<double> t(nOut);
            for (int o = 0; o < nOut; ++o) t[o] = ds->outputAt(r, o);
            total += computeLoss(m_net->lossType(), x, t);
        }
    }
    return total / n;
}

double Trainer::accOnExtDataset(const Dataset* ds) const {
    if (!ds || ds->sampleCount == 0) return -1.0;
    if (m_net->taskType() == TaskType::Regression) return -1.0;
    const bool isMultiClass = (m_net->taskType() == TaskType::MultiClassClassification);
    int correct = 0;
    const int n = ds->sampleCount;
    for (int r = 0; r < n; ++r) {
        auto x = m_net->normalize(ds->inputRow(r));
        for (auto& l : m_net->layers()) x = const_cast<Layer&>(l).forward(x);
        const double rawOut = ds->outputAt(r, 0);
        if (isMultiClass) {
            auto probs = softmax(x);
            int predClass = static_cast<int>(
                std::max_element(probs.begin(), probs.end()) - probs.begin());
            if (predClass == m_net->classIndex(rawOut)) ++correct;
        } else {
            double mid = 0.5 * (m_net->outputMin() + m_net->outputMax());
            int pc = (x[0] >= 0.5) ? 1 : 0;
            int tc = (rawOut > mid) ? 1 : 0;
            if (pc == tc) ++correct;
        }
    }
    return static_cast<double>(correct) / n;
}

// ─── run ─────────────────────────────────────────────────────────────────────
void Trainer::run() {
    try {
        if (!m_net || !m_ds) throw std::runtime_error("Trainer: network or dataset not set.");
        if (!m_net->isBuilt()) throw std::runtime_error("Trainer: network not built.");

        // Initialize optimizer state
        for (auto& l : m_net->layers()) {
            if (m_cfg.optimizer == OptimizerType::Adam  ||
                m_cfg.optimizer == OptimizerType::AdamW ||
                m_cfg.optimizer == OptimizerType::RMSProp ||
                m_cfg.optimizer == OptimizerType::Nadam)
                l.initAdam();
            else if (m_cfg.optimizer == OptimizerType::AdaGrad)
                l.initAdaGrad();
        }

        // ── Train/val split ───────────────────────────────────────────────────
        const int total = m_ds->sampleCount;
        std::vector<int> allIdx(total);
        std::iota(allIdx.begin(), allIdx.end(), 0);
        std::mt19937 rng{ 42 };
        std::shuffle(allIdx.begin(), allIdx.end(), rng);

        // If an external val dataset is provided, use ALL training samples and
        // evaluate on the external dataset — no internal split needed.
        const bool hasExtVal = (m_valDs != nullptr && m_valDs->sampleCount > 0);
        const int nVal   = hasExtVal ? 0 :
                           static_cast<int>(std::round(total * m_cfg.validationSplit));
        const int nTrain = total - nVal;
        std::vector<int> trainIdx(allIdx.begin(), allIdx.begin() + nTrain);
        std::vector<int> valIdx  (allIdx.begin() + nTrain, allIdx.end());

        buildCache();

        // ── Early stopping state ──────────────────────────────────────────────
        const bool useES = (m_cfg.earlyStoppingPatience > 0 &&
                             (nVal > 0 || (hasExtVal && !m_valIsTest)));
        if (m_cfg.earlyStoppingPatience > 0 && !useES)
            NS_INFO << "Early stopping disabled: the validation curve is the test file "
                       "(stopping on it would leak the test set).";
        double     bestValLoss = std::numeric_limits<double>::max();
        int        noImprove   = 0;
        // weights of the best-validation epoch, restored when training ends
        std::vector<std::vector<std::vector<double>>> bestW;
        std::vector<std::vector<double>>              bestB;
        auto snapshot = [&]() {
            bestW.clear(); bestB.clear();
            for (const auto& l : m_net->layers()) { bestW.push_back(l.weights); bestB.push_back(l.biases); }
        };

        const bool isCls = (m_net->taskType() != TaskType::Regression);
        const double baseLR = m_cfg.learningRate;

        // ── Training loop ─────────────────────────────────────────────────────
        int adamStep = 0;
        for (int epoch = 1; epoch <= m_cfg.epochs && !m_stop.load(); ++epoch) {
            // LR schedule
            double curLR = baseLR;
            switch (m_cfg.lrSchedule) {
                case LRSchedule::StepDecay:
                    curLR = baseLR * std::pow(m_cfg.lrDecayRate,
                                              (epoch - 1) / m_cfg.lrDecayStepSize);
                    break;
                case LRSchedule::ExponentialDecay:
                    curLR = baseLR * std::pow(m_cfg.lrDecayRate, epoch - 1);
                    break;
                default: break;
            }
            m_cfg.learningRate = curLR; // trainBatch reads from m_cfg

            // Enable dropout during training
            m_net->setTrainingMode(true);

            std::shuffle(trainIdx.begin(), trainIdx.end(), rng);
            const int bs = std::max(1, m_cfg.batchSize);
            for (int start = 0; start < nTrain && !m_stop.load(); start += bs) {
                const int end = std::min(start + bs, nTrain);
                std::vector<int> batch(trainIdx.begin() + start, trainIdx.begin() + end);
                ++adamStep;
                trainBatch(batch, adamStep);
            }

            // Disable dropout for metric evaluation
            m_net->setTrainingMode(false);

            EpochResult res;
            res.epoch     = epoch;
            res.currentLR = curLR;
            res.trainLoss = lossOnSet(trainIdx);
            if (hasExtVal) {
                res.valLoss = lossOnExtDataset(m_valDs);
                if (isCls) {
                    res.trainAcc = accOnSet(trainIdx);
                    res.valAcc   = accOnExtDataset(m_valDs);
                }
            } else {
                res.valLoss   = nVal > 0 ? lossOnSet(valIdx) : -1.0;
                if (isCls) {
                    res.trainAcc = accOnSet(trainIdx);
                    res.valAcc   = nVal > 0 ? accOnSet(valIdx) : -1.0;
                }
            }
            emit epochCompleted(res);

            // Early stopping
            if (useES && res.valLoss >= 0.0) {
                if (res.valLoss < bestValLoss - 1e-6) {
                    bestValLoss = res.valLoss;
                    noImprove   = 0;
                    snapshot();
                } else {
                    ++noImprove;
                    if (noImprove >= m_cfg.earlyStoppingPatience) {
                        NS_INFO << QString("Early stopping at epoch %1 "
                                           "(no improvement for %2 epochs)")
                                       .arg(epoch).arg(m_cfg.earlyStoppingPatience);
                        break;
                    }
                }
            }
        }

        // Early stopping keeps the weights of the best validation epoch (before,
        // the network was left at the last epoch, `patience` epochs past the best).
        if (useES && !bestW.empty()) {
            auto& layers = m_net->layers();
            for (size_t li = 0; li < layers.size() && li < bestW.size(); ++li) {
                layers[li].weights = bestW[li];
                layers[li].biases  = bestB[li];
            }
        }

        // Final reported loss: external val set takes priority; then internal
        // val split; only fall back to train loss when no validation exists.
        // (Previously this ignored the external val dataset because nVal==0
        //  in that mode, and wrongly reported the TRAIN loss as "final val".)
        const double finalVal = hasExtVal        ? lossOnExtDataset(m_valDs)
                              : (nVal > 0)        ? lossOnSet(valIdx)
                                                  : lossOnSet(trainIdx);
        emit trainingFinished(finalVal);

    } catch (const std::exception& ex) {
        emit trainingError(QString::fromStdString(ex.what()));
    }
}

} // namespace NeuralStudio
