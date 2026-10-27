#include "CrossValidator.h"
#include "utils/Logger.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace NeuralStudio {

CrossValidator::CrossValidator(QObject* parent) : QObject(parent) {}

// ─── run ─────────────────────────────────────────────────────────────────────
//  Standard K-fold: shuffle indices once, then for each fold use that slice
//  as validation and everything else as training. Train a fresh network on
//  each fold so weights don't leak between folds.
void CrossValidator::run() {
    try {
        if (!m_ds)              throw std::runtime_error("CrossValidator: dataset not set.");
        if (m_k < 2 || m_k > 20) throw std::runtime_error("CrossValidator: K must be 2..20.");

        const int n = m_ds->sampleCount;
        if (n < m_k) throw std::runtime_error("CrossValidator: not enough samples for K folds.");

        // Shuffle once
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::mt19937 rng{ 12345 };
        std::shuffle(idx.begin(), idx.end(), rng);

        // Compute fold boundaries
        const int foldSize = n / m_k;
        std::vector<FoldResult> results;

        // One Trainer for all folds (reconfigured per fold): it outlives every fold, so a
        // requestStop() from the GUI thread can never reach a destroyed object.
        Trainer trainer;
        FoldResult res;
        QString foldError;   // a training error used to be ignored (fold reported as 0)
        QObject::connect(&trainer, &Trainer::epochCompleted,
            this, [&res](EpochResult er){
                res.finalLoss = er.valLoss >= 0 ? er.valLoss : er.trainLoss;
                res.finalAcc  = er.valAcc  >= 0 ? er.valAcc  : er.trainAcc;
            });
        QObject::connect(&trainer, &Trainer::trainingError, this,
            [&foldError](QString m) { foldError = m; });
        m_current.store(&trainer);
        struct ClearCurrent { std::atomic<Trainer*>& p; ~ClearCurrent() { p.store(nullptr); } } clearCurrent{m_current};
        if (m_stop.load()) trainer.requestStop();

        for (int f = 0; f < m_k && !m_stop.load(); ++f) {
            const int valStart = f * foldSize;
            const int valEnd   = (f == m_k - 1) ? n : valStart + foldSize;

            // Build TWO separate datasets: the training fold (all rows except
            // the current fold) and the validation fold. The validation fold is
            // handed to Trainer::setValDataset() as an EXTERNAL val set.
            //
            // NOTE: the previous implementation rearranged one dataset so the
            // fold sat "at the end" and relied on Trainer's validationSplit to
            // carve it back off — but Trainer SHUFFLES its index array before
            // splitting, so the actual validation rows were a random subset,
            // not the intended fold. That silently violated the core k-fold
            // guarantee (each sample validated exactly once) and allowed
            // train/val leakage across folds. Using an explicit external val
            // dataset removes the entire failure mode.
            Dataset trainFold = *m_ds;
            Dataset valFold   = *m_ds;
            trainFold.inputs.clear();  trainFold.outputs.clear();
            valFold.inputs.clear();    valFold.outputs.clear();
            for (int j = 0; j < n; ++j) {
                const bool inVal = (j >= valStart && j < valEnd);
                Dataset& dst = inVal ? valFold : trainFold;
                dst.inputs .push_back(m_ds->inputs [idx[j]]);
                dst.outputs.push_back(m_ds->outputs[idx[j]]);
            }
            trainFold.sampleCount = static_cast<int>(trainFold.inputs.size());
            valFold.sampleCount   = static_cast<int>(valFold.inputs.size());
            // statistics of the TRAINING fold only: the min-max input scaling and
            // the regression output scaling must not see the held-out fold
            trainFold.computeStatistics();
            valFold.computeStatistics();

            // Build a fresh network per fold
            auto net = std::make_unique<NeuralNetwork>();
            net->setNormalization(trainFold.inputStats); // train-fold statistics (no leakage)

            int netOutSize;
            if (m_netCfg.task == TaskType::MultiClassClassification) {
                auto cv = m_ds->uniqueOutputValues(0);
                netOutSize = static_cast<int>(cv.size());
                net->setClassValues(std::vector<double>(cv.begin(), cv.end()));
            } else if (m_netCfg.task == TaskType::BinaryClassification) {
                netOutSize = 1;
            } else {
                netOutSize = m_ds->outputCount;
            }
            net->build(m_ds->inputCount, netOutSize, m_netCfg);
            net->fitOutputScaling(&trainFold);

            auto cv = m_ds->uniqueOutputValues(0);
            if (cv.size() >= 2) net->setOutputMapping(cv.front(), cv.back());

            // Configure Trainer: the validation fold is an external dataset,
            // so ALL trainFold samples are used for training and the internal
            // split is irrelevant (Trainer sets nVal=0 when an external val
            // dataset is present).
            TrainerConfig tc = m_trCfg;
            tc.validationSplit = 0.0;
            tc.earlyStoppingPatience = 0; // disable for CV to keep folds comparable

            trainer.setConfig(tc);
            trainer.setNetwork(net.get());
            trainer.setDataset(&trainFold);
            trainer.setValDataset(&valFold);   // fold = external validation set

            // Capture final epoch result
            res = FoldResult{};
            res.fold = f + 1;
            foldError.clear();

            // Run synchronously in this thread (we're already in a worker thread).
            // requestStop() reaches the running fold through m_current (it was only checked
            // between folds, so Stop waited for a whole fold of training).
            trainer.run();
            if (!foldError.isEmpty())
                throw std::runtime_error(QString("fold %1: %2").arg(f + 1).arg(foldError).toStdString());
            if (m_stop.load()) break;   // an interrupted fold is not a result

            results.push_back(res);
            emit foldCompleted(res);
        }

        // Compute summary stats
        CVSummary sum;
        sum.k = m_k;
        sum.folds = results;
        if (!results.empty()) {
            double ml = 0.0, ma = 0.0; int accN = 0;
            for (const auto& r : results) {
                ml += r.finalLoss;
                if (r.finalAcc >= 0) { ma += r.finalAcc; ++accN; }
            }
            sum.meanLoss = ml / results.size();
            sum.meanAcc  = accN > 0 ? ma / accN : -1.0;

            double vl = 0.0, va = 0.0;
            for (const auto& r : results) {
                vl += (r.finalLoss - sum.meanLoss) * (r.finalLoss - sum.meanLoss);
                if (r.finalAcc >= 0) va += (r.finalAcc - sum.meanAcc) * (r.finalAcc - sum.meanAcc);
            }
            // sample standard deviation over the folds (n - 1; it divided by n)
            const double nl = static_cast<double>(results.size());
            sum.stdLoss = nl > 1 ? std::sqrt(vl / (nl - 1)) : 0.0;
            sum.stdAcc  = accN > 1 ? std::sqrt(va / (accN - 1)) : (accN == 1 ? 0.0 : -1.0);
        }
        emit cvFinished(sum);

    } catch (const std::exception& ex) {
        emit cvError(QString::fromStdString(ex.what()));
    }
}

} // namespace NeuralStudio
