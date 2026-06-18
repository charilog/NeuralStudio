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

        for (int f = 0; f < m_k && !m_stop.load(); ++f) {
            const int valStart = f * foldSize;
            const int valEnd   = (f == m_k - 1) ? n : valStart + foldSize;

            // Build a temporary "rearranged" dataset where rows valStart..valEnd-1
            // become the validation set after the train/val split inside Trainer.
            // Simpler: write a *new* Dataset whose rows are reordered so that
            // the validation slice ends up at the end.
            Dataset folded   = *m_ds;
            folded.inputs    .resize(n);
            folded.outputs   .resize(n);
            int writePos = 0;
            // Train rows first (all except current fold)
            for (int j = 0; j < n; ++j) {
                if (j >= valStart && j < valEnd) continue;
                folded.inputs [writePos] = m_ds->inputs [idx[j]];
                folded.outputs[writePos] = m_ds->outputs[idx[j]];
                ++writePos;
            }
            // Validation rows last
            for (int j = valStart; j < valEnd; ++j) {
                folded.inputs [writePos] = m_ds->inputs [idx[j]];
                folded.outputs[writePos] = m_ds->outputs[idx[j]];
                ++writePos;
            }

            // Build a fresh network per fold
            auto net = std::make_unique<NeuralNetwork>();
            net->setNormalization(m_ds->inputStats); // use full-dataset stats

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

            auto cv = m_ds->uniqueOutputValues(0);
            if (cv.size() >= 2) net->setOutputMapping(cv.front(), cv.back());

            // Configure Trainer: validation slice = current fold (at the end of rearranged ds)
            TrainerConfig tc = m_trCfg;
            tc.validationSplit = static_cast<double>(valEnd - valStart) / n;
            tc.earlyStoppingPatience = 0; // disable for CV to keep folds comparable

            Trainer trainer;
            trainer.setConfig(tc);
            trainer.setNetwork(net.get());
            trainer.setDataset(&folded);

            // Capture final epoch result
            FoldResult res;
            res.fold = f + 1;
            QObject::connect(&trainer, &Trainer::epochCompleted,
                this, [&res](EpochResult er){
                    res.finalLoss = er.valLoss >= 0 ? er.valLoss : er.trainLoss;
                    res.finalAcc  = er.valAcc  >= 0 ? er.valAcc  : er.trainAcc;
                });

            // Run synchronously in this thread (we're already in a worker thread)
            trainer.run();

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
            sum.stdLoss = std::sqrt(vl / results.size());
            sum.stdAcc  = accN > 0 ? std::sqrt(va / accN) : -1.0;
        }
        emit cvFinished(sum);

    } catch (const std::exception& ex) {
        emit cvError(QString::fromStdString(ex.what()));
    }
}

} // namespace NeuralStudio
