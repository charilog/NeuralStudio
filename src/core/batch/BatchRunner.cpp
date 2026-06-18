#include "BatchRunner.h"

#include "core/dataset/DatasetLoader.h"
#include "core/nn/Trainer.h"
#include "core/nn/NeuralNetwork.h"
#include "core/optimizers/MetaTrainer.h"
#include "utils/Logger.h"

#include <QFileInfo>
#include <QElapsedTimer>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

namespace NeuralStudio {

BatchRunner::BatchRunner(QObject* parent) : QObject(parent) {}

// ─── Evaluate metrics on a dataset (no training, just inference) ─────────────
namespace {

struct EvalMetrics {
    // Classification
    double accuracy = -1.0;
    double errorPct = -1.0;
    // Regression
    double mae      = -1.0;
    double rmse     = -1.0;
    double r2       = -1.0;
};

EvalMetrics evaluate(NeuralNetwork* net, const Dataset* ds) {
    EvalMetrics m;
    if (!net || !ds || ds->sampleCount == 0) return m;

    const int n = ds->sampleCount;
    const TaskType task = net->taskType();
    const double midOut = 0.5 * (net->outputMin() + net->outputMax());

    if (task == TaskType::BinaryClassification) {
        int correct = 0;
        for (int r = 0; r < n; ++r) {
            int tc = (ds->outputAt(r, 0) > midOut) ? 1 : 0;
            auto pred = net->predict(ds->inputRow(r));
            int pc = (pred[0] >= 0.5) ? 1 : 0;
            if (pc == tc) ++correct;
        }
        m.accuracy = static_cast<double>(correct) / n;
        m.errorPct = (1.0 - m.accuracy) * 100.0;
    }
    else if (task == TaskType::MultiClassClassification) {
        int correct = 0;
        for (int r = 0; r < n; ++r) {
            int tc = net->classIndex(ds->outputAt(r, 0));
            auto pred = net->predict(ds->inputRow(r));
            int pc = static_cast<int>(std::max_element(pred.begin(), pred.end()) - pred.begin());
            if (pc == tc) ++correct;
        }
        m.accuracy = static_cast<double>(correct) / n;
        m.errorPct = (1.0 - m.accuracy) * 100.0;
    }
    else {
        // Regression — compute MAE, RMSE, R²
        double sumAbs = 0.0, sumSq = 0.0, sumTarget = 0.0;
        for (int r = 0; r < n; ++r) sumTarget += ds->outputAt(r, 0);
        const double meanT = sumTarget / n;

        double ssRes = 0.0, ssTot = 0.0;
        for (int r = 0; r < n; ++r) {
            const double t   = ds->outputAt(r, 0);
            const auto   pr  = net->predict(ds->inputRow(r));
            const double err = pr[0] - t;
            sumAbs += std::abs(err);
            sumSq  += err * err;
            ssRes  += err * err;
            ssTot  += (t - meanT) * (t - meanT);
        }
        m.mae  = sumAbs / n;
        m.rmse = std::sqrt(sumSq / n);
        m.r2   = (ssTot > 1e-12) ? 1.0 - ssRes / ssTot : 0.0;
    }
    return m;
}

} // anonymous


// ─── run ─────────────────────────────────────────────────────────────────────
void BatchRunner::run() {
    try {
        for (int i = 0; i < static_cast<int>(m_jobs.size()) && !m_stop.load(); ++i) {
            const auto& job = m_jobs[i];
            const QString dsName = QFileInfo(job.datasetPath).fileName();
            emit jobStarted(i, dsName);

            BatchJobResult res;
            res.datasetPath = job.datasetPath;
            res.datasetName   = QFileInfo(job.datasetPath).baseName();
            res.optimizerName = job.optimizerName;

            QElapsedTimer timer;
            timer.start();

            try {
                // ── Load dataset + companion ──────────────────────────────────
                DatasetBundle bundle = DatasetLoader::loadBundle(job.datasetPath);
                if (!bundle.train) throw std::runtime_error("Train dataset is null.");

                res.samples     = bundle.train->sampleCount;
                res.inputs      = bundle.train->inputCount;
                res.outputs     = bundle.train->outputCount;
                res.hasTestSet  = bundle.hasTest();
                res.testSamples = bundle.hasTest() ? bundle.test->sampleCount : 0;

                // ── Detect task ───────────────────────────────────────────────
                const TaskType task = detectTask(bundle.train.get());
                res.task = task;

                auto uniqueOuts = bundle.train->uniqueOutputValues(0);
                res.numClasses = (task != TaskType::Regression)
                                 ? static_cast<int>(uniqueOuts.size()) : 0;

                // ── Build network ─────────────────────────────────────────────
                NetworkConfig netCfg;
                netCfg.task               = task;
                netCfg.hiddenSizes        = job.hiddenSizes;
                netCfg.hiddenActivations  = job.hiddenActivations;
                netCfg.dropoutRate        = job.dropoutRate;

                int netOutSize;
                if (task == TaskType::MultiClassClassification)
                    netOutSize = static_cast<int>(uniqueOuts.size());
                else if (task == TaskType::BinaryClassification)
                    netOutSize = 1;
                else
                    netOutSize = bundle.train->outputCount;

                auto net = std::make_unique<NeuralNetwork>();
                net->build(bundle.train->inputCount, netOutSize, netCfg);
                net->setNormalization(bundle.train->inputStats);
                if (task == TaskType::MultiClassClassification)
                    net->setClassValues(std::vector<double>(uniqueOuts.begin(), uniqueOuts.end()));
                if (uniqueOuts.size() >= 2)
                    net->setOutputMapping(uniqueOuts.front(), uniqueOuts.back());

                // ── Train (synchronously, since we're on a worker thread) ─────
                const QString optimName = job.optimizerName.toLower();
                static const QSet<QString> gdSet{"adam","sgd","adamw","rmsprop","nadam","adagrad"};
                const bool isEvo = !gdSet.contains(optimName);

                double bestVL = std::numeric_limits<double>::max();
                int    bestEp = 0;
                int    lastEp = 0;
                double lastTL = 0.0, lastVL = 0.0;
                double lastTA = -1.0, lastVA = -1.0;
                const int totalEpochs = job.trainerCfg.epochs;

                // Shared epoch-capture lambda (works for both Trainer and MetaTrainer)
                auto onEpoch = [&](EpochResult e) {
                    lastEp = e.epoch;
                    lastTL = e.trainLoss;
                    lastVL = e.valLoss;
                    lastTA = e.trainAcc;
                    lastVA = e.valAcc;
                    if (e.valLoss >= 0 && e.valLoss < bestVL) {
                        bestVL = e.valLoss; bestEp = e.epoch;
                    }
                    if ((e.epoch % 10 == 0) || e.epoch == totalEpochs)
                        emit jobProgress(i, e.epoch, totalEpochs, e.trainLoss);
                };
                auto onError = [](QString msg) {
                    throw std::runtime_error(msg.toStdString());
                };

                if (!isEvo) {
                    // ── Gradient-based ────────────────────────────────────────
                    Trainer trainer;
                    TrainerConfig tc = job.trainerCfg;
                    if      (optimName=="sgd")     tc.optimizer=OptimizerType::SGD;
                    else if (optimName=="adamw")   tc.optimizer=OptimizerType::AdamW;
                    else if (optimName=="rmsprop") tc.optimizer=OptimizerType::RMSProp;
                    else if (optimName=="nadam")   tc.optimizer=OptimizerType::Nadam;
                    else if (optimName=="adagrad") tc.optimizer=OptimizerType::AdaGrad;
                    else                           tc.optimizer=OptimizerType::Adam;
                    trainer.setConfig(tc);
                    trainer.setNetwork(net.get());
                    trainer.setDataset(bundle.train.get());
                    QObject::connect(&trainer, &Trainer::epochCompleted, this,
                        [&](EpochResult e){ onEpoch(e); }, Qt::DirectConnection);
                    QObject::connect(&trainer, &Trainer::trainingError, this,
                        [&](QString m){ onError(m); }, Qt::DirectConnection);
                    trainer.run();
                } else {
                    // ── All other methods (quasi-Newton, single-point, evolutionary) ─
                    MetaTrainer metaTrainer;
                    metaTrainer.setConfig(job.trainerCfg);
                    metaTrainer.setOptimizerName(optimName);
                    metaTrainer.setNetwork(net.get());
                    metaTrainer.setDataset(bundle.train.get());
                    metaTrainer.setWeightBound(job.weightBound);
                    QObject::connect(&metaTrainer, &MetaTrainer::epochCompleted, this,
                        [&](EpochResult e){ onEpoch(e); }, Qt::DirectConnection);
                    QObject::connect(&metaTrainer, &MetaTrainer::trainingError, this,
                        [&](QString m){ onError(m); }, Qt::DirectConnection);
                    metaTrainer.run();
                }

                res.epochsRun      = lastEp;
                res.finalTrainLoss = lastTL;
                res.finalValLoss   = lastVL;
                res.bestValLoss    = (bestEp > 0) ? bestVL : lastVL;
                res.bestEpoch      = (bestEp > 0) ? bestEp : lastEp;

                // ── Evaluate on the FULL train set (after training) ───────────
                // Note: trainer's reported metrics use shuffled splits; we
                // re-evaluate on the whole dataset for a clean number, plus
                // on the test set if present.
                net->setTrainingMode(false);

                auto trainMetrics = evaluate(net.get(), bundle.train.get());
                if (task == TaskType::Regression) {
                    res.trainMAE  = trainMetrics.mae;
                    res.trainRMSE = trainMetrics.rmse;
                    res.trainR2   = trainMetrics.r2;
                } else {
                    res.trainAcc      = trainMetrics.accuracy;
                    res.trainErrorPct = trainMetrics.errorPct;
                }

                // We can also report the trainer's last val accuracy as "valAcc"
                if (task != TaskType::Regression && lastVA >= 0) {
                    res.valAcc      = lastVA;
                    res.valErrorPct = (1.0 - lastVA) * 100.0;
                }

                if (bundle.hasTest()) {
                    auto testMetrics = evaluate(net.get(), bundle.test.get());
                    if (task == TaskType::Regression) {
                        res.testMAE  = testMetrics.mae;
                        res.testRMSE = testMetrics.rmse;
                        res.testR2   = testMetrics.r2;
                    } else {
                        res.testAcc      = testMetrics.accuracy;
                        res.testErrorPct = testMetrics.errorPct;
                    }
                }

                res.durationMs = timer.elapsed();
                res.success    = true;

            } catch (const std::exception& ex) {
                res.success     = false;
                res.errorMsg    = QString::fromStdString(ex.what());
                res.durationMs  = timer.elapsed();
                NS_WARN << "Batch job" << dsName << "failed:" << ex.what();
            }

            emit jobCompleted(i, res);
        }

        emit allJobsFinished();
    } catch (const std::exception& ex) {
        emit batchError(QString::fromStdString(ex.what()));
    }
}

} // namespace NeuralStudio
