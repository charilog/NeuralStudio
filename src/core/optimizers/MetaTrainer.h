#pragma once
#include "core/nn/NeuralNetwork.h"
#include "core/nn/Trainer.h"        // for EpochResult, TrainerConfig
#include "core/dataset/Dataset.h"
#include <QObject>
#include <QString>
#include <atomic>
#include <memory>

namespace optimsolution { class Optimizer; }

namespace NeuralStudio {

// ─── MetaTrainer ──────────────────────────────────────────────────────────────
//  A drop-in replacement for Trainer that drives one of the optimsolution
//  metaheuristic optimisers (DE, PSO, CMA-ES, JSO, …) instead of gradient
//  descent.
//
//  Emits the same signals as Trainer so the TrainingPanel chart works without
//  changes.
//
//  Mapping of TrainerConfig fields to evolutionary concepts:
//    epochs    → number of optimizer iterations (= generations)
//    batchSize → population size
//    valSplit  → validation fraction (for monitoring only; loss computed on
//                the entire training set)
//    earlyStoppingPatience → stops when val-loss does not improve
//
//  LR, LR schedule, and Adam params are ignored.
// ─────────────────────────────────────────────────────────────────────────────
class MetaTrainer : public QObject {
    Q_OBJECT
public:
    explicit MetaTrainer(QObject* parent = nullptr);

    void setConfig     (const TrainerConfig& cfg)    { m_cfg = cfg; }
    void setOptimizerName(const QString& name)        { m_name = name; }
    void setNetwork    (NeuralNetwork* net)           { m_net = net; }
    void setDataset    (const Dataset*  ds)           { m_ds  = ds;  }
    void setWeightBound(double bound)                 { m_wBound = bound; }

public slots:
    void run();
    void requestStop() { m_stop.store(true); }

signals:
    void epochCompleted  (EpochResult result);
    void trainingFinished(double finalValLoss);
    void trainingError   (QString msg);

private:
    std::unique_ptr<optimsolution::Optimizer> createOptimizer() const;

    TrainerConfig       m_cfg;
    QString             m_name;
    NeuralNetwork*      m_net    = nullptr;
    const Dataset*      m_ds     = nullptr;
    double              m_wBound = 5.0;
    std::atomic<bool>   m_stop{false};
};

} // namespace NeuralStudio
