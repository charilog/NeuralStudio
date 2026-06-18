#pragma once
#include "core/dataset/Dataset.h"
#include "core/nn/NeuralNetwork.h"
#include "core/nn/Trainer.h"
#include <QObject>
#include <atomic>

namespace NeuralStudio {

// ─── CrossValidator ───────────────────────────────────────────────────────────
//  Runs K-fold cross validation: trains K independent models, each on K-1 folds
//  with the remaining fold as validation. Reports average + std of the metrics.
//
//  Lives in its own QThread (same pattern as Trainer).
// ─────────────────────────────────────────────────────────────────────────────
struct FoldResult {
    int    fold       = 0;
    double finalLoss  = 0.0;
    double finalAcc   = -1.0;
};

struct CVSummary {
    int    k          = 0;
    double meanLoss   = 0.0;
    double stdLoss    = 0.0;
    double meanAcc    = -1.0;
    double stdAcc     = -1.0;
    std::vector<FoldResult> folds;
};

class CrossValidator : public QObject {
    Q_OBJECT
public:
    explicit CrossValidator(QObject* parent = nullptr);

    void setK              (int k)                       { m_k = k; }
    void setDataset        (const Dataset* ds)            { m_ds = ds; }
    void setNetworkConfig  (const NetworkConfig& cfg)     { m_netCfg = cfg; }
    void setTrainerConfig  (const TrainerConfig& cfg)     { m_trCfg = cfg; }

public slots:
    void run();
    void requestStop() { m_stop.store(true); }

signals:
    void foldCompleted(FoldResult result);
    void cvFinished   (CVSummary summary);
    void cvError      (QString msg);

private:
    int                m_k = 5;
    const Dataset*     m_ds = nullptr;
    NetworkConfig      m_netCfg;
    TrainerConfig      m_trCfg;
    std::atomic<bool>  m_stop{ false };
};

} // namespace NeuralStudio
