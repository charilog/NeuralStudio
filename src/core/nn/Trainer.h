#pragma once
#include "NeuralNetwork.h"
#include "core/dataset/Dataset.h"
#include <QObject>
#include <atomic>

namespace NeuralStudio {

enum class OptimizerType { SGD, Adam, AdamW, RMSProp, Nadam, AdaGrad };
enum class LRSchedule { Constant, StepDecay, ExponentialDecay };

struct TrainerConfig {
    int           epochs              = 200;
    double        learningRate        = 0.001;
    int           batchSize           = 32;
    double        validationSplit     = 0.20;
    OptimizerType optimizer           = OptimizerType::Adam;
    double        adamBeta1           = 0.9;
    double        adamBeta2           = 0.999;
    double        adamEps             = 1e-8;
    double        weightDecay         = 0.01;   // AdamW
    double        rmsRho              = 0.9;    // RMSProp exponential decay
    int           earlyStoppingPatience = 0;

    // LR scheduling
    LRSchedule    lrSchedule          = LRSchedule::Constant;
    double        lrDecayRate         = 0.5;   // multiplier (StepDecay: per stepSize epochs; Exp: per epoch)
    int           lrDecayStepSize     = 50;    // only used for StepDecay
};

struct EpochResult {
    int    epoch     = 0;
    double trainLoss = 0.0;
    double valLoss   = 0.0;
    double trainAcc  = -1.0;
    double valAcc    = -1.0;
    double currentLR = 0.0;
};

class Trainer : public QObject {
    Q_OBJECT
public:
    explicit Trainer(QObject* parent = nullptr);
    void setConfig (const TrainerConfig& cfg) { m_cfg = cfg; }
    void setNetwork(NeuralNetwork* net)        { m_net = net; }
    void setDataset(const Dataset*  ds)        { m_ds  = ds;  }
    // External validation dataset (e.g. the companion .test file).
    // When set, ALL m_ds samples train and this dataset is used for val monitoring.
    // isTestSet = true: the dataset is the held-out TEST file, shown only as a
    // monitoring curve. Early stopping and best-epoch restoration are then
    // disabled, because choosing the epoch on the test file would leak it.
    void setValDataset(const Dataset* valDs, bool isTestSet = false) { m_valDs = valDs; m_valIsTest = isTestSet; }

public slots:
    void run();
    void requestStop() { m_stop.store(true); }

signals:
    void epochCompleted  (EpochResult result);
    void trainingFinished(double finalValLoss);
    void trainingError   (QString msg);

private:
    TrainerConfig     m_cfg;
    NeuralNetwork*    m_net = nullptr;
    const Dataset*    m_ds    = nullptr;
    const Dataset*    m_valDs = nullptr;   // optional external val dataset
    bool              m_valIsTest = false; // m_valDs is the test file (monitoring only)
    std::atomic<bool> m_stop{ false };

    // Cached (normalised inputs + encoded targets) indexed by row
    std::vector<std::vector<double>> m_normInputs;
    std::vector<std::vector<double>> m_targets;

    void   buildCache();
    void   trainBatch(const std::vector<int>& batchIdx, int adamStep);
    double lossOnSet (const std::vector<int>& idx) const;
    double sampleLoss(const std::vector<double>& out, const std::vector<double>& target) const;
    double accOnSet  (const std::vector<int>& idx) const;
    double lossOnExtDataset(const Dataset* ds) const;
    double accOnExtDataset (const Dataset* ds) const;
};

} // namespace NeuralStudio
