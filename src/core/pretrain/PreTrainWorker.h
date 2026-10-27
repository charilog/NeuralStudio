#pragma once
// PreTrainWorker.h — v4: multiple datasets, baseline comparison
#include "core/nn/NeuralNetwork.h"
#include "core/dataset/Dataset.h"
#include <QObject>
#include <QString>
#include <QStringList>
#include <QVector>
#include <atomic>
#include <memory>
#include <vector>

namespace NeuralStudio {

struct PhaseConfig {
    QString optimizer  = "ga";   // "none" = skip
    int     population = 200;
    int     iterations = 200;
    double  wBound     = 10.0;
};

struct PreTrainConfig {
    QStringList datasetPaths;          // ← multiple datasets
    QString hiddenLayout = "16";
    QString activation   = "relu";
    double  valSplit     = 0.2;

    // Phase 1: small pre-train network (Iw hidden nodes)
    int         Iw     = 10;
    // The baseline gets the same total number of evaluations as Phase 1 + Phase 2 (+ Phase 3),
    // so the comparison measures the pre-training and not the extra budget. false = the
    // baseline gets only the Phase-2 budget (the former behaviour, as in the original paper).
    bool        equalBudgetBaseline = true;
    PhaseConfig phase1;                // optimizer for Phase 1

    // Phase 2: fine-tune main network (warm-started from Phase 1)
    PhaseConfig phase2;

    // Phase 3: optional further fine-tune ("none" = skip)
    PhaseConfig phase3;
};

struct PreTrainRunResult {
    QString label;
    double  trainLoss = 0;
    double  trainErr  = 0;   // % (classification) or MSE (regression)
    double  valErr    = -1;  // -1 if no val split
    double  testErr   = -1;  // companion test file (never used for training or selection)
};

class PreTrainWorker : public QObject {
    Q_OBJECT
public:
    explicit PreTrainWorker(QObject* parent = nullptr);
    void setConfig(const PreTrainConfig& c) { m_cfg = c; }

public slots:
    void run();
    void requestStop() { m_stop.store(true); }

signals:
    void datasetStarted  (int dsIdx, int totalDs, QString dsName);
    void phaseStarted    (int phase, int totalIter, QString name);
    void iterationDone   (int phase, int iter, int totalIter,
                          double loss, double valLoss);
    void phaseFinished   (int phase, double finalLoss);
    void datasetFinished (QString dsName, QStringList configs,
                          QVector<double> trainLoss,
                          QVector<double> trainErr,
                          QVector<double> valErr,
                          QVector<double> testErr,    // companion .test file; -1 if none
                          bool isClassification);
    void allDone         (std::shared_ptr<NeuralNetwork> lastNet,
                          std::shared_ptr<Dataset>       lastDs);
    void errorOccurred   (QString msg);

private:
    std::shared_ptr<Dataset> m_testDs;   // companion test file of the current dataset

    PreTrainConfig    m_cfg;
    std::atomic<bool> m_stop{false};
    std::shared_ptr<NeuralNetwork> m_lastNet;
    std::shared_ptr<Dataset>       m_lastDs;

    bool runOnDataset(int dsIdx, const QString& dsPath);
    PreTrainRunResult runBaseline(NeuralNetwork* net, Dataset* ds,
                                  int phNum, const QString& dsName);
    PreTrainRunResult measureNet (NeuralNetwork* net, Dataset* ds) const;
};

} // namespace NeuralStudio
