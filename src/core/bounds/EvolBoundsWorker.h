#pragma once
// EvolBoundsWorker.h — v6: multiple datasets (like BatchPanel)
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

struct EvolBoundsConfig {
    QStringList datasetPaths;        // ← multiple datasets
    QString hiddenLayout = "16";
    QString activation   = "relu";
    // Phase 1: penalised optimizer → discovers per-parameter bounds
    QString opt1         = "ga";
    int     Nc1          = 500;
    int     Ng1          = 200;
    double  lambda       = 2.0;
    double  a            = 5.0;
    double  f            = 3.0;
    double  initialBound = 10.0;     // ± symmetric search range for Phase 1 & baseline
    // Phase 2: training within Phase-1 bounds  ("none" = skip)
    QString opt2         = "de";    // or "ge_grammar"
    int     Nc2          = 200;
    int     Ng2          = 200;
    int     Ns_ge        = 30;
    int     Nl_ge        = 10;
    double  pl_ge        = 0.05;
    // Phase 3: final training  ("none" = skip)
    QString opt3         = "ga";
    int     Nc3          = 500;
    int     Ng3          = 200;
    bool    bfgsEnd      = true;
    double  valSplit     = 0.2;
    // The baseline ("Standard") gets as many evaluations as Phases 1 + 2 + 3 (+ BFGS) together,
    // so the table compares the bounds and not the larger budget. false = the baseline gets the
    // Phase-2 budget only (the former behaviour).
    bool    equalBudgetBaseline = true;
};

class EvolBoundsWorker : public QObject {
    Q_OBJECT
public:
    explicit EvolBoundsWorker(QObject* parent = nullptr);
    void setConfig(const EvolBoundsConfig& c) { m_cfg = c; }

public slots:
    void run();
    void requestStop() { m_stop.store(true); }

signals:
    // ── Per-dataset ────────────────────────────────────────────────────────
    // Emitted when processing a new dataset begins
    void datasetStarted (int dsIdx, int totalDs, QString dsName);

    // Emitted each time a phase begins (name includes dataset name)
    void phaseStarted   (int phase, int totalIter, QString name);
    void iterationDone  (int phase, int iter, int totalIter,
                         double loss, double valLoss);
    void phaseFinished  (int phase, double finalLoss);

    // Emitted after Phase 1 of each dataset: per-parameter lb/rb for bounds chart
    void datasetBoundsReady (QString dsName,
                              QVector<double> lb,
                              QVector<double> rb,
                              double          initialBound);

    // Emitted when one dataset completes all phases: comparison data
    void datasetFinished (QString      dsName,
                          QStringList  configs,       // "Standard", "Ph.1+2", "Ph.1+2+3"
                          QVector<double> trainLoss,
                          QVector<double> trainErr,   // % (classif) or MSE (regress)
                          QVector<double> valErr,     // -1 if no val split
                          QVector<double> testErr,    // on the companion .test file; -1 if none
                          bool         isClassification);

    // Emitted when ALL datasets are done
    void allDone       (std::shared_ptr<NeuralNetwork> lastNet,
                        std::shared_ptr<Dataset>       lastDs);
    void errorOccurred (QString msg);

private:
    EvolBoundsConfig  m_cfg;
    std::atomic<bool> m_stop{false};
    std::shared_ptr<NeuralNetwork> m_lastNet;
    std::shared_ptr<Dataset>       m_lastDs;

    struct Bounds { std::vector<double> lb, ub; };

    // Runs the full 3-phase workflow on one dataset;
    // returns false if stopped early
    bool runOnDataset(int dsIdx, const QString& dsPath);

    struct RunResult {
        double trainLoss = 0;
        double trainErr  = 0;
        double valErr    = -1;
        double testErr   = -1;   // companion test file (never used for training or selection)
    };
    std::shared_ptr<Dataset> m_testDs;   // companion test file of the current dataset

    RunResult runBaseline (NeuralNetwork* net, Dataset* ds, int phNum,
                           const QString& dsName);
    Bounds    runPhase1   (NeuralNetwork* net, Dataset* ds, int phNum,
                           const QString& dsName);
    Bounds    runPhase2Opt(NeuralNetwork* net, Dataset* ds,
                           const Bounds& b, int phNum);
    Bounds    runPhase2GE (NeuralNetwork* net, Dataset* ds,
                           const Bounds& b, int phNum);
    void      runPhase3   (NeuralNetwork* net, Dataset* ds,
                           const Bounds& b, int phNum);
    RunResult measureNet  (NeuralNetwork* net, Dataset* ds) const;
};

} // namespace NeuralStudio
