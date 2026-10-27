#pragma once
// ArchSearchWorker.h — Mode 5: Evolutionary Architecture Search
// Outer optimizer searches architecture space; inner optimizer trains each candidate.
#include "core/nn/NeuralNetwork.h"
#include "core/dataset/Dataset.h"
#include <QObject>
#include <QString>
#include <QStringList>
#include <QVector>
#include <atomic>
#include <memory>

namespace NeuralStudio {

// ── Search configuration ──────────────────────────────────────────────────────
struct ArchSearchConfig {
    QStringList datasetPaths;
    // ── Search space ─────────────────────────────────────────────────────────
    int  minLayers = 1;
    int  maxLayers = 4;
    QVector<int>     candidateSizes = {4, 8, 16, 32, 64, 128};
    QVector<QString> candidateActs  = {"relu", "tanh", "sigmoid"};
    double valSplit = 0.2;
    // ── Outer optimizer (architecture search) ─────────────────────────────────
    QString outerOpt  = "ga";
    int     outerPop  = 30;
    int     outerGens = 50;
    // ── Inner optimizer (trains each candidate architecture) ──────────────────
    QString innerOpt    = "de";
    int     innerPop    = 50;
    int     innerIters  = 50;      // (150 here vs 50 in the panel; the panel value is the one used)
    double  innerWBound = 10.0;
    // bound-respecting L-BFGS after the inner optimizer, in evaluations (0 = off). Without it every
    // architecture got the same few thousand derivative-free evaluations whatever its number of
    // weights, so large networks were compared half-trained and the search favoured tiny ones.
    int     innerPolishEvals = 400;
    // ── Result ────────────────────────────────────────────────────────────────
    int topK = 10;
};

// ── Per-architecture result ───────────────────────────────────────────────────
struct ArchResult {
    int          rank       = 0;
    QString      desc;            // "2L: 64·ReLU → 32·Tanh"
    QVector<int>     sizes;
    QVector<QString> actNames;
    double       trainErr   = 0;  // error% (cls) or MSE (reg)
    double       valErr     = 0;  // lower = better (fitness)
    double       testErr    = -1; // companion test file (never used by the search); -1 = no test file
    double       trainLoss  = 0;
    int          numParams  = 0;
    int          evalNum    = 0;  // sequential unique-evaluation index
};

// ── Worker ────────────────────────────────────────────────────────────────────
class ArchSearchWorker : public QObject {
    Q_OBJECT
public:
    explicit ArchSearchWorker(QObject* parent = nullptr);
    void setConfig(const ArchSearchConfig& c) { m_cfg = c; }

public slots:
    void run();
    void requestStop() { m_stop.store(true); }

signals:
    // Emitted when a dataset starts being processed
    void datasetStarted(int dsIdx, int totalDs, QString dsName);

    // Emitted after every evaluate() call (cache hit or miss)
    // totalEvals = outerPop * outerGens; cacheHit = was this a repeat arch?
    void evalDone(int evalNum, int totalEvals,
                  double fitness, QString archDesc, bool cacheHit);

    // Emitted once per outer optimizer iteration (= one generation)
    void generationDone(int gen, int totalGens,
                        double bestFitness, QString bestDesc,
                        int uniqueEvals, int cacheHits);

    // Emitted after all generations finish for one dataset
    void datasetFinished(QString dsName,
                         QVector<ArchResult> topResults,
                         QVector<double>     convergence,
                         bool                isClassification);

    void allDone();
    void errorOccurred(QString msg);

private:
    ArchSearchConfig  m_cfg;
    std::atomic<bool> m_stop{false};

    bool runOnDataset(int dsIdx, const QString& dsPath);
};

} // namespace NeuralStudio
