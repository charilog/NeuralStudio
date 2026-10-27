#pragma once
// GEStudioWorker.h — Mode 6: Grammatical Evolution Studio.
//
// One BNF grammar derives a complete network specification:
//   architecture (layers, sizes, activations)
//   + per-layer bound commands (SHRINK / EXPAND ...) for the parameter intervals
//   + optional constructed features.
// Continuous parameters are NOT encoded in the chromosome: for every distinct
// phenotype an inner optimizer (DE, PSO, CMA-ES, ...) trains the weights inside
// the intervals the grammar produced. Fitness = validation error.
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

struct GEStudioConfig {
    QStringList datasetPaths;

    // ── Grammar ───────────────────────────────────────────────────────────────
    QString grammarText;            // empty → built-in unified grammar
    int     maxLayers   = 4;
    int     maxFeatures = 3;
    int     maxFeatureSize = 15;    // operations/operands per feature (after constant folding)

    // ── Outer evolution ───────────────────────────────────────────────────────
    QString genotype    = "ge";     // "ge" (codons) | "pge" (probabilistic)
    int     pop         = 40;
    int     gens        = 40;
    double  replaceRate = 0.30;
    double  mutationRate = 0.05;
    int     tournament  = 4;
    QString crossover   = "onepoint";   // "onepoint" | "local"
    double  localRate   = 0.10;
    int     localNcr    = 3;
    int     genomeLen   = 120;
    int     maxWraps    = 2;
    int     maxDepth    = 24;
    double  pgeAlpha    = 0.30;
    double  pgeElite    = 0.25;
    double  pgeFloor    = 0.02;
    bool    transferPrior = false;  // PGE: carry learned probabilities to the next dataset
    quint32 seed        = 0;        // 0 = random

    // ── Inner training of each phenotype ──────────────────────────────────────
    double  baseBound   = 10.0;     // "abs" mode: bound commands scale ±baseBound
    QString boundMode   = "fan";    // "fan": per-layer base = fanK*sqrt(6/(fanIn+fanOut)) (scale-aware); "abs": ±baseBound
    double  fanK        = 3.0;
    bool    innerPolish = true;     // inner training = global optimizer + projected L-BFGS (higher-fidelity fitness)
    int     innerPolishEvals = 300;
    int     lamarck     = 3;        // a revisited phenotype resumes training from its stored weights (up to N times)
    int     finalists   = 5;        // after the search the best N phenotypes are refined to the full Lamarckian depth and re-ranked
    qint64  totalBudget = 0;        // >0: fixed total budget T (loss evaluations) per run; the search stops at searchFrac*T
    double  searchFrac  = 0.5;      //     and GE rows are retrained with T - (search cost); Standard_T gets all of T
    double  polishFrac  = 0.5;      // fixed-budget rows: share of each row's budget given to bound-respecting L-BFGS
    QString innerOpt    = "de";
    int     innerPop    = 30;
    int     innerIters  = 60;
    double  valSplit    = 0.2;
    double  parsimony   = 0.0;      // fitness *= 1 + parsimony * params / 1000
    QString fitnessMetric = "loss"; // "loss" (validation loss, continuous) | "error" (validation error %)
    int     patience    = 15;       // stop when the best fitness has not improved for this many generations (0 = off)

    // ── Final comparison ──────────────────────────────────────────────────────
    int     topK        = 5;
    int     finalPop    = 80;
    int     finalIters  = 200;
    bool    finalPolish = true;     // unbounded L-BFGS on every final row alike
    bool    equalBudgetBaseline = true;
    bool    gridBaseline = true;    // best of a fixed architecture grid, searched with the GE search budget
    QString baselineHidden = "16";
    QString baselineAct    = "relu";
    QStringList methods;            // final rows to compute (ids, see GEReport); empty = all
    QString mode = "ge";
    // Self-adaptive extensions (fixed total budget only)
    bool    adaptiveSearch = false;  // "-sage" configurations: a generation only counts as progress when the best
                                     // validation loss improves by more than seRule standard errors (noise-aware stop)
    double  seRule = 1.0;            // standard errors for the noise-aware rules (selection, ensemble growth, search stop)
    int     ensPool = 12;            // candidate networks for the ensemble (best of the archive)
    int     ensMax = 8;              // most picks of the greedy ensemble selection (with replacement)            // "ge": BNF search + comparison rows | "baselines": only the reference models (Baselines.h)
};

struct GEPhenotype {
    int     rank = 0;
    QString desc;           // NetSpec::describe()
    double  fitness = 0;
    double  trainErr = 0;   // % (classification) or MSE (regression)
    double  valErr = 0;
    int     params = 0;
    int     inputs = 0;     // after feature construction
};

struct GEFinalRow {
    QString id;             // stable id: standard, standard_eq, grid_eq, ge1..geK, ge1_nobounds, ge1_eq, ge1_nobounds_eq;
                            // with totalBudget>0: standard, standard_T, grid_T, ge1, ge1_T, ge1_T_nobounds, ge1_T_nofeat
    QString label;
    double  trainErr = 0;
    double  testErr = -1;   // -1 when there is no companion test file
    double  valErr = -1;    // held-out part of the train file (when no test file)
    int     params = 0;
    qint64  evals = 0;      // loss evaluations spent on this row
    double  seconds = 0;
    bool    isGE = false;
};

struct GEDatasetResult {
    QString dsName;
    bool    isCls = false;
    bool    hasTest = false;
    int     trainSamples = 0;
    int     testSamples = 0;
    qint64  searchCalls = 0;     // loss evaluations spent by the whole GE search
    int     uniqueEvals = 0;     // distinct phenotypes trained
    int     cacheHits = 0;
    double  searchSeconds = 0;
    QVector<GEPhenotype> top;
    QVector<GEFinalRow>  rows;
    int                  gensRun = 0;        // generations actually executed
    int                  gensPlanned = 0;
    QVector<double>      convergence;   // best fitness per generation (incl. generation 0)
    QString              probsReport;   // learned production probabilities (PGE) or elite frequencies (GE)
    QString              bestProgram;   // text of the best derivation
};

class GEStudioWorker : public QObject {
    Q_OBJECT
public:
    explicit GEStudioWorker(QObject* parent = nullptr);
    void setConfig(const GEStudioConfig& c) { m_cfg = c; }

public slots:
    void run();
    void requestStop() { m_stop.store(true); }

signals:
    void datasetStarted (int dsIdx, int totalDs, QString dsName, QString info);
    void generationDone (int gen, int totalGens, double bestFitness, QString bestDesc,
                         double meanFitness, double diversity,
                         int uniqueEvals, int cacheHits, qint64 innerCalls);
    void finalStageStarted(int totalRows);
    void finalRowDone   (int row, int totalRows, QString label);
    void datasetFinished(GEDatasetResult result);
    void allDone        (std::shared_ptr<NeuralNetwork> lastNet,
                         std::shared_ptr<Dataset>       lastDs);
    void errorOccurred  (QString msg);

private:
    GEStudioConfig    m_cfg;
    std::atomic<bool> m_stop{false};
    std::shared_ptr<NeuralNetwork> m_lastNet;
    std::shared_ptr<Dataset>       m_lastDs;
    std::vector<std::vector<double>> m_prior;   // PGE probabilities carried across datasets

    bool runOnDataset(int dsIdx, const QString& dsPath);
};

} // namespace NeuralStudio
