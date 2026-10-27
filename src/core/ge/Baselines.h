#pragma once
// Baselines.h — non-BNF reference models of Mode 6 (Qt Core only).
//
// Every model is trained on the train file of a fold and measured on its test file
// with exactly the metrics of the GE rows:
//   classification: error % (class = nearest class value of the train file;
//                   binary: output >= 0.5 -> the larger class value)
//   regression:     mean squared error on the raw target scale.
// Inputs are min-max normalised with the train statistics, like every NeuralStudio network.
//
// Batch 1:  adam, bfgs, genetic  — MLP with H sigmoid hidden nodes (as in the BNF paper)
//           rbf                  — RBF network with H Gaussian centres
// Batch 2:  knn     — k nearest neighbours, k in {1,3,...,21} chosen on an 80/20 split of the train file
//           linear  — L2 logistic regression (classification) / ridge regression (regression),
//                     strength chosen on the 80/20 split
//           rf      — random forest, 100 CART trees (R randomForest defaults: mtry sqrt(d) / d/3,
//                     min leaf 1 / 5)
//           svm     — RBF-kernel C-SVC (one-vs-one) / epsilon-SVR (epsilon 0.1 on standardised targets),
//                     SMO with second-order working-set selection (as LIBSVM); gamma = 1/(d Var X),
//                     C in {1,10,100} chosen on the split (at most 2000/1000 rows), refitted on all rows
// Batch 3:  genetic — now the GA described in the BNF paper (keeps the best (1-ps)Nc, crossover
//                     a_i ~ U[-0.5,1.5], uniform re-draw with probability pm)
//           neat    — NEAT (Stanley & Miikkulainen 2002), feed-forward, 150 genomes, 200 generations
//           gnn     — the method of the BNF paper (uses the BNF grammar of its Figure 2): phase 1 GA,
//                     intervals [-F|w|, F|w|], phase 2 GE on bound programs with interval fitness
//                     (Ns = 50 random networks) and local search, phase 3 GA inside the bounds + BFGS.
//                     Not specified in the paper and set here: F = 2, 200 codons, tournament 4,
//                     phase-1 interval [-10, 10].
#include "core/dataset/Dataset.h"
#include <QString>
#include <QStringList>
#include <QVector>
#include <atomic>
#include <functional>

namespace NeuralStudio {

struct BaselineConfig {
    int     hidden = 10;            // H of the MLP / RBF baselines (paper: 10)
    // ADAM: mini-batch Adam (default hyper-parameters of Kingma & Ba)
    double  adamLr = 1e-3;
    int     adamBatch = 32;
    int     adamMinEpochs = 500;
    int     adamMinUpdates = 20000; // small datasets get more epochs so every run makes >= this many updates
    // BFGS: limited-memory BFGS from a random start
    int     bfgsMaxIters = 2000;
    // GENETIC: real-coded GA (paper Table 4: 500 chromosomes, 200 generations, ps 0.1, pm 0.05)
    int     gaPop = 500;
    int     gaGens = 200;
    double  gaSelection = 0.10;     // fraction of elites copied unchanged
    double  gaMutation = 0.05;
    double  gaBound = 10.0;         // weights searched in [-gaBound, gaBound]
    // RBF
    int     rbfKmeansIters = 100;
    double  rbfRidge = 1e-6;
    quint32 seed = 1;
};

struct BaselineRow {
    QString id;
    double  trainErr = 0;
    double  testErr = -1;
    int     params = 0;
    qint64  evals = 0;    // loss evaluations (a gradient counts as 3; Adam: 3 per sample seen / train size)
    double  seconds = 0;
    QString detail;       // chosen hyper-parameters etc.
};

// Ids implemented in this build, in report order.
QStringList geBaselineIds();

// Runs the requested baselines (unknown ids are skipped). `test` may be null (testErr = -1).
QVector<BaselineRow> geRunBaselines(const Dataset& train, const Dataset* test, const QStringList& ids,
                                    const BaselineConfig& cfg, std::atomic<bool>& stop,
                                    const std::function<void(int done, int total, const QString& id)>& progress);

} // namespace NeuralStudio
