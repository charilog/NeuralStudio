#pragma once
#include "optimizer.h"
#include "core/nn/NeuralNetwork.h"
#include "core/dataset/Dataset.h"
#include <limits>
#include <numeric>

namespace NeuralStudio {

// ─── NSProblem ────────────────────────────────────────────────────────────────
//  Wraps a NeuralNetwork + training Dataset as an optimsolution::Problem.
//
//  Decision variable x  = all network weights + biases, flattened into one
//                         vector of size = NeuralNetwork::totalParams().
//  Objective f(x)       = mean training loss (BCE for classification, MSE
//                         for regression) computed by a full forward pass.
//
//  Bounds: [-weightBound, +weightBound] for every parameter.
//          Defaults to 5.0 (wide enough to cover well-trained solutions yet
//          constrained enough for efficient evolutionary search).
//
//  Usage:
//    NSProblem prob(net, ds, 5.0, 0.2);
//    optimizer.setup(&prob, maxEvals, popSize, seed);
//    optimizer.init();
//    while (!optimizer.done()) optimizer.one_iteration();
//    NSProblem::unflatten(*net, optimizer.bestX());
// ─────────────────────────────────────────────────────────────────────────────
class NSProblem : public optimsolution::Problem {
public:
    NSProblem(NeuralNetwork* net, const Dataset* ds,
              double weightBound = 5.0,
              double valSplit    = 0.2,
              unsigned splitSeed = 42);

    // ── optimsolution::Problem interface ─────────────────────────────────────
    int                         dimension() const override { return m_dim;   }
    const std::vector<double>&  lb()        const override { return m_lb;   }
    const std::vector<double>&  ub()        const override { return m_ub;   }
    double evaluate(const std::vector<double>& x)  override;
    int    calls()  const override                         { return m_calls; }

    // Analytical (loss, gradient) via backpropagation — for L-BFGS
    std::pair<double, std::vector<double>>
    evalAndGrad(const std::vector<double>& x) override;

    // The same loss with its exact gradient (mean over the training rows), computed in parallel over the
    // samples with thread-local buffers (no layer state is touched). Used by the self-adaptive training.
    std::pair<double, std::vector<double>>
    evalAndGradParallel(const std::vector<double>& x);

    // Current network weights (for warm-start)
    std::vector<double> currentWeights() const { return flatten(*m_net); }

    // ── Monitoring (called by MetaTrainer after each iteration) ──────────────
    //  These use the CURRENT network weights (best so far, already applied).
    double valLoss() const;
    double valAcc()  const;   // -1 if regression
    double trainAcc() const;  // -1 if regression

    // Optional external validation dataset (e.g. the held-out .test file).
    // When set, ALL m_ds samples go to training and valLoss()/valAcc()
    // evaluate on this external dataset. Call after construction.
    void setExternalValDataset(const Dataset* valDs);

    // K-fold partition instead of the random hold-out: the rows are shuffled with the
    // split seed and fold k (0..K-1) is the validation set, the other folds train.
    // Used by cross-validation committees (every row is validated by exactly one member).
    void setFoldSplit(int k, int K);

    // ── Weight serialisation helpers ─────────────────────────────────────────
    static std::vector<double> flatten  (const NeuralNetwork& net);
    static void                unflatten(NeuralNetwork& net,
                                         const std::vector<double>& w);

    double bestLoss() const { return m_bestLoss; }

    // ── Predictions for ensembles / noise-aware selection (CURRENT weights) ──
    //  Per-sample predictions in probability space (softmax probabilities, the
    //  sigmoid output, or the regression outputs) and the encoded targets, for
    //  the validation rows (or the external dataset when set) or the train rows.
    //  Averaging predictions of several networks and scoring them with
    //  predLoss / predCorrect gives exactly the metrics of a single network.
    void   predictions(bool validation, std::vector<std::vector<double>>& pred,
                       std::vector<std::vector<double>>& target) const;
    double predLoss   (const std::vector<double>& p, const std::vector<double>& t) const;
    bool   predCorrect(const std::vector<double>& p, const std::vector<double>& t) const;
    bool   isClassification() const { return m_isCls; }

private:
    NeuralNetwork*  m_net;
    const Dataset*  m_ds;
    int             m_dim;
    int             m_calls = 0;
    double          m_bestLoss = std::numeric_limits<double>::infinity();

    std::vector<double> m_lb, m_ub;

    // Fixed train/val partition (reproducible, set once in constructor)
    std::vector<int> m_trainIdx;
    std::vector<int> m_valIdx;

    // Pre-normalised inputs and encoded targets (both train + val)
    std::vector<std::vector<double>> m_normInputs;
    std::vector<std::vector<double>> m_targets;

    // Meta
    bool   m_isCls  = false;
    double m_midOut = 0.5;

    const Dataset* m_externalValDs = nullptr;
    unsigned       m_splitSeed = 42;

    void   buildCache(double valSplit);
    double lossOnIdx(const std::vector<int>& idx) const;
    double accOnIdx (const std::vector<int>& idx) const;
    double lossOnDataset(const Dataset* ds) const;
    double accOnDataset (const Dataset* ds) const;
};

} // namespace NeuralStudio
