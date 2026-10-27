#pragma once
#include "core/nn/NeuralNetwork.h"
#include "core/nn/Trainer.h"
#include <QString>
#include <vector>

namespace NeuralStudio {

// ─── BatchJobConfig ───────────────────────────────────────────────────────────
//  One queued dataset + the architecture/training template it will use.
//  The architecture template is the SAME across all jobs in a batch; only the
//  input/output sizes adapt automatically per dataset.
// ─────────────────────────────────────────────────────────────────────────────
struct BatchJobConfig {
    QString  datasetPath;

    // Architecture template (input/output sizes filled in per-dataset)
    std::vector<int>        hiddenSizes;
    std::vector<Activation> hiddenActivations;
    double                  dropoutRate = 0.0;

    // Training config
    TrainerConfig trainerCfg;

    // Optimizer selection:
    //   "adam" / "sgd" → gradient-based (Trainer)
    //   "de" / "pso" / "cmaes" / … → evolutionary (MetaTrainer)
    QString optimizerName = "adam";
    double  weightBound   = 5.0;   // search-space bound for evolutionary methods
};

// ─── BatchJobResult ───────────────────────────────────────────────────────────
//  Everything we know about a finished job.  Empty / -1 fields are N/A for the
//  detected task type.
// ─────────────────────────────────────────────────────────────────────────────
struct BatchJobResult {
    // ── Identification ──
    QString datasetName;
    QString datasetPath;
    QString optimizerName;   // which optimizer was used for this run

    // ── Dataset structure ──
    int  samples     = 0;
    int  inputs      = 0;
    int  outputs     = 0;
    int  numClasses  = 0;
    bool hasTestSet  = false;
    int  testSamples = 0;

    TaskType task = TaskType::Regression;

    // ── Training metrics ──
    int    epochsRun     = 0;
    double durationMs    = 0.0;
    double finalTrainLoss = 0.0;
    double finalValLoss   = 0.0;
    double bestValLoss    = 0.0;
    int    bestEpoch      = 0;

    // ── Classification (accuracy 0..1, error % 0..100) ──
    //   -1 if not applicable
    double trainAcc       = -1.0;
    double valAcc         = -1.0;
    double testAcc        = -1.0;
    double trainErrorPct  = -1.0;
    double valErrorPct    = -1.0;
    double testErrorPct   = -1.0;

    // ── Regression metrics ──
    //   -1 if not applicable
    double trainMAE  = -1.0;
    double valMAE    = -1.0;
    double testMAE   = -1.0;
    double trainRMSE = -1.0;
    double valRMSE   = -1.0;
    double testRMSE  = -1.0;
    double trainR2   = -1.0;
    double valR2     = -1.0;
    double testR2    = -1.0;

    // ── Status ──
    bool    success  = false;
    QString errorMsg;
};

// ─── Pretty-printers ──────────────────────────────────────────────────────────
inline QString taskName(TaskType t) {
    switch (t) {
        case TaskType::BinaryClassification:     return "Binary";
        case TaskType::MultiClassClassification: return "Multi-class";
        default:                                 return "Regression";
    }
}

} // namespace NeuralStudio
