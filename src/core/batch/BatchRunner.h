#pragma once
#include "BatchJob.h"
#include <QObject>
#include <QStringList>
#include <atomic>
#include <vector>

namespace NeuralStudio {

// ─── BatchRunner ──────────────────────────────────────────────────────────────
//  QObject worker that loads each dataset, builds a fresh network, trains it,
//  evaluates it (incl. test set if a companion file exists), and emits a
//  result for each job.
//
//  Lives on its own QThread (same pattern as Trainer / CrossValidator).
// ─────────────────────────────────────────────────────────────────────────────
class BatchRunner : public QObject {
    Q_OBJECT
public:
    explicit BatchRunner(QObject* parent = nullptr);

    void setJobs(const std::vector<BatchJobConfig>& jobs) { m_jobs = jobs; }

public slots:
    void run();
    void requestStop() { m_stop.store(true); }

signals:
    void jobStarted     (int index, QString datasetName);
    void jobProgress    (int index, int epoch, int totalEpochs, double trainLoss);
    void jobCompleted   (int index, BatchJobResult result);
    void allJobsFinished();
    void batchError     (QString msg);

private:
    std::vector<BatchJobConfig> m_jobs;
    std::atomic<bool>           m_stop{ false };
};

} // namespace NeuralStudio
