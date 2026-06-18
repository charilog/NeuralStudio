#pragma once
#include "RBFNetwork.h"
#include "core/dataset/Dataset.h"
#include "core/nn/Trainer.h"   // for EpochResult
#include <QObject>
#include <memory>
#include <random>

namespace NeuralStudio {

// ─── RBFTrainer ───────────────────────────────────────────────────────────────
//  One-pass analytic training for RBFNetwork:
//
//   Phase 1 — Centre selection  (K-means++ initialisation + Lloyd iterations)
//   Phase 2 — Width computation  (based on RBFWidthStrategy)
//   Phase 3 — Output weights     (ridge regression via Cholesky)
//
//  Training is FAST (seconds, not minutes) regardless of network depth.
// ─────────────────────────────────────────────────────────────────────────────
class RBFTrainer : public QObject {
    Q_OBJECT
public:
    using Vec = std::vector<double>;
    using Mat = std::vector<Vec>;

    explicit RBFTrainer(QObject* parent = nullptr);

    void setNetwork(std::shared_ptr<RBFNetwork> net) { m_net = net; }
    void setDataset(const Dataset* ds)               { m_ds  = ds;  }

    // Call in a worker thread
    void run();

signals:
    void progressMessage(QString msg);
    void trainingCompleted(double trainError, double valError, double trainAcc, double valAcc);
    void trainingError(QString msg);

private:
    // K-means++ initialisation, then Lloyd iterations
    void kMeans(const Mat& X, int k, Mat& centres);

    // Width computation
    Vec computeWidths(const Mat& centres);

    // Cholesky decomposition of SPD matrix A  (in-place lower triangular)
    bool cholesky(Mat& A) const;
    // Solve A x = b  given Cholesky factor L  (in-place update of b)
    Vec  choleskySolve(const Mat& L, Vec b) const;

    // Ridge regression:  solve (ΦᵀΦ + λI) W = ΦᵀY  for W  (k+1 × nOut)
    Mat ridgeRegression(const Mat& Phi, const Mat& Y, double lambda);

    // Compute prediction error and accuracy on a subset
    std::pair<double,double> evaluate(const Mat& X, const Mat& Y);

    // Helpers
    static double dist2(const Vec& a, const Vec& b);

    std::shared_ptr<RBFNetwork> m_net;
    const Dataset*              m_ds = nullptr;
    std::mt19937                m_rng{ std::random_device{}() };
};

} // namespace NeuralStudio
