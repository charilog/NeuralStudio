#include "RBFTrainer.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>

namespace NeuralStudio {

RBFTrainer::RBFTrainer(QObject* parent) : QObject(parent) {}

// ─── dist2 ────────────────────────────────────────────────────────────────────
double RBFTrainer::dist2(const Vec& a, const Vec& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) { double d = a[i]-b[i]; s += d*d; }
    return s;
}

// ─── kMeans ───────────────────────────────────────────────────────────────────
void RBFTrainer::kMeans(const Mat& X, int k, Mat& centres) {
    const int N = int(X.size());
    const int D = int(X[0].size());
    k = std::min(k, N);

    // K-means++ initialisation
    centres.clear(); centres.reserve(k);
    std::uniform_int_distribution<int> uni(0, N-1);
    centres.push_back(X[uni(m_rng)]);

    std::vector<double> minDist(N, 1e300);
    for (int ci = 1; ci < k; ++ci) {
        // Update min distances
        double total = 0;
        for (int n = 0; n < N; ++n) {
            double d = dist2(X[n], centres.back());
            if (d < minDist[n]) minDist[n] = d;
            total += minDist[n];
        }
        // Sample proportional to distance²
        std::uniform_real_distribution<double> ud(0.0, total);
        double rnd = ud(m_rng);
        double cum = 0;
        int chosen = 0;
        for (int n = 0; n < N; ++n) {
            cum += minDist[n];
            if (cum >= rnd) { chosen = n; break; }
        }
        centres.push_back(X[chosen]);
    }

    // Lloyd's iterations
    std::vector<int> labels(N, 0);
    const int& maxIter = m_net->config().kmeansMaxIter;
    for (int iter = 0; iter < maxIter; ++iter) {
        bool changed = false;
        // Assignment step
        for (int n = 0; n < N; ++n) {
            double best = 1e300; int bestC = 0;
            for (int c = 0; c < k; ++c) {
                double d = dist2(X[n], centres[c]);
                if (d < best) { best = d; bestC = c; }
            }
            if (bestC != labels[n]) { labels[n] = bestC; changed = true; }
        }
        if (!changed) break;
        // Update step
        Mat newC(k, Vec(D, 0.0));
        std::vector<int> cnt(k, 0);
        for (int n = 0; n < N; ++n) {
            int c = labels[n]; ++cnt[c];
            for (int d = 0; d < D; ++d) newC[c][d] += X[n][d];
        }
        for (int c = 0; c < k; ++c) {
            if (cnt[c] > 0)
                for (int d = 0; d < D; ++d) newC[c][d] /= cnt[c];
            else
                newC[c] = X[uni(m_rng)]; // reinit empty cluster
        }
        centres = newC;
    }
}

// ─── computeWidths ────────────────────────────────────────────────────────────
RBFTrainer::Vec RBFTrainer::computeWidths(const Mat& centres) {
    const int k = int(centres.size());
    Vec widths(k, 1.0);

    switch (m_net->config().widthStrategy) {
    case RBFWidthStrategy::MaxDist: {
        // σ = d_max / sqrt(2k), same for all centres
        double dmax = 0;
        for (int i = 0; i < k; ++i)
            for (int j = i+1; j < k; ++j)
                dmax = std::max(dmax, std::sqrt(dist2(centres[i], centres[j])));
        double sigma = (k > 1) ? dmax / std::sqrt(2.0 * k) : 1.0;
        sigma = std::max(sigma, 1e-6);
        for (double& s : widths) s = sigma;
        break;
    }
    case RBFWidthStrategy::NearestNeighbor: {
        for (int i = 0; i < k; ++i) {
            double minD = 1e300;
            for (int j = 0; j < k; ++j)
                if (j != i) minD = std::min(minD, std::sqrt(dist2(centres[i], centres[j])));
            widths[i] = (k > 1) ? std::max(minD, 1e-6) : 1.0;
        }
        break;
    }
    case RBFWidthStrategy::MeanNN: {
        for (int i = 0; i < k; ++i) {
            std::vector<double> ds;
            for (int j = 0; j < k; ++j)
                if (j != i) ds.push_back(std::sqrt(dist2(centres[i], centres[j])));
            std::sort(ds.begin(), ds.end());
            int take = std::min(3, int(ds.size()));
            double mean = 0;
            for (int t = 0; t < take; ++t) mean += ds[t];
            widths[i] = (take > 0) ? std::max(mean/take, 1e-6) : 1.0;
        }
        break;
    }
    }
    return widths;
}

// ─── cholesky ────────────────────────────────────────────────────────────────
bool RBFTrainer::cholesky(Mat& A) const {
    const int n = int(A.size());
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = A[i][j];
            for (int k = 0; k < j; ++k) s -= A[i][k] * A[j][k];
            if (i == j) {
                if (s <= 1e-14) s = 1e-14;
                A[i][j] = std::sqrt(s);
            } else {
                A[i][j] = s / A[j][j];
            }
        }
        for (int j = i+1; j < n; ++j) A[i][j] = 0.0;   // zero upper triangle
    }
    return true;
}

RBFTrainer::Vec RBFTrainer::choleskySolve(const Mat& L, Vec b) const {
    const int n = int(L.size());
    // Forward substitution: L y = b
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) b[i] -= L[i][j] * b[j];
        b[i] /= L[i][i];
    }
    // Backward substitution: Lᵀ x = y
    for (int i = n-1; i >= 0; --i) {
        for (int j = i+1; j < n; ++j) b[i] -= L[j][i] * b[j];
        b[i] /= L[i][i];
    }
    return b;
}

// ─── ridgeRegression ─────────────────────────────────────────────────────────
RBFTrainer::Mat RBFTrainer::ridgeRegression(const Mat& Phi, const Mat& Y, double lambda) {
    const int N  = int(Phi.size());
    const int km1 = int(Phi[0].size());   // k+1
    const int nOut = int(Y[0].size());

    // A = ΦᵀΦ + λI    (km1 × km1)
    Mat A(km1, Vec(km1, 0.0));
    for (int i = 0; i < km1; ++i)
        for (int j = 0; j <= i; ++j) {
            double s = 0;
            for (int n = 0; n < N; ++n) s += Phi[n][i] * Phi[n][j];
            A[i][j] = A[j][i] = s;
        }
    for (int i = 0; i < km1; ++i) A[i][i] += lambda;

    // B = ΦᵀY    (km1 × nOut)
    Mat B(km1, Vec(nOut, 0.0));
    for (int i = 0; i < km1; ++i)
        for (int o = 0; o < nOut; ++o)
            for (int n = 0; n < N; ++n)
                B[i][o] += Phi[n][i] * Y[n][o];

    // Cholesky of A, solve for each output column
    cholesky(A);
    Mat W(km1, Vec(nOut));
    for (int o = 0; o < nOut; ++o) {
        Vec col(km1);
        for (int i = 0; i < km1; ++i) col[i] = B[i][o];
        col = choleskySolve(A, col);
        for (int i = 0; i < km1; ++i) W[i][o] = col[i];
    }
    return W;
}

// ─── evaluate ─────────────────────────────────────────────────────────────────
std::pair<double,double> RBFTrainer::evaluate(const Mat& X, const Mat& Y) {
    if (X.empty()) return {0.0, -1.0};
    const int N = int(X.size());
    const int nOut = m_net->nOutputs();
    const TaskType task = m_net->taskType();

    double loss = 0;
    int correct = 0;

    for (int n = 0; n < N; ++n) {
        Vec pred = m_net->predict(X[n]);
        for (int o = 0; o < nOut; ++o) {
            double e = pred[o] - Y[n][o];
            loss += e * e;
        }
        if (task != TaskType::Regression) {
            if (nOut == 1) {
                int p = (pred[0] >= 0.5) ? 1 : 0;
                if (p == int(Y[n][0] + 0.5)) ++correct;
            } else {
                int pi = int(std::max_element(pred.begin(), pred.end()) - pred.begin());
                int yi = int(std::max_element(Y[n].begin(), Y[n].end()) - Y[n].begin());
                if (pi == yi) ++correct;
            }
        }
    }
    double mse  = loss / (N * nOut);
    double acc  = (task != TaskType::Regression) ? double(correct)/N : -1.0;
    return {mse, acc};
}

// ─── run ──────────────────────────────────────────────────────────────────────
void RBFTrainer::run() {
    try {
        if (!m_net || !m_ds)
            throw std::runtime_error("RBFTrainer: network or dataset not set.");

        const int N      = m_ds->sampleCount;
        const int nIn    = m_ds->inputCount;
        const int nOut   = m_ds->outputCount;
        const int k      = m_net->config().nCenters;
        const double vsplit = m_net->config().valSplit;

        // ── Build input/output matrices ─────────────────────────────────────
        std::vector<int> idx(N); std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), m_rng);

        const int nVal   = int(N * vsplit);
        const int nTrain = N - nVal;

        Mat Xtr(nTrain, Vec(nIn)),  Ytr(nTrain, Vec(nOut));
        Mat Xval(nVal,  Vec(nIn)),  Yval(nVal,  Vec(nOut));

        for (int i = 0; i < nTrain; ++i) {
            int s = idx[i];
            Xtr[i] = Vec(m_ds->inputs[s].begin(), m_ds->inputs[s].end());
            Ytr[i] = Vec(m_ds->outputs[s].begin(), m_ds->outputs[s].end());
        }
        for (int i = 0; i < nVal; ++i) {
            int s = idx[nTrain + i];
            Xval[i] = Vec(m_ds->inputs[s].begin(), m_ds->inputs[s].end());
            Yval[i] = Vec(m_ds->outputs[s].begin(), m_ds->outputs[s].end());
        }

        // ── Phase 1: K-means++ centre selection ─────────────────────────────
        emit progressMessage("Phase 1/3 — K-means++ centre selection…");
        Mat centres;
        kMeans(Xtr, k, centres);

        // ── Phase 2: Width computation ───────────────────────────────────────
        emit progressMessage("Phase 2/3 — Computing RBF widths…");
        Vec widths = computeWidths(centres);
        m_net->setCenters(centres, widths);

        // ── Phase 3: Build design matrix + ridge regression ──────────────────
        emit progressMessage("Phase 3/3 — Solving normal equations (ridge regression)…");
        Mat Phi = m_net->designMatrix(Xtr);
        Mat W   = ridgeRegression(Phi, Ytr, m_net->config().ridgeLambda);
        m_net->setOutputWeights(W);

        // ── Evaluate ─────────────────────────────────────────────────────────
        auto [trLoss, trAcc]  = evaluate(Xtr,  Ytr);
        auto [valLoss, valAcc] = evaluate(Xval, Yval);

        emit progressMessage(QString(
            "Done — Train MSE: %1  |  Val MSE: %2  |  k=%3 centres  |  "
            "Train samples: %4  |  Val samples: %5")
            .arg(trLoss, 0,'f',5).arg(valLoss, 0,'f',5)
            .arg(centres.size()).arg(nTrain).arg(nVal));

        emit trainingCompleted(trLoss, valLoss, trAcc, valAcc);

    } catch (const std::exception& ex) {
        emit trainingError(QString::fromStdString(ex.what()));
    }
}

} // namespace NeuralStudio
