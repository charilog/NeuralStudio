// Baselines.cpp — non-BNF reference models of Mode 6 (see Baselines.h).
#include "Baselines.h"
#include "core/nn/NeuralNetwork.h"
#include "core/optimizers/optimizer.h"
#include "core/optimizers/genetic.h"
#include "core/optimizers/lbfgs.h"
#include <QHash>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <list>
#include <map>
#include <set>
#include <numeric>
#include <random>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace NeuralStudio {

namespace {

using Vec = std::vector<double>;
using Mat = std::vector<Vec>;

double nowSec() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ── Data in the form every baseline uses ──────────────────────────────────────
struct BLData {
    int  n = 0, d = 0;
    bool cls = false;
    int  K = 0;                 // classes (cls); 2 = binary
    Mat  X;                     // min-max normalised inputs (train statistics)
    std::vector<int> yc;        // class index (cls)
    Vec  y;                     // raw target (reg)
    Vec  ys;                    // standardised target (reg)
};

struct Encoder {
    bool cls = false;
    std::vector<double> classValues;   // sorted, from the train file
    std::vector<double> mn, rg;        // input min / range
    double yMean = 0, yStd = 1;
    int    d = 0;

    void fit(const Dataset& tr) {
        Dataset tmp = tr;
        tmp.computeStatistics();
        d = tmp.inputCount;
        mn.assign(d, 0); rg.assign(d, 1);
        for (int i = 0; i < d && i < static_cast<int>(tmp.inputStats.size()); ++i) {
            mn[i] = tmp.inputStats[i].min;
            rg[i] = tmp.inputStats[i].max - tmp.inputStats[i].min;
            if (rg[i] < 1e-10) rg[i] = 1.0;
        }
        cls = detectTask(&tmp) != TaskType::Regression;
        if (cls) {
            classValues = tmp.uniqueOutputValues(0);
            std::sort(classValues.begin(), classValues.end());
        } else {
            double s = 0, s2 = 0;
            for (int r = 0; r < tmp.sampleCount; ++r) { const double v = tmp.outputAt(r, 0); s += v; s2 += v * v; }
            const double n = std::max(1, tmp.sampleCount);
            yMean = s / n;
            yStd = std::sqrt(std::max(0.0, s2 / n - yMean * yMean));
            if (yStd < 1e-12) yStd = 1.0;
        }
    }
    int classIndex(double v) const {
        int best = 0; double bd = std::numeric_limits<double>::infinity();
        for (int k = 0; k < static_cast<int>(classValues.size()); ++k) {
            const double dd = std::fabs(v - classValues[k]);
            if (dd < bd) { bd = dd; best = k; }
        }
        return best;
    }
    BLData encode(const Dataset& ds) const {
        BLData D;
        D.n = ds.sampleCount; D.d = d; D.cls = cls; D.K = cls ? std::max(2, static_cast<int>(classValues.size())) : 0;
        D.X.resize(D.n);
        for (int r = 0; r < D.n; ++r) {
            const auto row = ds.inputRow(r);
            Vec x(d);
            for (int i = 0; i < d; ++i) x[i] = (i < static_cast<int>(row.size()) ? row[i] - mn[i] : 0.0) / rg[i];
            D.X[r] = std::move(x);
            const double v = ds.outputAt(r, 0);
            if (cls) D.yc.push_back(classIndex(v));
            else { D.y.push_back(v); D.ys.push_back((v - yMean) / yStd); }
        }
        return D;
    }
};

// Error of predictions: classification error % or regression MSE (raw scale).
double errorOf(const BLData& D, const std::vector<int>& pc, const Vec& pr) {
    if (D.n == 0) return -1;
    double e = 0;
    if (D.cls) { for (int i = 0; i < D.n; ++i) e += (pc[i] != D.yc[i]) ? 1.0 : 0.0; return 100.0 * e / D.n; }
    for (int i = 0; i < D.n; ++i) { const double t = pr[i] - D.y[i]; e += t * t; }
    return e / D.n;
}

Vec toRaw(const Vec& ps, const Encoder& enc) {
    Vec r(ps.size());
    for (size_t i = 0; i < ps.size(); ++i) r[i] = ps[i] * enc.yStd + enc.yMean;
    return r;
}

inline double sigm(double z) {
    if (z >= 0) { const double e = std::exp(-z); return 1.0 / (1.0 + e); }
    const double e = std::exp(z); return e / (1.0 + e);
}

// ── MLP with one sigmoid hidden layer (the network of the BNF paper) ──────────
struct Mlp {
    int d = 0, H = 10, O = 1;
    bool cls = false; int K = 0;
    int nParams() const { return H * (d + 1) + O * (H + 1); }

    void init(const BLData& D, int hidden) {
        d = D.d; H = hidden; cls = D.cls; K = D.K;
        O = cls ? (K == 2 ? 1 : K) : 1;
    }
    // forward for one sample; h (H) and out (O) are filled
    void forward(const double* w, const Vec& x, Vec& h, Vec& out) const {
        h.resize(H); out.resize(O);
        for (int j = 0; j < H; ++j) {
            const double* wj = w + j * (d + 1);
            double z = wj[d];
            for (int i = 0; i < d; ++i) z += wj[i] * x[i];
            h[j] = sigm(z);
        }
        const double* wo = w + H * (d + 1);
        for (int k = 0; k < O; ++k) {
            const double* wk = wo + k * (H + 1);
            double z = wk[H];
            for (int j = 0; j < H; ++j) z += wk[j] * h[j];
            out[k] = z;
        }
    }
    // loss of one sample; if g != nullptr the gradient is accumulated into g
    double sampleLoss(const double* w, const BLData& D, int r, Vec& h, Vec& out, Vec& dout, double* g) const {
        forward(w, D.X[r], h, out);
        double L = 0;
        dout.assign(O, 0.0);
        if (!cls) {
            const double e = out[0] - D.ys[r];
            L = e * e; dout[0] = 2 * e;
        } else if (O == 1) {
            const double p = sigm(out[0]);
            const double t = D.yc[r] == 1 ? 1.0 : 0.0;
            const double eps = 1e-12;
            L = -(t * std::log(std::max(eps, p)) + (1 - t) * std::log(std::max(eps, 1 - p)));
            dout[0] = p - t;
        } else {
            double m = out[0]; for (int k = 1; k < O; ++k) m = std::max(m, out[k]);
            double s = 0; for (int k = 0; k < O; ++k) s += std::exp(out[k] - m);
            const double lse = m + std::log(s);
            L = lse - out[D.yc[r]];
            for (int k = 0; k < O; ++k) dout[k] = std::exp(out[k] - lse) - (k == D.yc[r] ? 1.0 : 0.0);
        }
        if (g) {
            double* go = g + H * (d + 1);
            const double* wo = w + H * (d + 1);
            for (int k = 0; k < O; ++k) {
                double* gk = go + k * (H + 1);
                for (int j = 0; j < H; ++j) gk[j] += dout[k] * h[j];
                gk[H] += dout[k];
            }
            for (int j = 0; j < H; ++j) {
                double dh = 0;
                for (int k = 0; k < O; ++k) dh += dout[k] * wo[k * (H + 1) + j];
                const double dz = dh * h[j] * (1 - h[j]);
                double* gj = g + j * (d + 1);
                const Vec& x = D.X[r];
                for (int i = 0; i < d; ++i) gj[i] += dz * x[i];
                gj[d] += dz;
            }
        }
        return L;
    }
    // mean loss (and gradient) over the given rows
    double loss(const Vec& w, const BLData& D, const std::vector<int>& rows, Vec* grad) const {
        const int n = static_cast<int>(rows.size());
        const int P = nParams();
        if (grad) grad->assign(P, 0.0);
        double total = 0;
#ifdef _OPENMP
        const int nth = n >= 512 ? std::max(1, omp_get_max_threads()) : 1;
#else
        const int nth = 1;
#endif
        std::vector<double> part(nth, 0.0);
        std::vector<Vec> gpart(grad ? nth : 0, Vec(grad ? P : 0, 0.0));
#ifdef _OPENMP
        #pragma omp parallel num_threads(nth)
#endif
        {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            Vec h, out, dout;
            double local = 0;
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int q = 0; q < n; ++q)
                local += sampleLoss(w.data(), D, rows[q], h, out, dout, grad ? gpart[tid].data() : nullptr);
            part[tid] = local;
        }
        for (int t = 0; t < nth; ++t) total += part[t];
        if (grad) {
            for (int t = 0; t < nth; ++t) for (int p = 0; p < P; ++p) (*grad)[p] += gpart[t][p];
            for (double& v : *grad) v /= std::max(1, n);
        }
        return total / std::max(1, n);
    }
    // predictions in the encoded space
    void predict(const Vec& w, const BLData& D, std::vector<int>& pc, Vec& pr, const Encoder& enc) const {
        pc.assign(D.n, 0); pr.assign(D.n, 0.0);
        Vec h, out;
        for (int r = 0; r < D.n; ++r) {
            forward(w.data(), D.X[r], h, out);
            if (!cls) pr[r] = out[0] * enc.yStd + enc.yMean;
            else if (O == 1) pc[r] = sigm(out[0]) >= 0.5 ? 1 : 0;
            else pc[r] = static_cast<int>(std::max_element(out.begin(), out.end()) - out.begin());
        }
    }
};

// Problem for the framework optimizers (GA, L-BFGS)
class MlpProblem : public optimsolution::Problem {
public:
    MlpProblem(const Mlp& m, const BLData& D, double bound, std::atomic<bool>& stop)
        : m_mlp(m), m_D(D), m_stop(stop) {
        const int P = m.nParams();
        m_lb.assign(P, -bound); m_ub.assign(P, bound);
        m_rows.resize(D.n); std::iota(m_rows.begin(), m_rows.end(), 0);
    }
    int dimension() const override { return m_mlp.nParams(); }
    const std::vector<double>& lb() const override { return m_lb; }
    const std::vector<double>& ub() const override { return m_ub; }
    double evaluate(const std::vector<double>& x) override {
        ++m_calls;
        if (m_stop.load()) return 1e300;
        return m_mlp.loss(x, m_D, m_rows, nullptr);
    }
    int calls() const override { return m_calls + 2 * m_grads; }   // a gradient is charged as 3 evaluations
    std::pair<double, std::vector<double>> evalAndGrad(const std::vector<double>& x) override {
        ++m_calls; ++m_grads;
        Vec g;
        const double f = m_mlp.loss(x, m_D, m_rows, &g);
        return {f, g};
    }
private:
    const Mlp& m_mlp;
    const BLData& m_D;
    std::atomic<bool>& m_stop;
    std::vector<double> m_lb, m_ub;
    std::vector<int> m_rows;
    int m_calls = 0, m_grads = 0;
};

Vec glorotInit(const Mlp& m, std::mt19937& rng) {
    Vec w(m.nParams());
    const double a1 = std::sqrt(6.0 / (m.d + m.H)), a2 = std::sqrt(6.0 / (m.H + m.O));
    std::uniform_real_distribution<double> u1(-a1, a1), u2(-a2, a2);
    const int split = m.H * (m.d + 1);
    for (int p = 0; p < static_cast<int>(w.size()); ++p) w[p] = p < split ? u1(rng) : u2(rng);
    return w;
}

// ── ADAM ──────────────────────────────────────────────────────────────────────
Vec trainAdam(const Mlp& m, const BLData& D, const BaselineConfig& c, std::mt19937& rng,
              std::atomic<bool>& stop, qint64& evals) {
    Vec w = glorotInit(m, rng);
    const int P = m.nParams();
    Vec mo(P, 0.0), ve(P, 0.0), g(P, 0.0);
    const double b1 = 0.9, b2 = 0.999, eps = 1e-8;
    std::vector<int> idx(D.n); std::iota(idx.begin(), idx.end(), 0);
    const int B = std::max(1, std::min(c.adamBatch, D.n));
    const int stepsPerEpoch = (D.n + B - 1) / B;
    const int epochs = std::max(c.adamMinEpochs, (c.adamMinUpdates + stepsPerEpoch - 1) / stepsPerEpoch);
    long long t = 0;
    Vec h, out, dout;
    qint64 samples = 0;
    for (int ep = 0; ep < epochs && !stop.load(); ++ep) {
        std::shuffle(idx.begin(), idx.end(), rng);
        for (int s = 0; s < D.n; s += B) {
            const int e = std::min(D.n, s + B);
            std::fill(g.begin(), g.end(), 0.0);
            for (int q = s; q < e; ++q) m.sampleLoss(w.data(), D, idx[q], h, out, dout, g.data());
            const double inv = 1.0 / (e - s);
            ++t;
            const double c1 = 1 - std::pow(b1, static_cast<double>(t)), c2 = 1 - std::pow(b2, static_cast<double>(t));
            for (int p = 0; p < P; ++p) {
                const double gp = g[p] * inv;
                mo[p] = b1 * mo[p] + (1 - b1) * gp;
                ve[p] = b2 * ve[p] + (1 - b2) * gp * gp;
                w[p] -= c.adamLr * (mo[p] / c1) / (std::sqrt(ve[p] / c2) + eps);
            }
            samples += e - s;
        }
    }
    evals = 3 * samples / std::max(1, D.n);   // full-pass equivalents of forward+backward
    return w;
}

// ── BFGS (limited memory) ─────────────────────────────────────────────────────
Vec trainBfgs(const Mlp& m, const BLData& D, const BaselineConfig& c, std::mt19937& rng,
              std::atomic<bool>& stop, qint64& evals) {
    MlpProblem prob(m, D, 1e3, stop);   // practically unconstrained
    optimsolution::LBFGS opt;
    opt.setMemory(20);
    opt.setup(&prob, std::numeric_limits<int>::max() / 4, 1, rng());
    opt.setInitialPoint(glorotInit(m, rng));
    opt.init();
    for (int it = 0; it < c.bfgsMaxIters && !stop.load(); ++it) {
        const double before = opt.bestF();
        opt.one_iteration();
        opt.advanceIter();
        if (std::fabs(before - opt.bestF()) < 1e-12 * std::max(1.0, std::fabs(before)) && it > 50) {
            // no progress: one more try, then stop (converged)
            opt.one_iteration(); opt.advanceIter();
            if (std::fabs(before - opt.bestF()) < 1e-12 * std::max(1.0, std::fabs(before))) break;
        }
    }
    opt.end();
    evals = prob.calls();
    return opt.bestX();
}

// ── GENETIC: the GA of the BNF paper (defined with the batch-3 code below) ──────
Vec trainGeneticPaper(const Mlp& m, const BLData& D, const BaselineConfig& c, std::mt19937& rng,
                      std::atomic<bool>& stop, qint64& evals);

// ── RBF network: k-means++ centres, global width, ridge output layer ─────────
struct Rbf {
    Mat C; double sigma = 1; Mat W; int O = 1; bool cls = false; int K = 0;
    int nParams(int d) const { return static_cast<int>(C.size()) * (d + 1) + O * (static_cast<int>(C.size()) + 1); }
    Vec features(const Vec& x) const {
        Vec f(C.size() + 1);
        for (size_t j = 0; j < C.size(); ++j) {
            double r2 = 0; for (size_t i = 0; i < x.size(); ++i) { const double t = x[i] - C[j][i]; r2 += t * t; }
            f[j] = std::exp(-r2 / (2 * sigma * sigma));
        }
        f[C.size()] = 1.0;
        return f;
    }
};

bool choleskySolve(Mat A, Mat& B) {   // solves A X = B in place (A SPD, n x n; B n x m)
    const int n = static_cast<int>(A.size());
    for (int j = 0; j < n; ++j) {
        double s = A[j][j];
        for (int k = 0; k < j; ++k) s -= A[j][k] * A[j][k];
        if (s <= 0) return false;
        A[j][j] = std::sqrt(s);
        for (int i = j + 1; i < n; ++i) {
            double t = A[i][j];
            for (int k = 0; k < j; ++k) t -= A[i][k] * A[j][k];
            A[i][j] = t / A[j][j];
        }
    }
    const int m = B.empty() ? 0 : static_cast<int>(B[0].size());
    for (int c = 0; c < m; ++c) {
        for (int i = 0; i < n; ++i) { double t = B[i][c]; for (int k = 0; k < i; ++k) t -= A[i][k] * B[k][c]; B[i][c] = t / A[i][i]; }
        for (int i = n - 1; i >= 0; --i) { double t = B[i][c]; for (int k = i + 1; k < n; ++k) t -= A[k][i] * B[k][c]; B[i][c] = t / A[i][i]; }
    }
    return true;
}

Rbf trainRbf(const BLData& D, const BaselineConfig& c, std::mt19937& rng, qint64& evals) {
    Rbf R; R.cls = D.cls; R.K = D.K; R.O = D.cls ? (D.K == 2 ? 1 : D.K) : 1;
    const int k = std::max(1, std::min(c.hidden, D.n));
    // k-means++
    std::uniform_int_distribution<int> pick(0, D.n - 1);
    R.C.push_back(D.X[pick(rng)]);
    Vec dist(D.n, std::numeric_limits<double>::infinity());
    auto d2 = [](const Vec& a, const Vec& b) { double s = 0; for (size_t i = 0; i < a.size(); ++i) { const double t = a[i] - b[i]; s += t * t; } return s; };
    while (static_cast<int>(R.C.size()) < k) {
        double tot = 0;
        for (int r = 0; r < D.n; ++r) { dist[r] = std::min(dist[r], d2(D.X[r], R.C.back())); tot += dist[r]; }
        if (tot <= 0) { R.C.push_back(D.X[pick(rng)]); continue; }
        std::uniform_real_distribution<double> u(0, tot);
        double target = u(rng), acc = 0; int chosen = D.n - 1;
        for (int r = 0; r < D.n; ++r) { acc += dist[r]; if (acc >= target) { chosen = r; break; } }
        R.C.push_back(D.X[chosen]);
    }
    std::vector<int> assign(D.n, -1);
    for (int it = 0; it < c.rbfKmeansIters; ++it) {
        bool changed = false;
        for (int r = 0; r < D.n; ++r) {
            int b = 0; double bd = d2(D.X[r], R.C[0]);
            for (int j = 1; j < k; ++j) { const double t = d2(D.X[r], R.C[j]); if (t < bd) { bd = t; b = j; } }
            if (assign[r] != b) { assign[r] = b; changed = true; }
        }
        Mat S(k, Vec(D.d, 0.0)); std::vector<int> cnt(k, 0);
        for (int r = 0; r < D.n; ++r) { ++cnt[assign[r]]; for (int i = 0; i < D.d; ++i) S[assign[r]][i] += D.X[r][i]; }
        for (int j = 0; j < k; ++j) if (cnt[j] > 0) for (int i = 0; i < D.d; ++i) R.C[j][i] = S[j][i] / cnt[j];
        if (!changed) break;
    }
    double dmax = 0;
    for (int a = 0; a < k; ++a) for (int b = a + 1; b < k; ++b) dmax = std::max(dmax, std::sqrt(d2(R.C[a], R.C[b])));
    R.sigma = dmax > 0 ? dmax / std::sqrt(2.0 * k) : 1.0;
    // ridge regression on the Gaussian features
    const int F = k + 1;
    Mat A(F, Vec(F, 0.0)), B(F, Vec(R.O, 0.0));
    for (int r = 0; r < D.n; ++r) {
        const Vec f = R.features(D.X[r]);
        Vec t(R.O, 0.0);
        if (!D.cls) t[0] = D.ys[r];
        else if (R.O == 1) t[0] = D.yc[r] == 1 ? 1.0 : 0.0;
        else t[D.yc[r]] = 1.0;
        for (int a = 0; a < F; ++a) { for (int b = 0; b < F; ++b) A[a][b] += f[a] * f[b]; for (int o = 0; o < R.O; ++o) B[a][o] += f[a] * t[o]; }
    }
    double tr = 0; for (int a = 0; a < F; ++a) tr += A[a][a];
    double lam = c.rbfRidge * std::max(1e-12, tr / F);
    for (int attempt = 0; attempt < 8; ++attempt) {
        Mat A2 = A, B2 = B;
        for (int a = 0; a < F; ++a) A2[a][a] += lam;
        if (choleskySolve(A2, B2)) { R.W = B2; break; }
        lam *= 100;
    }
    if (R.W.empty()) R.W.assign(F, Vec(R.O, 0.0));
    evals = 1;
    return R;
}

void predictRbf(const Rbf& R, const BLData& D, const Encoder& enc, std::vector<int>& pc, Vec& pr) {
    pc.assign(D.n, 0); pr.assign(D.n, 0.0);
    for (int r = 0; r < D.n; ++r) {
        const Vec f = R.features(D.X[r]);
        Vec o(R.O, 0.0);
        for (int q = 0; q < R.O; ++q) for (size_t a = 0; a < f.size(); ++a) o[q] += f[a] * R.W[a][q];
        if (!R.cls) pr[r] = o[0] * enc.yStd + enc.yMean;
        else if (R.O == 1) pc[r] = o[0] >= 0.5 ? 1 : 0;
        else pc[r] = static_cast<int>(std::max_element(o.begin(), o.end()) - o.begin());
    }
}


// ══ Batch 2: kNN, logistic / ridge, random forest, SVM ══════════════════════════

// Subset of a dataset (rows by index)
BLData subset(const BLData& D, const std::vector<int>& idx) {
    BLData S;
    S.n = static_cast<int>(idx.size()); S.d = D.d; S.cls = D.cls; S.K = D.K;
    S.X.reserve(S.n);
    for (int i : idx) {
        S.X.push_back(D.X[i]);
        if (D.cls) S.yc.push_back(D.yc[i]);
        else { S.y.push_back(D.y[i]); S.ys.push_back(D.ys[i]); }
    }
    return S;
}

// Deterministic 80 / 20 split of the train file for hyper-parameter selection.
// Classification: stratified, so every class appears on both sides when possible.
void holdout(const BLData& D, std::mt19937& rng, std::vector<int>& tr, std::vector<int>& va, double frac = 0.2) {
    tr.clear(); va.clear();
    if (D.cls) {
        std::vector<std::vector<int>> byC(D.K);
        for (int i = 0; i < D.n; ++i) byC[D.yc[i]].push_back(i);
        for (auto& v : byC) {
            std::shuffle(v.begin(), v.end(), rng);
            const int nv = v.size() >= 2 ? std::max(1, static_cast<int>(std::lround(frac * v.size()))) : 0;
            for (int q = 0; q < static_cast<int>(v.size()); ++q) (q < nv ? va : tr).push_back(v[q]);
        }
    } else {
        std::vector<int> v(D.n); std::iota(v.begin(), v.end(), 0);
        std::shuffle(v.begin(), v.end(), rng);
        const int nv = std::max(1, static_cast<int>(std::lround(frac * D.n)));
        for (int q = 0; q < D.n; ++q) (q < nv ? va : tr).push_back(v[q]);
    }
    if (tr.empty()) { tr = va; }
    if (va.empty()) { va = tr; }
}

// validation score (lower is better) of predictions on an encoded subset
double scoreOf(const BLData& V, const std::vector<int>& pc, const Vec& prStd) {
    double e = 0;
    if (V.cls) { for (int i = 0; i < V.n; ++i) e += pc[i] != V.yc[i]; return e / std::max(1, V.n); }
    for (int i = 0; i < V.n; ++i) { const double t = prStd[i] - V.ys[i]; e += t * t; }
    return e / std::max(1, V.n);
}

inline double sqDist(const Vec& a, const Vec& b) {
    double s = 0; for (size_t i = 0; i < a.size(); ++i) { const double t = a[i] - b[i]; s += t * t; } return s;
}

// ── kNN ───────────────────────────────────────────────────────────────────────
// Predicts for every query with every k in ks (ascending), in one pass.
// out[kIndex][query]: class index (cls) or standardised mean (reg).
void knnPredictAll(const BLData& T, const BLData& Q, const std::vector<int>& ks,
                   std::vector<std::vector<int>>& pc, std::vector<Vec>& pr) {
    const int kmax = std::min(ks.back(), T.n);
    pc.assign(ks.size(), std::vector<int>(Q.n, 0));
    pr.assign(ks.size(), Vec(Q.n, 0.0));
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 16)
#endif
    for (int q = 0; q < Q.n; ++q) {
        std::vector<std::pair<double, int>> dist(T.n);
        for (int t = 0; t < T.n; ++t) dist[t] = {sqDist(Q.X[q], T.X[t]), t};
        std::partial_sort(dist.begin(), dist.begin() + kmax, dist.end());
        for (size_t a = 0; a < ks.size(); ++a) {
            const int k = std::min(ks[a], T.n);
            if (T.cls) {
                std::vector<int> votes(T.K, 0); std::vector<double> dsum(T.K, 0.0);
                for (int r = 0; r < k; ++r) { ++votes[T.yc[dist[r].second]]; dsum[T.yc[dist[r].second]] += dist[r].first; }
                int best = 0;
                for (int c = 1; c < T.K; ++c)
                    if (votes[c] > votes[best] || (votes[c] == votes[best] && votes[c] > 0 && dsum[c] < dsum[best])) best = c;
                pc[a][q] = best;
            } else {
                double s = 0; for (int r = 0; r < k; ++r) s += T.ys[dist[r].second];
                pr[a][q] = s / k;
            }
        }
    }
}

int knnChooseK(const BLData& D, std::mt19937& rng) {
    std::vector<int> tr, va; holdout(D, rng, tr, va);
    const BLData T = subset(D, tr), V = subset(D, va);
    std::vector<int> ks;
    for (int k : {1, 3, 5, 7, 9, 11, 15, 21}) if (k <= T.n) ks.push_back(k);
    if (ks.empty()) return 1;
    std::vector<std::vector<int>> pc; std::vector<Vec> pr;
    knnPredictAll(T, V, ks, pc, pr);
    int best = ks[0]; double bs = std::numeric_limits<double>::infinity();
    for (size_t a = 0; a < ks.size(); ++a) {
        const double s = scoreOf(V, pc[a], pr[a]);
        if (s < bs - 1e-12) { bs = s; best = ks[a]; }
    }
    return best;
}

// ── Logistic regression (L2) and ridge regression ────────────────────────────
struct Linear {
    int d = 0, O = 1; bool cls = false; int K = 0;
    Vec w;                         // O x (d+1), bias last
    int nParams() const { return O * (d + 1); }
    void scores(const Vec& x, Vec& s) const {
        s.assign(O, 0.0);
        for (int o = 0; o < O; ++o) { const double* wo = w.data() + o * (d + 1); double z = wo[d]; for (int i = 0; i < d; ++i) z += wo[i] * x[i]; s[o] = z; }
    }
};

class LogisticProblem : public optimsolution::Problem {
public:
    LogisticProblem(const BLData& D, int O, double lambda) : m_D(D), m_O(O), m_lambda(lambda) {
        const int P = O * (D.d + 1);
        m_lb.assign(P, -1e4); m_ub.assign(P, 1e4);
    }
    int dimension() const override { return m_O * (m_D.d + 1); }
    const std::vector<double>& lb() const override { return m_lb; }
    const std::vector<double>& ub() const override { return m_ub; }
    double evaluate(const std::vector<double>& x) override { ++m_calls; return lossGrad(x, nullptr); }
    int calls() const override { return m_calls + 2 * m_grads; }
    std::pair<double, std::vector<double>> evalAndGrad(const std::vector<double>& x) override {
        ++m_calls; ++m_grads; Vec g; const double f = lossGrad(x, &g); return {f, g};
    }
private:
    double lossGrad(const Vec& w, Vec* g) const {
        const int d = m_D.d, P = m_O * (d + 1);
        if (g) g->assign(P, 0.0);
        double L = 0;
        Vec s(m_O);
        for (int r = 0; r < m_D.n; ++r) {
            const Vec& x = m_D.X[r];
            for (int o = 0; o < m_O; ++o) { const double* wo = w.data() + o * (d + 1); double z = wo[d]; for (int i = 0; i < d; ++i) z += wo[i] * x[i]; s[o] = z; }
            Vec dz(m_O);
            if (m_O == 1) {
                const double p = sigm(s[0]); const double t = m_D.yc[r] == 1 ? 1.0 : 0.0;
                L -= t * std::log(std::max(1e-12, p)) + (1 - t) * std::log(std::max(1e-12, 1 - p));
                dz[0] = p - t;
            } else {
                double m = s[0]; for (int o = 1; o < m_O; ++o) m = std::max(m, s[o]);
                double se = 0; for (int o = 0; o < m_O; ++o) se += std::exp(s[o] - m);
                const double lse = m + std::log(se);
                L += lse - s[m_D.yc[r]];
                for (int o = 0; o < m_O; ++o) dz[o] = std::exp(s[o] - lse) - (o == m_D.yc[r] ? 1.0 : 0.0);
            }
            if (g) for (int o = 0; o < m_O; ++o) { double* go = g->data() + o * (d + 1); for (int i = 0; i < d; ++i) go[i] += dz[o] * x[i]; go[d] += dz[o]; }
        }
        const double inv = 1.0 / std::max(1, m_D.n);
        L *= inv;
        if (g) for (double& v : *g) v *= inv;
        for (int o = 0; o < m_O; ++o) for (int i = 0; i < d; ++i) {   // L2 on weights, not on the bias
            const double wi = w[o * (d + 1) + i];
            L += 0.5 * m_lambda * wi * wi;
            if (g) (*g)[o * (d + 1) + i] += m_lambda * wi;
        }
        return L;
    }
    const BLData& m_D; int m_O; double m_lambda;
    std::vector<double> m_lb, m_ub;
    int m_calls = 0, m_grads = 0;
};

Linear fitLinear(const BLData& D, double lambda, std::mt19937& rng, qint64& evals) {
    Linear M; M.d = D.d; M.cls = D.cls; M.K = D.K; M.O = D.cls ? (D.K == 2 ? 1 : D.K) : 1;
    if (!D.cls) {   // ridge, closed form on standardised targets (no penalty on the bias)
        const int F = D.d + 1;
        Mat A(F, Vec(F, 0.0)), B(F, Vec(1, 0.0));
        Vec f(F);
        for (int r = 0; r < D.n; ++r) {
            for (int i = 0; i < D.d; ++i) f[i] = D.X[r][i];
            f[D.d] = 1.0;
            for (int a = 0; a < F; ++a) { for (int b = 0; b < F; ++b) A[a][b] += f[a] * f[b]; B[a][0] += f[a] * D.ys[r]; }
        }
        double lam = lambda * std::max(1, D.n);
        for (int attempt = 0; attempt < 8; ++attempt) {
            Mat A2 = A, B2 = B;
            for (int a = 0; a < D.d; ++a) A2[a][a] += lam;
            A2[D.d][D.d] += 1e-10 * std::max(1, D.n);
            if (choleskySolve(A2, B2)) { M.w.resize(F); for (int a = 0; a < F; ++a) M.w[a] = B2[a][0]; break; }
            lam = std::max(lam * 100, 1e-8);
        }
        if (M.w.empty()) M.w.assign(F, 0.0);
        evals = 1;
        return M;
    }
    LogisticProblem prob(D, M.O, lambda);
    optimsolution::LBFGS opt;
    opt.setMemory(20);
    opt.setup(&prob, std::numeric_limits<int>::max() / 4, 1, rng());
    opt.setInitialPoint(Vec(M.nParams(), 0.0));
    opt.init();
    for (int it = 0; it < 2000; ++it) {
        const double before = opt.bestF();
        opt.one_iteration(); opt.advanceIter();
        if (it > 10 && std::fabs(before - opt.bestF()) < 1e-10 * std::max(1.0, std::fabs(before))) break;
    }
    opt.end();
    M.w = opt.bestX();
    evals = prob.calls();
    return M;
}

void predictLinear(const Linear& M, const BLData& D, std::vector<int>& pc, Vec& prStd) {
    pc.assign(D.n, 0); prStd.assign(D.n, 0.0);
    Vec s;
    for (int r = 0; r < D.n; ++r) {
        M.scores(D.X[r], s);
        if (!M.cls) prStd[r] = s[0];
        else if (M.O == 1) pc[r] = s[0] >= 0.0 ? 1 : 0;
        else pc[r] = static_cast<int>(std::max_element(s.begin(), s.end()) - s.begin());
    }
}

// ── Random forest (CART, bootstrap, random feature subsets) ───────────────────
struct TreeNode { int feat = -1; double thr = 0; int left = -1, right = -1; int leaf = -1; };
struct Tree { std::vector<TreeNode> nodes; std::vector<Vec> leafDist; Vec leafVal; };

struct ForestCfg { int trees = 100; int mtry = 1; int minLeaf = 1; };

Tree growTree(const BLData& D, std::vector<int> rows, const ForestCfg& fc, std::mt19937& rng) {
    Tree T;
    struct Item { std::vector<int> rows; int node; };
    std::vector<Item> stack;
    T.nodes.push_back(TreeNode());
    stack.push_back({std::move(rows), 0});
    std::vector<int> feats(D.d); std::iota(feats.begin(), feats.end(), 0);
    auto makeLeaf = [&](int node, const std::vector<int>& rr) {
        if (D.cls) {
            Vec p(D.K, 0.0);
            for (int r : rr) p[D.yc[r]] += 1.0;
            for (double& v : p) v /= std::max<size_t>(1, rr.size());
            T.nodes[node].leaf = static_cast<int>(T.leafDist.size());
            T.leafDist.push_back(p);
        } else {
            double s = 0; for (int r : rr) s += D.y[r];
            T.nodes[node].leaf = static_cast<int>(T.leafVal.size());
            T.leafVal.push_back(s / std::max<size_t>(1, rr.size()));
        }
    };
    std::vector<std::pair<double, int>> vals;
    while (!stack.empty()) {
        Item it = std::move(stack.back()); stack.pop_back();
        const int n = static_cast<int>(it.rows.size());
        bool pure = true;
        if (D.cls) { for (int r : it.rows) if (D.yc[r] != D.yc[it.rows[0]]) { pure = false; break; } }
        else { for (int r : it.rows) if (D.y[r] != D.y[it.rows[0]]) { pure = false; break; } }
        if (pure || n < 2 * fc.minLeaf) { makeLeaf(it.node, it.rows); continue; }

        // parent statistics
        Vec totC(D.cls ? D.K : 0, 0.0); double totS = 0;
        for (int r : it.rows) { if (D.cls) totC[D.yc[r]] += 1; else totS += D.y[r]; }

        double bestGain = -std::numeric_limits<double>::infinity(); int bestF = -1; double bestT = 0;
        double parentScore;
        if (D.cls) { parentScore = 0; for (double c : totC) parentScore += c * c; parentScore /= n; }
        else parentScore = totS * totS / n;

        std::shuffle(feats.begin(), feats.end(), rng);
        for (int fi = 0; fi < fc.mtry && fi < D.d; ++fi) {
            const int f = feats[fi];
            vals.resize(n);
            for (int q = 0; q < n; ++q) vals[q] = {D.X[it.rows[q]][f], it.rows[q]};
            std::sort(vals.begin(), vals.end());
            if (vals.front().first == vals.back().first) continue;
            Vec lc(D.cls ? D.K : 0, 0.0); double ls = 0;
            double lsq = 0;   // sum of squared class counts on the left (cls)
            Vec rc = totC; double rsq = 0; for (double c : rc) rsq += c * c;
            for (int q = 0; q < n - 1; ++q) {
                const int r = vals[q].second;
                if (D.cls) {
                    const int c = D.yc[r];
                    lsq += 2 * lc[c] + 1; lc[c] += 1;
                    rsq -= 2 * rc[c] - 1; rc[c] -= 1;
                } else ls += D.y[r];
                const int nl = q + 1, nr = n - nl;
                if (nl < fc.minLeaf || nr < fc.minLeaf) continue;
                if (vals[q].first == vals[q + 1].first) continue;
                double score;
                if (D.cls) score = lsq / nl + rsq / nr;            // maximising this minimises weighted Gini
                else { const double rs = totS - ls; score = ls * ls / nl + rs * rs / nr; }   // minimises SSE
                const double gain = score - parentScore;
                if (gain > bestGain + 1e-12) { bestGain = gain; bestF = f; bestT = 0.5 * (vals[q].first + vals[q + 1].first); }
            }
        }
        if (bestF < 0 || bestGain <= 1e-12) { makeLeaf(it.node, it.rows); continue; }
        std::vector<int> L, R;
        for (int r : it.rows) (D.X[r][bestF] <= bestT ? L : R).push_back(r);
        if (L.empty() || R.empty()) { makeLeaf(it.node, it.rows); continue; }
        const int li = static_cast<int>(T.nodes.size()); T.nodes.push_back(TreeNode());
        const int ri = static_cast<int>(T.nodes.size()); T.nodes.push_back(TreeNode());
        T.nodes[it.node].feat = bestF; T.nodes[it.node].thr = bestT;
        T.nodes[it.node].left = li; T.nodes[it.node].right = ri;
        stack.push_back({std::move(L), li});
        stack.push_back({std::move(R), ri});
    }
    return T;
}

int leafOf(const Tree& T, const Vec& x) {
    int k = 0;
    while (T.nodes[k].leaf < 0) k = x[T.nodes[k].feat] <= T.nodes[k].thr ? T.nodes[k].left : T.nodes[k].right;
    return T.nodes[k].leaf;
}

std::vector<Tree> fitForest(const BLData& D, const ForestCfg& fc, quint32 seed) {
    std::vector<Tree> F(fc.trees);
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int t = 0; t < fc.trees; ++t) {
        std::mt19937 rng(seed * 2654435761u + static_cast<quint32>(t) * 40503u + 17u);
        std::uniform_int_distribution<int> pick(0, D.n - 1);
        std::vector<int> boot(D.n);
        for (int q = 0; q < D.n; ++q) boot[q] = pick(rng);
        F[t] = growTree(D, std::move(boot), fc, rng);
    }
    return F;
}

void predictForest(const std::vector<Tree>& F, const BLData& D, std::vector<int>& pc, Vec& pr) {
    pc.assign(D.n, 0); pr.assign(D.n, 0.0);
    for (int r = 0; r < D.n; ++r) {
        if (D.cls) {
            Vec p(D.K, 0.0);
            for (const Tree& T : F) { const Vec& l = T.leafDist[leafOf(T, D.X[r])]; for (int c = 0; c < D.K; ++c) p[c] += l[c]; }
            pc[r] = static_cast<int>(std::max_element(p.begin(), p.end()) - p.begin());
        } else {
            double s = 0; for (const Tree& T : F) s += T.leafVal[leafOf(T, D.X[r])];
            pr[r] = s / std::max<size_t>(1, F.size());
        }
    }
}

// ── SVM: SMO with second-order working-set selection (LIBSVM, Fan et al. 2005) ─
// Solves  min 1/2 a'Qa + p'a   s.t.  y'a = 0,  0 <= a <= C,   Q_ij = y_i y_j K(x_{i%l}, x_{j%l}).
class KernelRows {
public:
    KernelRows(const Mat& X, double gamma, size_t cacheBytes) : m_X(X), m_gamma(gamma) {
        m_l = static_cast<int>(X.size());
        m_sq.resize(m_l);
        for (int i = 0; i < m_l; ++i) { double s = 0; for (double v : X[i]) s += v * v; m_sq[i] = s; }
        m_maxRows = std::max<size_t>(2, cacheBytes / (sizeof(float) * std::max(1, m_l)));
        m_slot.assign(m_l, m_lru.end());
    }
    const std::vector<float>& row(int i) {
        if (m_slot[i] != m_lru.end()) { m_lru.splice(m_lru.begin(), m_lru, m_slot[i]); return m_lru.front().second; }
        std::vector<float> r;
        if (m_lru.size() >= m_maxRows) {   // recycle the least recently used row
            auto last = std::prev(m_lru.end());
            m_slot[last->first] = m_lru.end();
            r = std::move(last->second);
            m_lru.erase(last);
        }
        r.resize(m_l);
        const Vec& xi = m_X[i];
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if (m_l > 2000)
#endif
        for (int j = 0; j < m_l; ++j) {
            double dot = 0; const Vec& xj = m_X[j];
            for (size_t k = 0; k < xi.size(); ++k) dot += xi[k] * xj[k];
            r[j] = static_cast<float>(std::exp(-m_gamma * std::max(0.0, m_sq[i] + m_sq[j] - 2 * dot)));
        }
        m_lru.emplace_front(i, std::move(r));
        m_slot[i] = m_lru.begin();
        return m_lru.front().second;
    }
    int size() const { return m_l; }
private:
    const Mat& m_X; double m_gamma; int m_l = 0;
    Vec m_sq;
    size_t m_maxRows = 2;
    std::list<std::pair<int, std::vector<float>>> m_lru;
    std::vector<std::list<std::pair<int, std::vector<float>>>::iterator> m_slot;
};

struct SmoResult { Vec alpha; double rho = 0; qint64 kernelRows = 0; bool converged = true; };

SmoResult smoSolve(KernelRows& K, const std::vector<int>& y, const Vec& p, double C, double eps,
                   std::atomic<bool>& stop) {
    const int L = static_cast<int>(y.size());     // L = l (SVC) or 2l (SVR)
    const int l = K.size();
    SmoResult R; R.alpha.assign(L, 0.0);
    Vec G = p;                                    // gradient at alpha = 0
    Vec QD(L, 1.0);                               // K(x,x) = 1 for the RBF kernel
    auto isUp = [&](int t) { return R.alpha[t] >= C; };
    auto isLow = [&](int t) { return R.alpha[t] <= 0; };
    const double TAU = 1e-12;
    const qint64 maxIter = std::max<qint64>(10000000, 100LL * L);
    qint64 iter = 0;
    while (true) {
        if (stop.load()) { R.converged = false; break; }
        // working set selection (WSS2)
        double Gmax = -std::numeric_limits<double>::infinity(), Gmax2 = -std::numeric_limits<double>::infinity();
        int gi = -1;
        for (int t = 0; t < L; ++t) {
            if (y[t] == +1) { if (!isUp(t) && -G[t] >= Gmax) { Gmax = -G[t]; gi = t; } }
            else            { if (!isLow(t) && G[t] >= Gmax) { Gmax = G[t]; gi = t; } }
        }
        if (gi < 0) break;
        const std::vector<float>& Ki = K.row(gi % l); ++R.kernelRows;
        int gj = -1; double objMin = std::numeric_limits<double>::infinity();
        for (int t = 0; t < L; ++t) {
            const double Qit = y[gi] * y[t] * Ki[t % l];
            if (y[t] == +1) {
                if (!isLow(t)) {
                    const double gd = Gmax + G[t];
                    if (G[t] >= Gmax2) Gmax2 = G[t];
                    if (gd > 0) {
                        double qc = QD[gi] + QD[t] - 2.0 * y[gi] * Qit;
                        const double od = -(gd * gd) / (qc > 0 ? qc : TAU);
                        if (od <= objMin) { gj = t; objMin = od; }
                    }
                }
            } else {
                if (!isUp(t)) {
                    const double gd = Gmax - G[t];
                    if (-G[t] >= Gmax2) Gmax2 = -G[t];
                    if (gd > 0) {
                        double qc = QD[gi] + QD[t] + 2.0 * y[gi] * Qit;
                        const double od = -(gd * gd) / (qc > 0 ? qc : TAU);
                        if (od <= objMin) { gj = t; objMin = od; }
                    }
                }
            }
        }
        if (Gmax + Gmax2 < eps || gj < 0) break;
        if (++iter > maxIter) { R.converged = false; break; }

        const int i = gi, j = gj;
        // Ki may be invalidated by fetching Kj: copy the two values we need first
        const std::vector<float> KiCopy = Ki;
        const std::vector<float>& Kj = K.row(j % l); ++R.kernelRows;
        const double Qij = y[i] * y[j] * KiCopy[j % l];
        const double oldAi = R.alpha[i], oldAj = R.alpha[j];
        if (y[i] != y[j]) {
            double qc = QD[i] + QD[j] + 2 * Qij; if (qc <= 0) qc = TAU;
            const double delta = (-G[i] - G[j]) / qc;
            const double diff = R.alpha[i] - R.alpha[j];
            R.alpha[i] += delta; R.alpha[j] += delta;
            if (diff > 0) { if (R.alpha[j] < 0) { R.alpha[j] = 0; R.alpha[i] = diff; } }
            else          { if (R.alpha[i] < 0) { R.alpha[i] = 0; R.alpha[j] = -diff; } }
            if (diff > 0) { if (R.alpha[i] > C) { R.alpha[i] = C; R.alpha[j] = C - diff; } }
            else          { if (R.alpha[j] > C) { R.alpha[j] = C; R.alpha[i] = C + diff; } }
        } else {
            double qc = QD[i] + QD[j] - 2 * Qij; if (qc <= 0) qc = TAU;
            const double delta = (G[i] - G[j]) / qc;
            const double sum = R.alpha[i] + R.alpha[j];
            R.alpha[i] -= delta; R.alpha[j] += delta;
            if (sum > C) { if (R.alpha[i] > C) { R.alpha[i] = C; R.alpha[j] = sum - C; } }
            else         { if (R.alpha[j] < 0) { R.alpha[j] = 0; R.alpha[i] = sum; } }
            if (sum > C) { if (R.alpha[j] > C) { R.alpha[j] = C; R.alpha[i] = sum - C; } }
            else         { if (R.alpha[i] < 0) { R.alpha[i] = 0; R.alpha[j] = sum; } }
        }
        const double dAi = R.alpha[i] - oldAi, dAj = R.alpha[j] - oldAj;
        for (int t = 0; t < L; ++t)
            G[t] += y[i] * y[t] * KiCopy[t % l] * dAi + y[j] * y[t] * Kj[t % l] * dAj;
    }
    // rho
    int nFree = 0; double sumFree = 0;
    double ub = std::numeric_limits<double>::infinity(), lb = -std::numeric_limits<double>::infinity();
    for (int t = 0; t < L; ++t) {
        const double yG = y[t] * G[t];
        if (isUp(t)) { if (y[t] == -1) ub = std::min(ub, yG); else lb = std::max(lb, yG); }
        else if (isLow(t)) { if (y[t] == +1) ub = std::min(ub, yG); else lb = std::max(lb, yG); }
        else { ++nFree; sumFree += yG; }
    }
    R.rho = nFree > 0 ? sumFree / nFree : (std::isfinite(ub) && std::isfinite(lb) ? 0.5 * (ub + lb) : 0.0);
    return R;
}

struct SvmModel {
    // one binary machine (SVC pair or SVR): f(x) = sum coef_s K(sv_s, x) - rho
    struct Machine { Mat sv; Vec coef; double rho = 0; int ca = 0, cb = 1; };
    std::vector<Machine> m;
    bool cls = false; int K = 0; double gamma = 1;
    int nParams() const { int s = 0; for (const auto& q : m) s += static_cast<int>(q.sv.size()); return s; }
    double decision(const Machine& q, const Vec& x) const {
        double f = -q.rho;
        for (size_t s = 0; s < q.sv.size(); ++s) f += q.coef[s] * std::exp(-gamma * sqDist(q.sv[s], x));
        return f;
    }
};

double svmGamma(const BLData& D) {   // "scale": 1 / (d * Var(X))
    double s = 0, s2 = 0; const double N = static_cast<double>(D.n) * D.d;
    for (const Vec& x : D.X) for (double v : x) { s += v; s2 += v * v; }
    const double var = N > 0 ? std::max(1e-12, s2 / N - (s / N) * (s / N)) : 1.0;
    return 1.0 / (std::max(1, D.d) * var);
}

SvmModel fitSvm(const BLData& D, double C, double gamma, std::atomic<bool>& stop, qint64& rows) {
    SvmModel M; M.cls = D.cls; M.K = D.K; M.gamma = gamma;
    const size_t cache = 256ull << 20;
    if (!D.cls) {   // epsilon-SVR on standardised targets, epsilon = 0.1
        KernelRows Kr(D.X, gamma, cache);
        const int l = D.n;
        std::vector<int> y(2 * l); Vec p(2 * l);
        for (int i = 0; i < l; ++i) { y[i] = +1; y[i + l] = -1; p[i] = 0.1 - D.ys[i]; p[i + l] = 0.1 + D.ys[i]; }
        const SmoResult R = smoSolve(Kr, y, p, C, 1e-3, stop);
        rows += R.kernelRows;
        SvmModel::Machine q; q.rho = R.rho;
        for (int i = 0; i < l; ++i) { const double c = R.alpha[i] - R.alpha[i + l]; if (std::fabs(c) > 0) { q.sv.push_back(D.X[i]); q.coef.push_back(c); } }
        M.m.push_back(std::move(q));
        return M;
    }
    for (int a = 0; a < D.K; ++a)
        for (int b = a + 1; b < D.K; ++b) {   // one-vs-one
            if (stop.load()) return M;
            Mat X; std::vector<int> y;
            for (int r = 0; r < D.n; ++r)
                if (D.yc[r] == a || D.yc[r] == b) { X.push_back(D.X[r]); y.push_back(D.yc[r] == a ? +1 : -1); }
            SvmModel::Machine q; q.ca = a; q.cb = b;
            const bool hasA = std::count(y.begin(), y.end(), +1) > 0, hasB = std::count(y.begin(), y.end(), -1) > 0;
            if (!hasA || !hasB) { q.rho = hasA ? -1 : 1; M.m.push_back(std::move(q)); continue; }   // constant vote
            KernelRows Kr(X, gamma, cache);
            const SmoResult R = smoSolve(Kr, y, Vec(y.size(), -1.0), C, 1e-3, stop);
            rows += R.kernelRows;
            q.rho = R.rho;
            for (size_t i = 0; i < y.size(); ++i) if (R.alpha[i] > 0) { q.sv.push_back(X[i]); q.coef.push_back(y[i] * R.alpha[i]); }
            M.m.push_back(std::move(q));
        }
    return M;
}

void predictSvm(const SvmModel& M, const BLData& D, std::vector<int>& pc, Vec& prStd) {
    pc.assign(D.n, 0); prStd.assign(D.n, 0.0);
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 16)
#endif
    for (int r = 0; r < D.n; ++r) {
        if (!M.cls) { prStd[r] = M.decision(M.m[0], D.X[r]); continue; }
        std::vector<int> votes(M.K, 0);
        for (const auto& q : M.m) ++votes[M.decision(q, D.X[r]) > 0 ? q.ca : q.cb];
        pc[r] = static_cast<int>(std::max_element(votes.begin(), votes.end()) - votes.begin());
    }
}


// ══ Batch 3: the GA of the BNF paper, GNN (the paper's method) and NEAT ═════════

// ── Real-coded GA as described in the BNF paper (phases 1 and 3, and GENETIC) ──
// Each generation the best (1 - ps) * Nc chromosomes are copied intact; the others are
// replaced by children: parents by tournament, z~ = a z + (1-a) w, w~ = a w + (1-a) z with
// a_i ~ U[-0.5, 1.5] per gene; every gene of a child is re-drawn uniformly in its interval
// with probability pm.
struct PaperGA {
    int    Nc = 500, Ng = 200, tournament = 4;
    double ps = 0.1, pm = 0.05;
};

Vec runPaperGA(const std::function<double(const Vec&)>& f, const Vec& L, const Vec& R, const PaperGA& g,
               std::mt19937& rng, std::atomic<bool>& stop, qint64& evals, const Vec* seedPoint = nullptr) {
    const int P = static_cast<int>(L.size());
    std::vector<Vec> pop(g.Nc, Vec(P));
    Vec fit(g.Nc);
    std::uniform_real_distribution<double> u01(0, 1), ua(-0.5, 1.5);
    auto draw = [&](int i) { return L[i] + (R[i] - L[i]) * u01(rng); };
    for (auto& c : pop) for (int i = 0; i < P; ++i) c[i] = draw(i);
    if (seedPoint && static_cast<int>(seedPoint->size()) == P) pop[0] = *seedPoint;
    for (int k = 0; k < g.Nc; ++k) { fit[k] = f(pop[k]); ++evals; }
    const int keep = std::max(1, std::min(g.Nc - 1, static_cast<int>(std::lround((1.0 - g.ps) * g.Nc))));
    std::vector<int> ord(g.Nc);
    std::uniform_int_distribution<int> pick(0, g.Nc - 1);
    auto tour = [&]() { int b = pick(rng); for (int t = 1; t < g.tournament; ++t) { const int o = pick(rng); if (fit[o] < fit[b]) b = o; } return b; };
    for (int gen = 0; gen < g.Ng && !stop.load(); ++gen) {
        std::iota(ord.begin(), ord.end(), 0);
        std::sort(ord.begin(), ord.end(), [&](int a, int b) { return fit[a] < fit[b]; });
        std::vector<Vec> np; Vec nf;
        np.reserve(g.Nc); nf.reserve(g.Nc);
        for (int q = 0; q < keep; ++q) { np.push_back(pop[ord[q]]); nf.push_back(fit[ord[q]]); }
        while (static_cast<int>(np.size()) < g.Nc) {
            const Vec& z = pop[tour()]; const Vec& w = pop[tour()];
            Vec c1(P), c2(P);
            for (int i = 0; i < P; ++i) {
                const double a = ua(rng);
                c1[i] = std::min(R[i], std::max(L[i], a * z[i] + (1 - a) * w[i]));
                c2[i] = std::min(R[i], std::max(L[i], a * w[i] + (1 - a) * z[i]));
                if (u01(rng) <= g.pm) c1[i] = draw(i);
                if (u01(rng) <= g.pm) c2[i] = draw(i);
            }
            for (Vec* c : {&c1, &c2}) {
                if (static_cast<int>(np.size()) >= g.Nc) break;
                nf.push_back(f(*c)); ++evals;
                np.push_back(std::move(*c));
            }
        }
        pop.swap(np); fit.swap(nf);
    }
    const int b = static_cast<int>(std::min_element(fit.begin(), fit.end()) - fit.begin());
    return pop[b];
}

// ── GNN: the three phases of the BNF paper on the H-node sigmoid MLP ──────────
struct GnnCfg {
    PaperGA ga;                 // Nc 500, Ng 200, ps 0.1, pm 0.05 (paper Table 4)
    int    Ns = 50;             // random samples per program
    int    Nl = 20;             // chromosomes used in local search (also Ncr)
    double pl = 0.001;          // local search rate
    double F = 2.0;             // phase 1 margin: [-F|w*|, F|w*|]  (not given in the paper)
    int    genomeLen = 200;     // codons per chromosome           (not given in the paper)
    double initBound = 10.0;    // phase 1 search interval          (not given in the paper)
};

// One program: list of (parameter, left command factor, right command factor)
struct BoundCmd { int idx; double lf, rf; };

// Grammar of the paper (Figure 2):  <expr> ::= <expr>,<expr> | (<xlist>,<command>,<command>)
// (the worked example in the paper maps codon%2 == 0 to <expr>,<expr>); <xlist> ::= x1 | ... | xn;
// <command> ::= NOTHING | SHRINK | EXPAND | SHRINK_EXPAND | EXPAND_SHRINK | SHRINK_SHRINK | EXPAND_EXPAND.
std::vector<BoundCmd> mapProgram(const std::vector<int>& g, int n, bool& ok) {
    static const double fac[7] = {1.0, 0.5, 1.5, 0.75, 0.75, 0.25, 2.25};
    std::vector<BoundCmd> out;
    size_t pos = 0; int wraps = 0; ok = true;
    auto next = [&]() -> int {
        if (pos >= g.size()) { pos = 0; if (++wraps > 2) { ok = false; return 0; } }
        return g[pos++];
    };
    // iterative expansion with an explicit stack of pending <expr>
    int pending = 1, guard = 0;
    while (pending > 0 && ok) {
        if (++guard > 4 * n + 64) { ok = false; break; }
        --pending;
        const int c = next(); if (!ok) break;
        if (c % 2 == 0 && static_cast<int>(out.size()) + pending < 2 * n) { pending += 2; continue; }
        const int idx = next() % n; if (!ok) break;
        const int lc = next() % 7; if (!ok) break;
        const int rc = next() % 7; if (!ok) break;
        out.push_back({idx, fac[lc], fac[rc]});
    }
    return out;
}

void applyProgram(const std::vector<BoundCmd>& p, const Vec& L0, const Vec& R0, Vec& L, Vec& R) {
    L = L0; R = R0;
    for (const auto& c : p) { L[c.idx] *= c.lf; R[c.idx] *= c.rf; }
}

Vec trainGnn(const Mlp& m, const BLData& D, const GnnCfg& c, std::mt19937& rng,
             std::atomic<bool>& stop, qint64& evals, QString& detail) {
    const int P = m.nParams();
    std::vector<int> rows(D.n); std::iota(rows.begin(), rows.end(), 0);
    auto loss = [&](const Vec& w) { return m.loss(w, D, rows, nullptr); };
    // Phase 1: GA training, then the initial intervals
    const Vec L1(P, -c.initBound), R1(P, c.initBound);
    const Vec w1 = runPaperGA(loss, L1, R1, c.ga, rng, stop, evals);
    Vec L0(P), R0(P);
    for (int i = 0; i < P; ++i) { L0[i] = -c.F * std::fabs(w1[i]); R0[i] = c.F * std::fabs(w1[i]); }
    if (stop.load()) return w1;

    // Phase 2: GE over bound programs; fitness = interval [min, max] of the error of Ns random networks
    struct Ind { std::vector<int> g; double lo = 0, hi = 0; bool ok = false; };
    std::uniform_int_distribution<int> codon(0, 255);
    std::uniform_real_distribution<double> u01(0, 1);
    auto fitness = [&](Ind& z) {
        bool ok = true;
        const auto prog = mapProgram(z.g, P, ok);
        z.ok = ok;
        if (!ok) { z.lo = z.hi = std::numeric_limits<double>::infinity(); return; }
        Vec L, R; applyProgram(prog, L0, R0, L, R);
        z.lo = std::numeric_limits<double>::infinity(); z.hi = -z.lo;
        Vec w(P);
        for (int s = 0; s < c.Ns && !stop.load(); ++s) {
            for (int i = 0; i < P; ++i) w[i] = L[i] + (R[i] - L[i]) * u01(rng);
            const double e = loss(w); ++evals;
            z.lo = std::min(z.lo, e); z.hi = std::max(z.hi, e);
        }
    };
    auto better = [](const Ind& a, const Ind& b) { return a.lo < b.lo || (a.lo == b.lo && a.hi < b.hi); };
    const int Nc = c.ga.Nc;
    std::vector<Ind> pop(Nc);
    for (auto& z : pop) { z.g.resize(c.genomeLen); for (int& v : z.g) v = codon(rng); fitness(z); }
    const int keep = std::max(1, std::min(Nc - 1, static_cast<int>(std::lround((1.0 - c.ga.ps) * Nc))));
    std::uniform_int_distribution<int> pick(0, Nc - 1);
    auto tour = [&]() { int b = pick(rng); for (int t = 1; t < c.ga.tournament; ++t) { const int o = pick(rng); if (better(pop[o], pop[b])) b = o; } return b; };
    auto onePoint = [&](const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c1, std::vector<int>& c2) {
        const int cut = 1 + static_cast<int>(rng() % std::max<size_t>(1, a.size() - 1));
        c1 = a; c2 = b;
        for (size_t i = cut; i < a.size(); ++i) { c1[i] = b[i]; c2[i] = a[i]; }
    };
    for (int gen = 0; gen < c.ga.Ng && !stop.load(); ++gen) {
        std::sort(pop.begin(), pop.end(), better);
        std::vector<Ind> np(pop.begin(), pop.begin() + keep);
        while (static_cast<int>(np.size()) < Nc && !stop.load()) {
            Ind a, b;
            onePoint(pop[tour()].g, pop[tour()].g, a.g, b.g);
            for (Ind* z : {&a, &b}) {
                for (int& v : z->g) if (u01(rng) <= c.ga.pm) v = codon(rng);
                if (static_cast<int>(np.size()) < Nc) { fitness(*z); np.push_back(std::move(*z)); }
            }
        }
        pop.swap(np);
        // local search (Algorithm 2): one-point crossover with Nl random partners, keep improvements
        for (int k = 0; k < Nc && !stop.load(); ++k) {
            if (u01(rng) > c.pl) continue;
            for (int t = 0; t < c.Nl && !stop.load(); ++t) {
                Ind g1, g2;
                onePoint(pop[k].g, pop[pick(rng)].g, g1.g, g2.g);
                fitness(g1); fitness(g2);
                if (!better(pop[k], g1)) pop[k] = g1;
                else if (!better(pop[k], g2)) pop[k] = g2;
            }
        }
    }
    std::sort(pop.begin(), pop.end(), better);
    bool ok = true;
    const auto best = mapProgram(pop[0].g, P, ok);
    Vec Ls, Rs; applyProgram(best, L0, R0, Ls, Rs);
    if (stop.load()) return w1;

    // Phase 3: GA inside the final intervals, then BFGS on the best network (kept inside them)
    Vec w3 = runPaperGA(loss, Ls, Rs, c.ga, rng, stop, evals);
    if (stop.load()) return w3;
    class BoxProblem : public optimsolution::Problem {
    public:
        BoxProblem(const Mlp& m, const BLData& D, const Vec& L, const Vec& R, std::vector<int>& rows)
            : m_m(m), m_D(D), m_L(L), m_R(R), m_rows(rows) {}
        int dimension() const override { return static_cast<int>(m_L.size()); }
        const std::vector<double>& lb() const override { return m_L; }
        const std::vector<double>& ub() const override { return m_R; }
        double evaluate(const std::vector<double>& x) override { ++m_calls; return m_m.loss(x, m_D, m_rows, nullptr); }
        int calls() const override { return m_calls + 2 * m_grads; }
        std::pair<double, std::vector<double>> evalAndGrad(const std::vector<double>& x) override {
            ++m_calls; ++m_grads; Vec g; const double f = m_m.loss(x, m_D, m_rows, &g); return {f, g};
        }
    private:
        const Mlp& m_m; const BLData& m_D; const Vec& m_L; const Vec& m_R; std::vector<int>& m_rows;
        int m_calls = 0, m_grads = 0;
    };
    BoxProblem bp(m, D, Ls, Rs, rows);
    optimsolution::LBFGS opt;
    opt.setMemory(20);
    opt.setup(&bp, std::numeric_limits<int>::max() / 4, 1, rng());
    opt.setInitialPoint(w3);
    opt.init();
    for (int it = 0; it < 2000 && !stop.load(); ++it) {
        const double before = opt.bestF();
        opt.one_iteration(); opt.advanceIter();
        if (it > 50 && std::fabs(before - opt.bestF()) < 1e-12 * std::max(1.0, std::fabs(before))) break;
    }
    opt.end();
    evals += bp.calls();
    int fixedZero = 0;
    for (int i = 0; i < P; ++i) if (Ls[i] == 0 && Rs[i] == 0) ++fixedZero;
    detail = QString("%1 bound commands, %2 parameters fixed to 0").arg(best.size()).arg(fixedZero);
    const Vec wb = opt.bestX();
    return loss(wb) <= loss(w3) ? wb : w3;
}

// ── NEAT (Stanley & Miikkulainen 2002), feed-forward ──────────────────────────
struct NeatCfg {
    int    pop = 150, gens = 200;
    double c1 = 1.0, c2 = 1.0, c3 = 0.4, deltaT = 3.0;
    double pWeightMut = 0.8, pPerturb = 0.9, perturbSd = 0.5, weightRange = 4.0;
    double pAddConn = 0.05, pAddNode = 0.03, pCrossover = 0.75, pInterspecies = 0.001;
    double pToggle = 0.01;
    double survival = 0.2;
    int    stagnation = 15;
};

struct NeatConn { int in, out; double w; bool en; int innov; };
struct NeatGenome {
    std::vector<NeatConn> conns;   // sorted by innov
    int nNodes = 0;                // ids: 0..nIn-1 inputs, nIn bias, nIn+1..nIn+O outputs, then hidden
    double fitness = 0, adj = 0, loss = std::numeric_limits<double>::infinity();
};

// A genome compiled for fast evaluation: nodes in topological order, incoming edges as flat arrays.
struct NeatNet {
    int nIn = 0, nOut = 0;
    std::vector<int> order;             // dense node indices to compute, topological order
    std::vector<int> start, src;        // incoming edges of order[k]: src[start[k] .. start[k+1])
    std::vector<double> w;
    std::vector<char> isOut;
    std::vector<int> outIdx;            // dense index of every output
    int nDense = 0;
};

class Neat {
public:
    Neat(const BLData& D, const NeatCfg& c, std::mt19937& rng) : m_D(D), m_c(c), m_rng(rng) {
        m_in = D.d; m_out = D.cls ? (D.K == 2 ? 1 : D.K) : 1;
        m_bias = m_in; m_firstOut = m_in + 1;
        m_rows.resize(D.n); std::iota(m_rows.begin(), m_rows.end(), 0);
        m_nextNode = m_firstOut + m_out;
    }
    NeatGenome run(std::atomic<bool>& stop, qint64& evals, int& generationsRun) {
        std::uniform_real_distribution<double> uw(-1, 1);
        std::vector<NeatGenome> pop(m_c.pop);
        for (auto& g : pop) {
            g.nNodes = m_firstOut + m_out;
            for (int o = 0; o < m_out; ++o)
                for (int i = 0; i <= m_in; ++i) g.conns.push_back({i, m_firstOut + o, uw(m_rng), true, innovation(i, m_firstOut + o)});
            sortConns(g);
        }
        NeatGenome best; best.loss = std::numeric_limits<double>::infinity();
        struct Species { NeatGenome rep; std::vector<int> members; double bestFit = -1; int stale = 0; int id; };
        std::vector<Species> species;
        int nextId = 0;
        generationsRun = 0;
        for (int gen = 0; gen < m_c.gens && !stop.load(); ++gen) {
            for (auto& g : pop) { g.loss = lossOf(g); ++evals; g.fitness = 1.0 / (1.0 + g.loss); if (g.loss < best.loss) best = g; }
            ++generationsRun;
            // speciation
            for (auto& s : species) s.members.clear();
            for (int k = 0; k < static_cast<int>(pop.size()); ++k) {
                bool placed = false;
                for (auto& s : species) if (distance(pop[k], s.rep) < m_c.deltaT) { s.members.push_back(k); placed = true; break; }
                if (!placed) { Species s; s.rep = pop[k]; s.members = {k}; s.id = nextId++; species.push_back(s); }
            }
            species.erase(std::remove_if(species.begin(), species.end(), [](const Species& s) { return s.members.empty(); }), species.end());
            // stagnation and fitness sharing
            for (auto& s : species) {
                double bf = 0; for (int k : s.members) bf = std::max(bf, pop[k].fitness);
                if (bf > s.bestFit + 1e-12) { s.bestFit = bf; s.stale = 0; } else ++s.stale;
                for (int k : s.members) pop[k].adj = pop[k].fitness / s.members.size();
            }
            if (species.size() > 2) {
                std::vector<int> ord(species.size()); std::iota(ord.begin(), ord.end(), 0);
                std::sort(ord.begin(), ord.end(), [&](int a, int b) { return species[a].bestFit > species[b].bestFit; });
                std::vector<Species> kept;
                for (size_t q = 0; q < ord.size(); ++q)
                    if (q < 2 || species[ord[q]].stale < m_c.stagnation) kept.push_back(species[ord[q]]);
                species.swap(kept);
            }
            // offspring allocation
            double total = 0; std::vector<double> sAdj(species.size(), 0.0);
            for (size_t s = 0; s < species.size(); ++s) { for (int k : species[s].members) sAdj[s] += pop[k].adj; total += sAdj[s]; }
            std::vector<int> quota(species.size(), 0); int assigned = 0;
            for (size_t s = 0; s < species.size(); ++s) { quota[s] = total > 0 ? static_cast<int>(std::floor(m_c.pop * sAdj[s] / total)) : 0; assigned += quota[s]; }
            for (size_t s = 0; assigned < m_c.pop && !species.empty(); s = (s + 1) % species.size()) { ++quota[s]; ++assigned; }
            std::vector<NeatGenome> next;
            std::uniform_real_distribution<double> u01(0, 1);
            for (size_t s = 0; s < species.size(); ++s) {
                auto& mem = species[s].members;
                std::sort(mem.begin(), mem.end(), [&](int a, int b) { return pop[a].fitness > pop[b].fitness; });
                if (quota[s] <= 0) continue;
                int q = quota[s];
                if (mem.size() > 5) { next.push_back(pop[mem[0]]); --q; }   // species champion
                const int parents = std::max(1, static_cast<int>(std::ceil(m_c.survival * mem.size())));
                std::uniform_int_distribution<int> pp(0, parents - 1);
                for (; q > 0; --q) {
                    NeatGenome child;
                    if (u01(m_rng) < m_c.pCrossover) {
                        const NeatGenome* a = &pop[mem[pp(m_rng)]];
                        const NeatGenome* b;
                        if (u01(m_rng) < m_c.pInterspecies && species.size() > 1) {
                            const auto& other = species[m_rng() % species.size()].members;
                            b = &pop[other[m_rng() % other.size()]];
                        } else b = &pop[mem[pp(m_rng)]];
                        child = crossover(*a, *b);
                    } else {
                        child = pop[mem[pp(m_rng)]];
                    }
                    mutate(child);
                    next.push_back(std::move(child));
                }
                species[s].rep = pop[mem[m_rng() % mem.size()]];
            }
            while (static_cast<int>(next.size()) < m_c.pop) next.push_back(best);
            pop.swap(next);
            m_genInnov.clear();
        }
        return best;
    }
    NeatNet compile(const NeatGenome& g) const {
        NeatNet N; N.nIn = m_in; N.nOut = m_out;
        // dense ids: inputs, bias, outputs first, then hidden nodes that appear in enabled genes
        std::map<int, int> dense;
        for (int v = 0; v < m_firstOut + m_out; ++v) dense[v] = v;
        int next = m_firstOut + m_out;
        for (const auto& c : g.conns) if (c.en) for (int v : {c.in, c.out}) if (!dense.count(v)) dense[v] = next++;
        N.nDense = next;
        std::vector<std::vector<std::pair<int, double>>> inc(next);
        std::vector<int> indeg(next, 0);
        std::vector<std::vector<int>> outs(next);
        for (const auto& c : g.conns) if (c.en) {
            const int a = dense[c.in], b = dense[c.out];
            inc[b].push_back({a, c.w}); ++indeg[b]; outs[a].push_back(b);
        }
        std::vector<int> q, order;
        for (int v = 0; v < next; ++v) if (indeg[v] == 0) q.push_back(v);
        while (!q.empty()) {
            const int v = q.back(); q.pop_back(); order.push_back(v);
            for (int b : outs[v]) if (--indeg[b] == 0) q.push_back(b);
        }
        N.isOut.assign(next, 0);
        for (int o = 0; o < m_out; ++o) { N.isOut[m_firstOut + o] = 1; N.outIdx.push_back(m_firstOut + o); }
        N.start.push_back(0);
        for (int v : order) {
            if (v < m_firstOut) continue;          // inputs and bias are given
            N.order.push_back(v);
            for (const auto& e : inc[v]) { N.src.push_back(e.first); N.w.push_back(e.second); }
            N.start.push_back(static_cast<int>(N.src.size()));
        }
        return N;
    }
    // network output (pre-activation of the outputs) for one input row
    static void forward(const NeatNet& N, const Vec& x, Vec& val, Vec& out) {
        val.assign(N.nDense, 0.0);
        for (int i = 0; i < N.nIn; ++i) val[i] = x[i];
        val[N.nIn] = 1.0;
        for (size_t k = 0; k < N.order.size(); ++k) {
            double s = 0;
            for (int e = N.start[k]; e < N.start[k + 1]; ++e) s += N.w[e] * val[N.src[e]];
            const int v = N.order[k];
            val[v] = N.isOut[v] ? s : sigm(4.9 * s);
        }
        out.resize(N.nOut);
        for (int o = 0; o < N.nOut; ++o) out[o] = val[N.outIdx[o]];
    }
    int outputs() const { return m_out; }
    static int countParams(const NeatGenome& g) { int n = 0; for (const auto& c : g.conns) n += c.en ? 1 : 0; return n; }

private:
    int innovation(int in, int out) {
        const long long key = static_cast<long long>(in) * 1000003LL + out;
        auto it = m_innov.find(key);
        if (it != m_innov.end()) return it->second;
        const int v = m_nextInnov++;
        m_innov[key] = v;
        return v;
    }
    int splitNode(int innov) {   // the same split in one generation gets the same new node id
        auto it = m_genInnov.find(innov);
        if (it != m_genInnov.end()) return it->second;
        const int id = m_nextNode++;
        m_genInnov[innov] = id;
        return id;
    }
    static void sortConns(NeatGenome& g) { std::sort(g.conns.begin(), g.conns.end(), [](const NeatConn& a, const NeatConn& b) { return a.innov < b.innov; }); }
    double distance(const NeatGenome& a, const NeatGenome& b) const {
        size_t i = 0, j = 0; int disjoint = 0, matching = 0; double wd = 0;
        while (i < a.conns.size() && j < b.conns.size()) {
            if (a.conns[i].innov == b.conns[j].innov) { wd += std::fabs(a.conns[i].w - b.conns[j].w); ++matching; ++i; ++j; }
            else if (a.conns[i].innov < b.conns[j].innov) { ++disjoint; ++i; }
            else { ++disjoint; ++j; }
        }
        const int excess = static_cast<int>((a.conns.size() - i) + (b.conns.size() - j));
        const double N = std::max<size_t>(1, std::max(a.conns.size(), b.conns.size()) < 20 ? 1 : std::max(a.conns.size(), b.conns.size()));
        return m_c.c1 * excess / N + m_c.c2 * disjoint / N + m_c.c3 * (matching ? wd / matching : 0.0);
    }
    NeatGenome crossover(const NeatGenome& a0, const NeatGenome& b0) {
        const NeatGenome& a = a0.fitness >= b0.fitness ? a0 : b0;   // a = fitter parent
        const NeatGenome& b = a0.fitness >= b0.fitness ? b0 : a0;
        NeatGenome c; c.nNodes = std::max(a.nNodes, b.nNodes);
        size_t j = 0;
        std::uniform_real_distribution<double> u01(0, 1);
        for (const auto& ga : a.conns) {
            while (j < b.conns.size() && b.conns[j].innov < ga.innov) ++j;
            NeatConn g = ga;
            if (j < b.conns.size() && b.conns[j].innov == ga.innov) {
                if (u01(m_rng) < 0.5) g.w = b.conns[j].w;
                if (!ga.en || !b.conns[j].en) g.en = u01(m_rng) >= 0.75;
            }
            c.conns.push_back(g);   // disjoint / excess genes from the fitter parent
        }
        if (hasCycle(c)) return a;   // safety: keep the network feed-forward
        return c;
    }
    bool reaches(const NeatGenome& g, int from, int to) const {   // path from -> to over all genes
        std::vector<int> st{from}; std::vector<char> seen(g.nNodes, 0);
        while (!st.empty()) {
            const int v = st.back(); st.pop_back();
            if (v == to) return true;
            if (v < 0 || v >= g.nNodes || seen[v]) continue;
            seen[v] = 1;
            for (const auto& c : g.conns) if (c.in == v) st.push_back(c.out);
        }
        return false;
    }
    bool hasCycle(const NeatGenome& g) const {
        for (const auto& c : g.conns) if (reaches(g, c.out, c.in)) return true;
        return false;
    }
    void mutate(NeatGenome& g) {
        std::uniform_real_distribution<double> u01(0, 1), uw(-m_c.weightRange, m_c.weightRange);
        std::normal_distribution<double> nd(0.0, m_c.perturbSd);
        if (u01(m_rng) < m_c.pWeightMut)
            for (auto& c : g.conns) c.w = u01(m_rng) < m_c.pPerturb ? c.w + nd(m_rng) : uw(m_rng);
        for (auto& c : g.conns) if (u01(m_rng) < m_c.pToggle) c.en = !c.en;
        if (u01(m_rng) < m_c.pAddNode && !g.conns.empty()) {
            std::vector<int> enabled;
            for (int k = 0; k < static_cast<int>(g.conns.size()); ++k) if (g.conns[k].en) enabled.push_back(k);
            if (!enabled.empty()) {
                const int k = enabled[m_rng() % enabled.size()];
                g.conns[k].en = false;
                const int nid = splitNode(g.conns[k].innov);
                bool exists = false;
                for (const auto& c : g.conns) if (c.in == nid || c.out == nid) exists = true;
                if (!exists) {
                    g.nNodes = std::max(g.nNodes, nid + 1);
                    const NeatConn old = g.conns[k];
                    g.conns.push_back({old.in, nid, 1.0, true, innovation(old.in, nid)});
                    g.conns.push_back({nid, old.out, old.w, true, innovation(nid, old.out)});
                    sortConns(g);
                } else g.conns[k].en = true;
            }
        }
        if (u01(m_rng) < m_c.pAddConn) {
            std::vector<int> nodes;   // nodes present in this genome
            {
                std::vector<char> on(g.nNodes, 0);
                for (int v = 0; v < m_firstOut + m_out; ++v) on[v] = 1;
                for (const auto& c : g.conns) { on[c.in] = 1; on[c.out] = 1; }
                for (int v = 0; v < g.nNodes; ++v) if (on[v]) nodes.push_back(v);
            }
            for (int attempt = 0; attempt < 20; ++attempt) {
                const int from = nodes[m_rng() % nodes.size()];
                const int to = nodes[m_rng() % nodes.size()];
                if (to <= m_bias) continue;                                  // no edges into inputs / bias
                if (from >= m_firstOut && from < m_firstOut + m_out) continue;  // outputs do not feed others
                if (from == to) continue;
                bool dup = false; for (const auto& c : g.conns) if (c.in == from && c.out == to) dup = true;
                if (dup || reaches(g, to, from)) continue;                   // keep it acyclic
                g.conns.push_back({from, to, uw(m_rng), true, innovation(from, to)});
                sortConns(g);
                break;
            }
        }
    }
    double lossOf(const NeatGenome& g) const {
        const int n = m_D.n;
        const NeatNet N = compile(g);
        double total = 0;
#ifdef _OPENMP
        #pragma omp parallel for reduction(+:total) schedule(static) if (n > 512)
#endif
        for (int r = 0; r < n; ++r) {
            Vec val, out;
            forward(N, m_D.X[r], val, out);
            if (!m_D.cls) { const double e = out[0] - m_D.ys[r]; total += e * e; }
            else if (m_out == 1) {
                const double p = sigm(out[0]); const double t = m_D.yc[r] == 1 ? 1.0 : 0.0;
                total += -(t * std::log(std::max(1e-12, p)) + (1 - t) * std::log(std::max(1e-12, 1 - p)));
            } else {
                double mx = out[0]; for (double v : out) mx = std::max(mx, v);
                double se = 0; for (double v : out) se += std::exp(v - mx);
                total += mx + std::log(se) - out[m_D.yc[r]];
            }
        }
        const double L = total / std::max(1, n);
        return std::isfinite(L) ? L : 1e30;
    }

    const BLData& m_D; NeatCfg m_c; std::mt19937& m_rng;
    int m_in = 0, m_out = 1, m_bias = 0, m_firstOut = 0;
    std::vector<int> m_rows;
    std::map<long long, int> m_innov; int m_nextInnov = 0; int m_nextNode = 0;
    std::map<int, int> m_genInnov;
};

void predictNeat(const Neat& N, const NeatGenome& g, const BLData& D, const Encoder& enc, std::vector<int>& pc, Vec& pr) {
    pc.assign(D.n, 0); pr.assign(D.n, 0.0);
    const NeatNet net = N.compile(g);
    Vec val, out;
    for (int r = 0; r < D.n; ++r) {
        Neat::forward(net, D.X[r], val, out);
        if (!D.cls) pr[r] = out[0] * enc.yStd + enc.yMean;
        else if (N.outputs() == 1) pc[r] = sigm(out[0]) >= 0.5 ? 1 : 0;
        else pc[r] = static_cast<int>(std::max_element(out.begin(), out.end()) - out.begin());
    }
}

Vec trainGeneticPaper(const Mlp& m, const BLData& D, const BaselineConfig& c, std::mt19937& rng,
                      std::atomic<bool>& stop, qint64& evals) {
    std::vector<int> rows(D.n); std::iota(rows.begin(), rows.end(), 0);
    PaperGA g; g.Nc = c.gaPop; g.Ng = c.gaGens; g.ps = c.gaSelection; g.pm = c.gaMutation;
    const int P = m.nParams();
    evals = 0;
    return runPaperGA([&](const Vec& w) { return m.loss(w, D, rows, nullptr); },
                      Vec(P, -c.gaBound), Vec(P, c.gaBound), g, rng, stop, evals);
}

} // namespace

QStringList geBaselineIds() { return {"adam", "bfgs", "genetic", "neat", "rbf", "knn", "linear", "rf", "svm", "gnn"}; }

QVector<BaselineRow> geRunBaselines(const Dataset& train, const Dataset* test, const QStringList& idsIn,
                                    const BaselineConfig& cfg, std::atomic<bool>& stop,
                                    const std::function<void(int, int, const QString&)>& progress) {
    QVector<BaselineRow> out;
    QStringList ids;
    for (const QString& id : geBaselineIds()) if (idsIn.contains(id)) ids << id;   // report order
    if (ids.isEmpty() || train.sampleCount == 0) return out;

    Encoder enc;
    enc.fit(train);
    const BLData Dtr = enc.encode(train);
    BLData Dte;
    const bool hasTest = test && test->sampleCount > 0 && test->inputCount == train.inputCount;
    if (hasTest) Dte = enc.encode(*test);

    int done = 0;
    for (const QString& id : ids) {
        if (stop.load()) break;
        if (progress) progress(done, ids.size(), id);
        // independent, reproducible stream per model
        // FNV-1a of the id: qHash(QString) is seeded randomly per process in Qt 6, so the
        // reference models drew different random streams in every run of the same job
        quint32 idHash = 2166136261u;
        for (const QChar ch : id) { idHash ^= ch.unicode(); idHash *= 16777619u; }
        std::mt19937 rng(cfg.seed * 7919u + idHash);
        const double t0 = nowSec();
        BaselineRow row; row.id = id;
        std::vector<int> pcTr, pcTe; Vec prTr, prTe;
        if (id == "adam" || id == "bfgs" || id == "genetic") {
            Mlp m; m.init(Dtr, cfg.hidden);
            Vec w;
            qint64 ev = 0;
            if (id == "adam") w = trainAdam(m, Dtr, cfg, rng, stop, ev);
            else if (id == "bfgs") w = trainBfgs(m, Dtr, cfg, rng, stop, ev);
            else w = trainGeneticPaper(m, Dtr, cfg, rng, stop, ev);
            if (stop.load()) break;
            m.predict(w, Dtr, pcTr, prTr, enc);
            if (hasTest) m.predict(w, Dte, pcTe, prTe, enc);
            row.params = m.nParams(); row.evals = ev;
        } else if (id == "rbf") {
            qint64 ev = 0;
            const Rbf R = trainRbf(Dtr, cfg, rng, ev);
            predictRbf(R, Dtr, enc, pcTr, prTr);
            if (hasTest) predictRbf(R, Dte, enc, pcTe, prTe);
            row.params = R.nParams(Dtr.d); row.evals = ev;
        } else if (id == "gnn") {
            Mlp m; m.init(Dtr, cfg.hidden);
            GnnCfg gc;
            gc.ga.Nc = cfg.gaPop; gc.ga.Ng = cfg.gaGens; gc.ga.ps = cfg.gaSelection; gc.ga.pm = cfg.gaMutation;
            gc.initBound = cfg.gaBound;
            qint64 ev = 0; QString det;
            const Vec w = trainGnn(m, Dtr, gc, rng, stop, ev, det);
            if (stop.load()) break;
            m.predict(w, Dtr, pcTr, prTr, enc);
            if (hasTest) m.predict(w, Dte, pcTe, prTe, enc);
            row.params = m.nParams(); row.evals = ev; row.detail = det;
        } else if (id == "neat") {
            NeatCfg nc; nc.gens = cfg.gaGens;
            Neat N(Dtr, nc, rng);
            qint64 ev = 0; int gensRun = 0;
            const NeatGenome g = N.run(stop, ev, gensRun);
            if (stop.load()) break;
            predictNeat(N, g, Dtr, enc, pcTr, prTr);
            if (hasTest) predictNeat(N, g, Dte, enc, pcTe, prTe);
            int hidden = 0;
            {
                std::set<int> nodes;
                for (const auto& c : g.conns) if (c.en) { nodes.insert(c.in); nodes.insert(c.out); }
                for (int v : nodes) if (v >= Dtr.d + 1 + N.outputs()) ++hidden;
            }
            row.params = Neat::countParams(g); row.evals = ev;
            row.detail = QString("%1 hidden nodes, %2 enabled connections").arg(hidden).arg(row.params);
        } else if (id == "knn") {
            const int k = knnChooseK(Dtr, rng);
            std::vector<std::vector<int>> pc; std::vector<Vec> pr;
            knnPredictAll(Dtr, Dtr, {k}, pc, pr);
            pcTr = pc[0]; prTr = toRaw(pr[0], enc);
            if (hasTest) { knnPredictAll(Dtr, Dte, {k}, pc, pr); pcTe = pc[0]; prTe = toRaw(pr[0], enc); }
            row.params = 0; row.evals = 0;
            row.detail = QString("k=%1").arg(k);
        } else if (id == "linear") {
            // L2 strength chosen on an 80/20 split of the train file, then refitted on all of it
            std::vector<int> tr, va; holdout(Dtr, rng, tr, va);
            const BLData T = subset(Dtr, tr), V = subset(Dtr, va);
            double bestL = 1e-3, bestS = std::numeric_limits<double>::infinity();
            qint64 ev = 0, e1 = 0;
            for (double lam : {1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0}) {
                const Linear M = fitLinear(T, lam, rng, e1); ev += e1;
                std::vector<int> pc; Vec ps; predictLinear(M, V, pc, ps);
                const double sc = scoreOf(V, pc, ps);
                if (sc < bestS - 1e-12) { bestS = sc; bestL = lam; }
            }
            const Linear M = fitLinear(Dtr, bestL, rng, e1); ev += e1;
            Vec ps;
            predictLinear(M, Dtr, pcTr, ps); prTr = toRaw(ps, enc);
            if (hasTest) { predictLinear(M, Dte, pcTe, ps); prTe = toRaw(ps, enc); }
            row.params = M.nParams(); row.evals = ev;
            row.detail = QString("lambda=%1").arg(bestL);
        } else if (id == "rf") {
            ForestCfg fc;
            fc.trees = 100;
            fc.mtry = Dtr.cls ? std::max(1, static_cast<int>(std::floor(std::sqrt(static_cast<double>(Dtr.d)))))
                              : std::max(1, Dtr.d / 3);
            fc.minLeaf = Dtr.cls ? 1 : 5;
            const std::vector<Tree> F = fitForest(Dtr, fc, rng());
            predictForest(F, Dtr, pcTr, prTr);
            if (hasTest) predictForest(F, Dte, pcTe, prTe);
            int nodes = 0; for (const Tree& T : F) nodes += static_cast<int>(T.nodes.size());
            row.params = nodes; row.evals = 0;
            row.detail = QString("%1 trees, mtry=%2, min leaf=%3").arg(fc.trees).arg(fc.mtry).arg(fc.minLeaf);
        } else if (id == "svm") {
            const double gamma = svmGamma(Dtr);
            // C chosen on an 80/20 split (at most 2000 / 1000 rows for speed), then refitted on all rows
            std::vector<int> tr, va; holdout(Dtr, rng, tr, va);
            if (tr.size() > 2000) { std::shuffle(tr.begin(), tr.end(), rng); tr.resize(2000); }
            if (va.size() > 1000) { std::shuffle(va.begin(), va.end(), rng); va.resize(1000); }
            const BLData T = subset(Dtr, tr), V = subset(Dtr, va);
            double bestC = 1, bestS = std::numeric_limits<double>::infinity();
            qint64 rowsK = 0;
            for (double C : {1.0, 10.0, 100.0}) {
                if (stop.load()) break;
                const SvmModel M = fitSvm(T, C, gamma, stop, rowsK);
                std::vector<int> pc; Vec ps; predictSvm(M, V, pc, ps);
                const double sc = scoreOf(V, pc, ps);
                if (sc < bestS - 1e-12) { bestS = sc; bestC = C; }
            }
            if (stop.load()) break;
            const SvmModel M = fitSvm(Dtr, bestC, gamma, stop, rowsK);
            if (stop.load()) break;
            Vec ps;
            predictSvm(M, Dtr, pcTr, ps); prTr = toRaw(ps, enc);
            if (hasTest) { predictSvm(M, Dte, pcTe, ps); prTe = toRaw(ps, enc); }
            row.params = M.nParams(); row.evals = rowsK;
            row.detail = QString("C=%1, gamma=%2").arg(bestC).arg(gamma, 0, 'g', 4);
        } else {
            continue;
        }
        row.trainErr = errorOf(Dtr, pcTr, prTr);
        row.testErr = hasTest ? errorOf(Dte, pcTe, prTe) : -1;
        row.seconds = nowSec() - t0;
        out.append(row);
        ++done;
        if (progress) progress(done, ids.size(), id);
    }
    return out;
}

} // namespace NeuralStudio
