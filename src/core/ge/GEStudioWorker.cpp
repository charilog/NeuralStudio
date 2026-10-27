#include "GEStudioWorker.h"
#include "core/ge/Grammar.h"
#include "core/ge/GEEngine.h"
#include "core/ge/NetSpec.h"
#include "core/ge/Baselines.h"
#include "core/dataset/DatasetLoader.h"
#include "core/optimizers/NSProblem.h"
#include "core/optimizers/optimizer.h"
#include "core/optimizers/genetic.h"
#include "core/optimizers/de.h"
#include "core/optimizers/pso.h"
#include "core/optimizers/cmaes.h"
#include "core/optimizers/jso.h"
#include "core/optimizers/lmcmaes.h"
#include "core/optimizers/acor.h"
#include "core/optimizers/clpso.h"
#include "core/optimizers/mlshaderl.h"
#include "core/optimizers/arq3.h"
#include "core/optimizers/simann.h"
#include "core/optimizers/neldermead.h"
#include "core/optimizers/lbfgs.h"
#include <QFileInfo>
#include <QtGlobal>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <map>
#include <set>
#include <random>
#include <unordered_map>

namespace NeuralStudio {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Diagnostic output of the self-adaptive rows: environment variable GE_ADAPT_DEBUG=1 (selection) or 2 (+ training).
int adaptDebug() {
    static const int level = qEnvironmentVariableIntValue("GE_ADAPT_DEBUG");
    return level;
}

// With a configured seed (the batch engine always sets one) every training seed is drawn
// from a stream seeded by it, so a run is reproducible; before, the network trainings used
// clock-based seeds and two runs of the same job gave different numbers.
thread_local std::mt19937_64* g_seedStream = nullptr;

unsigned int makeSeed() {
    if (g_seedStream) return static_cast<unsigned int>((*g_seedStream)() >> 16);
    static std::atomic<uint64_t> ctr{0};
    const auto t = static_cast<uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    return static_cast<unsigned int>(t ^ (++ctr * 6364136223846793005ULL));
}

double nowSec() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::unique_ptr<optimsolution::Optimizer> makeOpt(const QString& n) {
    if (n == "lbfgs")      { auto p = std::make_unique<optimsolution::LBFGS>(); p->setMemory(7);  return p; }
    if (n == "bfgs")       { auto p = std::make_unique<optimsolution::LBFGS>(); p->setMemory(20); return p; }
    if (n == "sa")         return std::make_unique<optimsolution::SimulatedAnnealing>();
    if (n == "neldermead") return std::make_unique<optimsolution::NelderMead>();
    if (n == "de")         return std::make_unique<optimsolution::DE>();
    if (n == "pso")        return std::make_unique<optimsolution::PSO>();
    if (n == "cmaes")      return std::make_unique<optimsolution::CMAES>();
    if (n == "jso")        return std::make_unique<optimsolution::JSO>();
    if (n == "lmcmaes")    return std::make_unique<optimsolution::LMCMAES>();
    if (n == "acor")       return std::make_unique<optimsolution::ACOR>();
    if (n == "clpso")      return std::make_unique<optimsolution::CLPSO>();
    if (n == "mlshaderl")  return std::make_unique<optimsolution::mLSHADE_RL>();
    if (n == "arq3")       return std::make_unique<optimsolution::ARQ3>();
    return std::make_unique<optimsolution::GeneticAlgorithm>();
}

Activation parseAct(const std::string& s) {
    if (s == "sigmoid") return Activation::Sigmoid;
    if (s == "tanh")    return Activation::Tanh;
    if (s == "linear")  return Activation::Linear;
    return Activation::ReLU;
}

std::vector<int> parseHidden(const QString& s) {
    std::vector<int> o;
    for (const QString& t : s.split(',', Qt::SkipEmptyParts)) {
        bool ok; const int n = t.trimmed().toInt(&ok);
        if (ok && n > 0) o.push_back(n);
    }
    return o.empty() ? std::vector<int>{16} : o;
}

std::shared_ptr<NeuralNetwork> buildNet(const Dataset* ds, const std::vector<int>& h,
                                        const std::vector<Activation>& acts) {
    NetworkConfig cfg;
    cfg.task = detectTask(ds);
    cfg.hiddenSizes = h;
    cfg.hiddenActivations = acts;
    auto uv = ds->uniqueOutputValues(0);
    const int netOut = (cfg.task == TaskType::MultiClassClassification)
                       ? static_cast<int>(uv.size())
                       : (cfg.task == TaskType::BinaryClassification) ? 1 : ds->outputCount;
    auto net = std::make_shared<NeuralNetwork>();
    net->build(ds->inputCount, netOut, cfg);
    net->setNormalization(ds->inputStats);
    net->fitOutputScaling(ds);   // regression: standardised output layer (train targets)
    if (cfg.task != TaskType::Regression) {
        net->setClassValues(uv);
        if (!uv.empty()) net->setOutputMapping(uv.front(), uv.back());
    } else if (!ds->outputStats.empty()) {
        net->setOutputMapping(ds->outputStats[0].min, ds->outputStats[0].max);
    }
    return net;
}

// Constructed features are computed on the raw inputs (so ratios such as x4/x1 keep
// their physical meaning), non-finite values are replaced, and every feature column
// is clipped to the 1st..99th percentile of the TRAIN rows. The network then min-max
// normalises it like any other input. Heavy tails (exp, products, ratios) would
// otherwise squeeze almost all rows into a tiny part of [0,1] and the feature would
// be useless. The test file uses the train percentiles.
struct FeatClip { std::vector<double> lo, hi; std::vector<double> emMin, emRange; };

// Periodic embedding: for every original input i (min-max scaled with the TRAIN statistics) and
// m frequencies c_ij = sigma * z_ij (z_ij fixed N(0,1) draws, the same for every dataset and run),
// the features sin(2 pi c u) and cos(2 pi c u) are appended. m = clamp(64 / d, 1, 4).
void appendEmbedding(const Dataset& base, Dataset& out, double sigma, FeatClip& clip, bool fit) {
    const int d = base.inputCount;
    if (sigma <= 0 || d <= 0) return;
    if (fit) {
        clip.emMin.assign(d, 0.0); clip.emRange.assign(d, 1.0);
        for (int i = 0; i < d; ++i) {
            double lo = std::numeric_limits<double>::infinity(), hi = -lo;
            for (int r = 0; r < base.sampleCount; ++r) { const double v = base.inputs[r][i]; if (std::isfinite(v)) { lo = std::min(lo, v); hi = std::max(hi, v); } }
            clip.emMin[i] = std::isfinite(lo) ? lo : 0.0;
            clip.emRange[i] = (std::isfinite(hi - lo) && hi - lo > 1e-12) ? hi - lo : 1.0;
        }
    }
    const int m = std::max(1, std::min(4, 64 / d));
    std::vector<double> c(static_cast<size_t>(d) * m);
    {
        std::mt19937_64 zr(0x5EEDF00DULL);   // fixed frequencies: part of the method, not of the run
        std::normal_distribution<double> nd(0.0, 1.0);
        for (double& v : c) v = sigma * nd(zr);
    }
    const double twoPi = 6.283185307179586;
    for (int r = 0; r < out.sampleCount; ++r) {
        auto& row = out.inputs[r];
        for (int i = 0; i < d; ++i) {
            const double x = base.inputs[r][i];
            const double u = std::isfinite(x) && i < static_cast<int>(clip.emMin.size()) ? (x - clip.emMin[i]) / clip.emRange[i] : 0.5;
            for (int j = 0; j < m; ++j) {
                const double a = twoPi * c[static_cast<size_t>(i) * m + j] * u;
                row.push_back(std::sin(a)); row.push_back(std::cos(a));
            }
        }
    }
    out.inputCount += 2 * m * d;
    for (int i = 0; i < 2 * m * d; ++i) out.inputNames << QString("P%1").arg(i + 1);
}

std::shared_ptr<Dataset> augment(const Dataset& base, const ge::NetSpec& spec,
                                 FeatClip& clip, bool fit) {
    const std::vector<std::string>& feats = spec.features;
    auto out = std::make_shared<Dataset>(base);
    if (feats.empty()) {
        if (spec.embSigma > 0) { appendEmbedding(base, *out, spec.embSigma, clip, fit); out->computeStatistics(); }
        return out;
    }
    std::vector<std::shared_ptr<ge::Expr>> ex;
    for (const auto& f : feats) {
        auto e = ge::Expr::parse(f, base.inputCount);
        if (e) ex.push_back(e);
    }
    const int k = static_cast<int>(ex.size());
    const int n = out->sampleCount;
    std::vector<std::vector<double>> col(k, std::vector<double>(n, 0.0));
    for (int r = 0; r < n; ++r) {
        const auto& row = out->inputs[r];
        for (int j = 0; j < k; ++j) {
            const double v = ex[j]->eval(row.data());
            col[j][r] = std::isfinite(v) ? v : 0.0;
        }
    }
    if (fit) {
        clip.lo.assign(k, 0.0); clip.hi.assign(k, 0.0);
        for (int j = 0; j < k; ++j) {
            std::vector<double> v = col[j];
            if (v.empty()) continue;
            const size_t a = static_cast<size_t>(0.01 * (v.size() - 1));
            const size_t b = static_cast<size_t>(0.99 * (v.size() - 1));
            std::nth_element(v.begin(), v.begin() + a, v.end()); clip.lo[j] = v[a];
            std::nth_element(v.begin(), v.begin() + b, v.end()); clip.hi[j] = v[b];
        }
    }
    for (int r = 0; r < n; ++r) {
        auto& row = out->inputs[r];
        for (int j = 0; j < k; ++j) {
            double v = col[j][r];
            if (j < static_cast<int>(clip.lo.size())) v = std::min(clip.hi[j], std::max(clip.lo[j], v));
            row.push_back(v);
        }
    }
    out->inputCount += k;
    for (int i = 0; i < k; ++i) out->inputNames << QString("F%1").arg(i + 1);
    appendEmbedding(base, *out, spec.embSigma, clip, fit);
    out->computeStatistics();
    return out;
}

// NSProblem with per-parameter bounds (the grammar's bound commands).
class BoundedProblem : public optimsolution::Problem {
public:
    BoundedProblem(NeuralNetwork* n, const Dataset* d, std::vector<double> lb,
                   std::vector<double> ub, double valSplit, unsigned splitSeed = 42)
        : m_inner(n, d, 1.0, valSplit, splitSeed), m_lb(std::move(lb)), m_ub(std::move(ub)) {}
    int dimension() const override { return static_cast<int>(m_lb.size()); }
    const std::vector<double>& lb() const override { return m_lb; }
    const std::vector<double>& ub() const override { return m_ub; }
    double evaluate(const std::vector<double>& x) override { return m_inner.evaluate(x); }
    // A loss+gradient evaluation (forward + backward pass) is charged as kGradCost loss
    // evaluations, so budgets stay comparable between gradient-free and gradient-based training.
    static constexpr int kGradCost = 3;
    int calls() const override { return m_inner.calls() + (kGradCost - 1) * m_gradCalls; }
    std::pair<double, std::vector<double>> evalAndGrad(const std::vector<double>& x) override {
        ++m_gradCalls;
        return m_parallelGrad ? m_inner.evalAndGradParallel(x) : m_inner.evalAndGrad(x);
    }
    void setParallelGradient(bool on) { m_parallelGrad = on; }
    NSProblem& inner() { return m_inner; }
private:
    NSProblem m_inner;
    std::vector<double> m_lb, m_ub;
    int m_gradCalls = 0;
    bool m_parallelGrad = false;
};

struct TrainParams {
    QString opt = "de";
    int     pop = 30;
    int     iters = 60;
    double  baseBound = 10.0;
    double  valSplit = 0.2;
    bool    polish = false;
    int     polishEvals = 400;
    QString boundMode = "fan";
    double  fanK = 3.0;
    // self-adaptive extensions
    bool     evalOnly = false;   // with x0: only evaluate x0 (no training)
    bool     wantPreds = false;  // fill the per-sample predictions of the Outcome
    bool     adaptive = false;   // self-adaptive training with validation early stopping (see adaptiveTrain)
    qint64   budget = 0;         // adaptive: loss evaluations to spend
    unsigned splitSeed = 42;     // train / validation partition of the adaptive training
    int      cvK = 0;            // adaptive: > 1 = validation is fold cvFold of a cvK-fold partition
    int      cvFold = 0;
};

struct Outcome {
    double trainErr = 0, valErr = -1, testErr = -1, trainLoss = 0, valLoss = -1;
    int    params = 0;
    int    inputs = 0;
    qint64 calls = 0;
    bool   stopped = false;
    std::vector<double> x;          // trained parameters (inside the bounds)
    std::shared_ptr<NeuralNetwork> net;
    std::shared_ptr<Dataset>       trainDs;
    // noise-aware selection / ensembles
    double valLossSE = -1;          // standard error of the mean validation loss (search training)
    int    kind = 0;                // 0 regression, 1 binary, 2 multi-class
    std::vector<std::vector<double>> valPred, valTarget, testPred, testTarget, trainPred, trainTarget;
};

// Per-sample loss / correctness of a prediction in probability space (the metrics of NSProblem).
double predLossK(int kind, const std::vector<double>& p, const std::vector<double>& t) {
    const double eps = 1e-7;
    double s = 0.0;
    if (kind == 2) { for (size_t i = 0; i < p.size() && i < t.size(); ++i) s -= t[i] * std::log(std::max(eps, p[i])); }
    else if (kind == 1) { const double q = std::max(eps, std::min(1.0 - eps, p[0])); s -= t[0] * std::log(q) + (1.0 - t[0]) * std::log(1.0 - q); }
    else {   // mean over outputs, as NSProblem's loss
        size_t m = 0;
        for (size_t i = 0; i < p.size() && i < t.size(); ++i, ++m) { const double d = p[i] - t[i]; s += d * d; }
        if (m > 0) s /= static_cast<double>(m);
    }
    return s;
}
bool predCorrectK(int kind, const std::vector<double>& p, const std::vector<double>& t) {
    if (kind == 2) return std::max_element(p.begin(), p.end()) - p.begin() == std::max_element(t.begin(), t.end()) - t.begin();
    return (p[0] >= 0.5) == (t[0] >= 0.5);
}
// mean and standard error of a sample
void meanSE(const std::vector<double>& v, double& mean, double& se) {
    mean = 0; se = 0;
    if (v.empty()) return;
    for (double x : v) mean += x;
    mean /= static_cast<double>(v.size());
    if (v.size() < 2) return;
    double ss = 0;
    for (double x : v) ss += (x - mean) * (x - mean);
    se = std::sqrt(ss / static_cast<double>(v.size() - 1) / static_cast<double>(v.size()));
}
// Error of averaged predictions: % wrong (classification) or mean squared error (regression).
double predError(int kind, const std::vector<std::vector<double>>& p, const std::vector<std::vector<double>>& t) {
    if (p.empty()) return -1;
    double s = 0;
    for (size_t i = 0; i < p.size(); ++i)
        s += kind == 0 ? predLossK(0, p[i], t[i]) : (predCorrectK(kind, p[i], t[i]) ? 0.0 : 1.0);
    return kind == 0 ? s / static_cast<double>(p.size()) : 100.0 * s / static_cast<double>(p.size());
}

// Self-adaptive training inside the grammar's intervals, with the budget tp.budget:
//  1. global phase (tp.opt, e.g. DE) in blocks of 5 generations, until a block improves the training loss
//     by less than 1 % or 30 % of the budget is spent;
//  2. bound-respecting L-BFGS in chunks; after every chunk the validation loss (held-out tp.valSplit of the
//     training rows) is measured and the best-validation weights are kept; it stops when the validation
//     loss has not improved for 10 chunks (early stopping) or the budget is spent;
//  3. when L-BFGS converges (an iteration evaluates nothing) while budget remains, it restarts from the
//     best-validation weights perturbed by 5 % of the interval widths (iterated local search).
// Every decision is driven by the progress measured during the run; nothing depends on the dataset size.
std::vector<double> adaptiveTrain(BoundedProblem& prob, NeuralNetwork& net, const std::vector<double>& lb,
                                  const std::vector<double>& ub, const TrainParams& tp, std::atomic<bool>& stop,
                                  double& bestTrainLoss) {
    const qint64 start = prob.calls();
    const qint64 B = std::max<qint64>(tp.budget, 200);
    auto spent = [&]() { return static_cast<qint64>(prob.calls()) - start; };
    std::mt19937 rng(tp.splitSeed * 2654435761u + 17u);

    // 1. global phase
    auto opt = makeOpt(tp.opt);
    const qint64 globalCap = std::max<qint64>(static_cast<qint64>(0.3 * static_cast<double>(B)), tp.pop * 5);
    opt->setup(&prob, static_cast<int>(std::min<qint64>(globalCap, std::numeric_limits<int>::max() / 4)) + prob.calls(),
               tp.pop, rng());
    opt->init();
    double prev = kInf;
    while (!stop.load() && spent() < globalCap && !opt->done()) {
        for (int k = 0; k < 5 && !opt->done() && !stop.load(); ++k) { opt->one_iteration(); opt->advanceIter(); }
        const double b = opt->bestF();
        if (std::isfinite(prev) && prev - b < 0.01 * std::max(1e-12, std::fabs(prev))) break;
        prev = b;
    }
    opt->end();
    std::vector<double> x = opt->bestX();
    if (x.empty()) return x;
    auto clipTo = [&](std::vector<double>& w) { for (size_t i = 0; i < w.size(); ++i) w[i] = std::min(ub[i], std::max(lb[i], w[i])); };
    auto valOf = [&](const std::vector<double>& w) { NSProblem::unflatten(net, w); return prob.inner().valLoss(); };
    clipTo(x);
    std::vector<double> bestX = x;
    double bestVal = valOf(x);
    bestTrainLoss = opt->bestF();
    if (adaptDebug() >= 2)
        std::fprintf(stderr, "  [train] global phase: %lld evals, train %.4g val %.4g (dim %zu)\n",
                     static_cast<long long>(spent()), bestTrainLoss, bestVal, x.size());

    // 2./3. local descents with early stopping; a descent ends when its validation loss has not improved
    // for as long as it took to reach its best (at least 20 chunks), or when L-BFGS converges.
    // While budget remains, the next descent starts around the best-validation weights (iterated local search),
    // so every row spends its whole budget, like the other rows.
    const qint64 chunk = std::max<qint64>(60, B / 400);
    int restarts = 0;
    std::vector<double> from = x;
    while (!stop.load() && spent() < B && restarts <= 20) {
        optimsolution::LBFGS bfgs;
        bfgs.setMemory(20);
        bfgs.setActiveSetProjection(true);
        bfgs.setup(&prob, static_cast<int>(start + B), 1, rng());
        bfgs.setInitialPoint(from);
        bfgs.init();
        const qint64 d0 = spent();
        qint64 atBest = d0;
        double descentBest = kInf;
        while (!stop.load() && spent() < B) {
            const qint64 end = std::min<qint64>(start + B, prob.calls() + chunk);
            bool converged = false;
            while (prob.calls() < end && !stop.load()) {
                const int c0 = prob.calls();
                bfgs.one_iteration(); bfgs.advanceIter();
                if (prob.calls() == c0) { converged = true; break; }
            }
            std::vector<double> cur = bfgs.bestX();
            clipTo(cur);
            const double v = valOf(cur);
            if (std::isfinite(v) && (!std::isfinite(descentBest) || v < descentBest - 1e-12 * std::max(1.0, std::fabs(descentBest)))) {
                descentBest = v; atBest = spent();
                if (v < bestVal) { bestVal = v; bestX = cur; bestTrainLoss = std::min(bestTrainLoss, bfgs.bestF()); }
            }
            const qint64 patience = std::max<qint64>(20 * chunk, atBest - d0);
            if (adaptDebug() >= 2)
                std::fprintf(stderr, "  [train] descent %d spent %lld/%lld train %.4g val %.4g best val %.4g%s\n", restarts,
                             static_cast<long long>(spent()), static_cast<long long>(B), bfgs.bestF(), v, bestVal, converged ? " converged" : "");
            if (converged || spent() - atBest > patience) break;
        }
        ++restarts;
        from = bestX;
        for (size_t i = 0; i < from.size(); ++i) {
            std::normal_distribution<double> nd(0.0, 0.05 * (ub[i] - lb[i]));
            from[i] += nd(rng);
        }
        clipTo(from);
    }
    return bestX;
}

// Train one phenotype inside the grammar-derived intervals.
// When `test` is given, all training rows are used for training and the test
// file provides testErr; otherwise a held-out slice (valSplit) gives valErr.
// x0 != nullptr: Lamarckian resume. No global phase; projected L-BFGS continues from x0.
Outcome trainOne(const ge::NetSpec& spec, const Dataset& train, const Dataset* test,
                 const TrainParams& tp, std::atomic<bool>& stop,
                 const std::vector<double>* x0 = nullptr) {
    Outcome o;
    FeatClip clip;
    auto trainA = augment(train, spec, clip, true);
    std::shared_ptr<Dataset> testA;
    if (test) testA = augment(*test, spec, clip, false);

    std::vector<int> sizes;
    std::vector<Activation> acts;
    for (const auto& l : spec.hidden) { sizes.push_back(l.size); acts.push_back(parseAct(l.act)); }
    auto net = buildNet(trainA.get(), sizes, acts);
    net->setTrainingMode(false);

    // Per-parameter bounds, in NSProblem::flatten order (weights then biases, layer by layer).
    std::vector<double> lb, ub;
    auto& layers = net->layers();
    for (size_t li = 0; li < layers.size(); ++li) {
        const double lf = li < spec.hidden.size() ? spec.hidden[li].lbFactor : spec.outLbFactor;
        const double uf = li < spec.hidden.size() ? spec.hidden[li].ubFactor : spec.outUbFactor;
        const int n = layers[li].paramCount();
        // Scale-aware base: the Glorot limit of the layer times fanK, so SHRINK/EXPAND act
        // relative to the natural weight scale of that layer instead of a global ±B.
        const double base = tp.boundMode == "abs"
            ? tp.baseBound
            : tp.fanK * std::sqrt(6.0 / std::max(1, layers[li].inputSize() + layers[li].outputSize()));
        lb.insert(lb.end(), n, -base * lf);
        ub.insert(ub.end(), n, +base * uf);
    }
    if (static_cast<int>(lb.size()) != net->totalParams())
        throw std::runtime_error("internal: bound vector does not match the parameter count");

    const bool adaptive = tp.adaptive && !tp.evalOnly;
    // the adaptive training holds out tp.valSplit of the training rows for early stopping, also in the final rows
    const double vs = adaptive ? tp.valSplit : (test ? 0.0 : tp.valSplit);
    BoundedProblem prob(net.get(), trainA.get(), lb, ub, vs, adaptive ? tp.splitSeed : 42u);
    if (testA && !adaptive) prob.inner().setExternalValDataset(testA.get());
    if (adaptive) prob.setParallelGradient(true);   // exact gradient, computed in parallel over the samples
    if (adaptive && tp.cvK > 1) prob.inner().setFoldSplit(tp.cvFold, tp.cvK);

    std::vector<double> best;
    double bestLoss = kInf;
    if (adaptive) {
        best = adaptiveTrain(prob, *net, lb, ub, tp, stop, bestLoss);
    } else if (x0 && x0->size() == lb.size()) {
        best = *x0;
        for (size_t i = 0; i < best.size(); ++i) best[i] = std::min(ub[i], std::max(lb[i], best[i]));
        bestLoss = prob.evaluate(best);
    } else {
        auto opt = makeOpt(tp.opt);
        opt->setup(&prob, tp.iters * tp.pop, tp.pop, makeSeed());
        opt->init();
        for (int it = 1; it <= tp.iters && !stop.load() && !opt->done(); ++it) {
            opt->one_iteration();
            opt->advanceIter();
        }
        opt->end();
        best = opt->bestX();
        bestLoss = opt->bestF();
    }

    if ((tp.polish || x0) && !tp.evalOnly && !adaptive && !stop.load() && !best.empty()) {
        auto bfgs = std::make_unique<optimsolution::LBFGS>();
        bfgs->setMemory(20);
        bfgs->setup(&prob, prob.calls() + tp.polishEvals, 1, makeSeed());
        bfgs->setInitialPoint(best);
        bfgs->init();
        // An iteration that evaluates nothing (zero gradient, e.g. all units saturated or dead)
        // would repeat forever, because done() waits for the evaluation budget: stop there.
        while (!bfgs->done() && !stop.load()) {
            const int before = prob.calls();
            bfgs->one_iteration(); bfgs->advanceIter();
            if (prob.calls() == before) break;
        }
        bfgs->end();
        // L-BFGS is unconstrained: project onto the grammar's intervals and keep the
        // result only if it is still not worse, so the bound commands stay in force.
        std::vector<double> cand = bfgs->bestX();
        if (!cand.empty()) {
            for (size_t i = 0; i < cand.size(); ++i) cand[i] = std::min(ub[i], std::max(lb[i], cand[i]));
            const double f = prob.evaluate(cand);
            if (std::isfinite(f) && f <= bestLoss) { best = cand; bestLoss = f; }
        }
    }
    o.calls = prob.calls();
#ifdef NS_GE_DEBUG
    {
        double viol = 0;
        for (size_t i = 0; i < best.size(); ++i)
            viol = std::max(viol, std::max(lb[i] - best[i], best[i] - ub[i]));
        std::fprintf(stderr, "[bounds] %s  max violation = %g\n", spec.describe().c_str(), viol);
    }
#endif
    if (!best.empty()) NSProblem::unflatten(*net, best);
    o.x = best;

    const bool cls = net->taskType() != TaskType::Regression;
    o.trainLoss = bestLoss;
    if (!test && vs > 0) o.valLoss = prob.inner().valLoss();
    o.params = net->totalParams();
    o.inputs = trainA->inputCount;
    if (cls) {
        // accuracies are -1 for an empty row set (which used to become a 200 % error)
        const double ta = prob.inner().trainAcc(), va = prob.inner().valAcc();
        o.trainErr = ta >= 0 ? (1.0 - ta) * 100.0 : -1;
        if (test)        o.testErr = va >= 0 ? (1.0 - va) * 100.0 : -1;
        else if (vs > 0) o.valErr  = va >= 0 ? (1.0 - va) * 100.0 : -1;
    } else {
        o.trainErr = bestLoss;
        if (test)        o.testErr = prob.inner().valLoss();
        else if (vs > 0) o.valErr  = prob.inner().valLoss();
    }
    o.kind = net->taskType() == TaskType::MultiClassClassification ? 2 : net->taskType() == TaskType::BinaryClassification ? 1 : 0;
    if (adaptive) {
        // metrics of the best-validation weights on the whole train file and on the test file
        NSProblem full(net.get(), trainA.get(), 1.0, 0.0);
        full.predictions(false, o.trainPred, o.trainTarget);
        o.trainErr = predError(o.kind, o.trainPred, o.trainTarget);
        o.valLoss = prob.inner().valLoss();
        if (testA) {
            NSProblem tq(net.get(), testA.get(), 1.0, 0.0);
            tq.predictions(false, o.testPred, o.testTarget);
            o.testErr = predError(o.kind, o.testPred, o.testTarget);
        }
    } else if (!test && vs > 0) {
        // search training: the standard error of the validation loss (noise-aware rules)
        prob.inner().predictions(true, o.valPred, o.valTarget);
        std::vector<double> l(o.valPred.size());
        for (size_t i = 0; i < l.size(); ++i) l[i] = predLossK(o.kind, o.valPred[i], o.valTarget[i]);
        double m = 0; meanSE(l, m, o.valLossSE);
        if (!tp.wantPreds) { o.valPred.clear(); o.valTarget.clear(); }
    } else if (test && tp.wantPreds) {
        prob.inner().predictions(true, o.testPred, o.testTarget);
    }
    o.stopped = stop.load();
    o.net = net;
    o.trainDs = trainA;
    return o;
}

struct CacheEntry {
    ge::NetSpec spec;
    std::vector<double> x;  // trained weights (Lamarckian resume)
    int    refines = 0;
    double fitness = kInf;
    double trainErr = 0, valErr = -1, valLoss = -1;
    double valLossSE = 0;   // standard error of valLoss (noise-aware rules)
    int    params = 0, inputs = 0;
};

// Human-readable production label: its terminals, <nt> in brackets.
std::string prodLabel(const ge::Production& p) {
    std::string s;
    for (const auto& sym : p.syms) s += sym.terminal ? sym.text : "<" + sym.text + ">";
    if (s.empty()) s = "(empty)";
    if (s.size() > 30) s = s.substr(0, 27) + "...";
    return s;
}

QString buildProbsReport(const ge::Grammar& g, const std::vector<std::vector<double>>& p,
                         const QString& header) {
    QString out = header + "\n";
    for (size_t i = 0; i < g.rules().size(); ++i) {
        const auto& r = g.rules()[i];
        if (r.prods.size() < 2) continue;
        std::vector<int> ord(r.prods.size());
        for (size_t k = 0; k < ord.size(); ++k) ord[k] = static_cast<int>(k);
        std::sort(ord.begin(), ord.end(), [&](int a, int b) { return p[i][a] > p[i][b]; });
        out += QString("<%1>:").arg(QString::fromStdString(r.name));
        const size_t shown = std::min<size_t>(ord.size(), 7);
        for (size_t k = 0; k < shown; ++k)
            out += QString("  %1 %2").arg(QString::fromStdString(prodLabel(r.prods[ord[k]])))
                                    .arg(p[i][ord[k]], 0, 'f', 2);
        if (ord.size() > shown) out += QString("  (+%1 more)").arg(ord.size() - shown);
        out += "\n";
    }
    return out;
}

} // namespace

// ── Worker ────────────────────────────────────────────────────────────────────
GEStudioWorker::GEStudioWorker(QObject* parent) : QObject(parent) {}

void GEStudioWorker::run() {
    try {
        if (m_cfg.datasetPaths.isEmpty()) throw std::runtime_error("No datasets queued.");
        m_stop.store(false);
        m_lastNet.reset(); m_lastDs.reset(); m_prior.clear();
        for (int i = 0; i < m_cfg.datasetPaths.size() && !m_stop.load(); ++i)
            if (!runOnDataset(i, m_cfg.datasetPaths[i])) break;
        emit allDone(m_lastNet, m_lastDs);
    } catch (const std::exception& ex) {
        emit errorOccurred(QString::fromStdString(ex.what()));
    }
}

namespace {
GEFinalRow baselineFinalRow(const BaselineRow& b) {
    GEFinalRow r;
    r.id = b.id; r.label = b.detail.isEmpty() ? b.id : b.id + " (" + b.detail + ")"; r.trainErr = b.trainErr; r.testErr = b.testErr;
    r.params = b.params; r.evals = b.evals; r.seconds = b.seconds; r.isGE = false;
    return r;
}
} // namespace

bool GEStudioWorker::runOnDataset(int dsIdx, const QString& dsPath) {
    const QString dsName = QFileInfo(dsPath).baseName();
    std::mt19937_64 seedStream(0x9E3779B97F4A7C15ULL ^ (static_cast<uint64_t>(m_cfg.seed) * 1000003ULL + static_cast<uint64_t>(dsIdx)));
    struct StreamGuard { explicit StreamGuard(std::mt19937_64* s) { g_seedStream = s; } ~StreamGuard() { g_seedStream = nullptr; } };
    StreamGuard guard(m_cfg.seed ? &seedStream : nullptr);
    DatasetBundle bundle = DatasetLoader::loadBundle(dsPath);
    if (!bundle.train) throw std::runtime_error("Failed to load dataset: " + dsPath.toStdString());
    bundle.train->computeStatistics();
    const bool hasTest = bundle.hasTest() &&
        bundle.test->inputCount == bundle.train->inputCount &&
        bundle.test->outputCount == bundle.train->outputCount;
    if (hasTest) bundle.test->computeStatistics();
    const Dataset& train = *bundle.train;
    const Dataset* test = hasTest ? bundle.test.get() : nullptr;
    const bool isCls = detectTask(&train) != TaskType::Regression;

    // Reference models only (no grammar, no search)
    if (m_cfg.mode == "baselines") {
        emit datasetStarted(dsIdx, m_cfg.datasetPaths.size(), dsName,
            QString("baselines: %1 inputs, %2 train samples%3").arg(train.inputCount).arg(train.sampleCount)
                .arg(hasTest ? QString(", %1 test samples").arg(test->sampleCount) : QString(", no test file")));
        QStringList ids = m_cfg.methods.isEmpty() ? geBaselineIds() : m_cfg.methods;
        QStringList run;
        for (const QString& id : geBaselineIds()) if (ids.contains(id)) run << id;
        GEDatasetResult res;
        res.dsName = dsName; res.isCls = isCls; res.hasTest = hasTest;
        res.trainSamples = train.sampleCount; res.testSamples = hasTest ? test->sampleCount : 0;
        emit finalStageStarted(std::max(1, static_cast<int>(run.size())));
        BaselineConfig bc;
        bc.seed = m_cfg.seed ? m_cfg.seed + static_cast<quint32>(dsIdx) : makeSeed();
        const double t0 = nowSec();
        const QVector<BaselineRow> rows = geRunBaselines(train, test, run, bc, m_stop,
            [&](int done, int total, const QString& id) {
                if (done > 0) emit finalRowDone(done, total, id);
            });
        for (const BaselineRow& b : rows) res.rows.append(baselineFinalRow(b));
        res.searchSeconds = nowSec() - t0;
        emit datasetFinished(res);
        return !m_stop.load();
    }

    // Grammar
    const std::string bnf = m_cfg.grammarText.trimmed().isEmpty()
        ? ge::defaultUnifiedGrammar(train.inputCount)
        : ge::substituteVars(m_cfg.grammarText.toStdString(), train.inputCount);
    ge::Grammar grammar;
    try { grammar = ge::Grammar::parse(bnf); }
    catch (const std::exception& ex) { throw std::runtime_error(std::string("Grammar error: ") + ex.what()); }

    ge::InterpretOptions io;
    io.maxLayers = m_cfg.maxLayers;
    io.maxFeatures = m_cfg.maxFeatures;
    io.maxFeatureSize = m_cfg.maxFeatureSize;
    io.nInputs = train.inputCount;

    emit datasetStarted(dsIdx, m_cfg.datasetPaths.size(), dsName,
        QString("%1 inputs, %2 train samples%3, %4 grammar rules")
            .arg(train.inputCount).arg(train.sampleCount)
            .arg(hasTest ? QString(", %1 test samples").arg(test->sampleCount) : QString(", no test file"))
            .arg(grammar.rules().size()));

    // Engine config
    ge::GEConfig ec;
    ec.genotype = m_cfg.genotype == "pge" ? ge::Genotype::Probabilistic : ge::Genotype::Codons;
    ec.popSize = m_cfg.pop;
    ec.generations = m_cfg.gens;
    ec.genomeLen = m_cfg.genomeLen;
    ec.maxWraps = m_cfg.maxWraps;
    ec.maxDepth = m_cfg.maxDepth;
    ec.replaceRate = m_cfg.replaceRate;
    ec.tournament = m_cfg.tournament;
    ec.mutationRate = m_cfg.mutationRate;
    ec.crossover = m_cfg.crossover == "local" ? ge::Crossover::Local : ge::Crossover::OnePoint;
    ec.localRate = m_cfg.localRate;
    ec.localNcr = m_cfg.localNcr;
    ec.pgeAlpha = m_cfg.pgeAlpha;
    ec.pgeElite = m_cfg.pgeElite;
    ec.pgeFloor = m_cfg.pgeFloor;
    ec.seed = m_cfg.seed ? m_cfg.seed + static_cast<quint32>(dsIdx) : makeSeed();

    // Search-phase training parameters (cheap, but with a short gradient polish so the
    // fitness reflects what a network can reach, not only how fast it starts).
    TrainParams sp;
    sp.opt = m_cfg.innerOpt; sp.pop = m_cfg.innerPop; sp.iters = m_cfg.innerIters;
    sp.baseBound = m_cfg.baseBound; sp.valSplit = m_cfg.valSplit;
    sp.polish = m_cfg.innerPolish; sp.polishEvals = m_cfg.innerPolishEvals;
    sp.boundMode = m_cfg.boundMode; sp.fanK = m_cfg.fanK;

    std::unordered_map<std::string, CacheEntry> cache;
    int evalCalls = 0, cacheHits = 0, refinements = 0;
    qint64 innerCalls = 0;
    const qint64 searchBudget = m_cfg.totalBudget > 0
        ? static_cast<qint64>(m_cfg.searchFrac * static_cast<double>(m_cfg.totalBudget)) : 0;

    auto fitnessOf = [&](const Outcome& oc) {
        double base;
        if (sp.valSplit > 0 && oc.valErr >= 0) {
            if (m_cfg.fitnessMetric == "error")
                // error %, with a 0.1 % train-error tie-break for when a small validation set saturates at 0
                base = oc.valErr + 0.001 * oc.trainErr;
            else
                base = oc.valLoss >= 0 ? oc.valLoss : oc.valErr;   // continuous: never saturates
        } else {
            base = oc.trainLoss;
        }
        const double f = base * (1.0 + m_cfg.parsimony * oc.params / 1000.0);
        return std::isfinite(f) ? f : kInf;
    };

    // Lamarckian resume: continue training a cached phenotype from its stored weights.
    // The new weights are kept only if the validation-based fitness improves (early stopping),
    // so a refinement can never make a phenotype look worse.
    auto refine = [&](CacheEntry& e) {
        if (m_cfg.lamarck <= 0 || e.refines >= m_cfg.lamarck || e.x.empty() || m_stop.load()) return;
        TrainParams rp = sp;
        rp.polish = true;
        rp.polishEvals = std::max(50, m_cfg.innerPolishEvals);
        Outcome oc = trainOne(e.spec, train, nullptr, rp, m_stop, &e.x);
        innerCalls += oc.calls;
        ++e.refines; ++refinements;
        if (oc.stopped) return;
        const double f = fitnessOf(oc);
        if (f < e.fitness) {
            e.fitness = f; e.x = oc.x;
            e.trainErr = oc.trainErr; e.valErr = oc.valErr; e.valLoss = oc.valLoss;
            if (oc.valLossSE >= 0) e.valLossSE = oc.valLossSE;
        }
    };

    auto fitnessFn = [&](const ge::Node& tree) -> double {
        if (m_stop.load()) return kInf;
        ge::NetSpec spec = ge::interpret(grammar, tree, io);
        if (!spec.valid()) return kInf;
        ++evalCalls;
        const std::string key = spec.key();
        auto it = cache.find(key);
        if (it != cache.end()) {
            ++cacheHits;
            // GE's many-to-one mapping makes selected phenotypes re-appear: every re-appearance
            // buys them more training, so the budget flows to the networks selection favours.
            if (searchBudget <= 0 || innerCalls < searchBudget) refine(it->second);
            return it->second.fitness;
        }
        Outcome oc = trainOne(spec, train, nullptr, sp, m_stop);
        innerCalls += oc.calls;
        if (oc.stopped) return kInf;           // do not cache a half-trained network
        CacheEntry e;
        e.spec = spec;
        e.x = oc.x;
        e.trainErr = oc.trainErr;
        e.valErr = oc.valErr;
        e.params = oc.params;
        e.inputs = oc.inputs;
        e.valLoss = oc.valLoss;
        e.valLossSE = std::max(0.0, oc.valLossSE);
        e.fitness = fitnessOf(oc);
        cache[key] = e;
        return e.fitness;
    };

    ge::GEEngine engine(grammar, ec, fitnessFn);
    if (ec.genotype == ge::Genotype::Probabilistic && m_cfg.transferPrior && !m_prior.empty())
        engine.setProbs(m_prior);

    const double t0 = nowSec();
    QVector<double> convergence;
    auto report = [&](int gen) {
        const double b = engine.best().fitness;
        convergence.append(std::isfinite(b) ? b : 100.0);
        QString desc;
        {
            ge::Node tr;
            if (engine.derive(engine.best(), tr))
                desc = QString::fromStdString(ge::interpret(grammar, tr, io).describe());
        }
        emit generationDone(gen, ec.generations, b, desc, engine.meanFitness(),
                            engine.diversity(), static_cast<int>(cache.size()), cacheHits, innerCalls);
    };

    engine.initPopulation();
    report(0);
    int gensRun = 0, stall = 0;
    double lastBest = engine.best().fitness;
    for (int gen = 1; gen <= ec.generations && !m_stop.load(); ++gen) {
        engine.step();
        report(gen);
        gensRun = gen;
        const double b = engine.best().fitness;
        double tol = 1e-9 * std::max(1.0, std::fabs(lastBest));
        if (m_cfg.adaptiveSearch) {
            // noise-aware: an improvement smaller than seRule standard errors of the best network's
            // validation loss is indistinguishable from noise and counts as a stalled generation;
            // the search then stops earlier and leaves more of T to the training of the final rows
            double bestF = kInf, se = 0;
            for (const auto& kv : cache) if (kv.second.fitness < bestF) { bestF = kv.second.fitness; se = kv.second.valLossSE; }
            tol = std::max(tol, m_cfg.seRule * se);
        }
        if (b < lastBest - tol) { lastBest = b; stall = 0; }
        else if (m_cfg.patience > 0 && ++stall >= m_cfg.patience) break;   // converged: more generations only repeat cached networks
        if (searchBudget > 0 && innerCalls >= searchBudget) break;         // fixed total budget: search share spent
    }
    // Two-stage selection: the best few phenotypes are refined to the full Lamarckian
    // depth, so the final ranking is not biased toward networks that merely train fast.
    {
        std::vector<CacheEntry*> fin;
        for (auto& kv : cache) fin.push_back(&kv.second);
        std::sort(fin.begin(), fin.end(), [](const CacheEntry* a, const CacheEntry* b) { return a->fitness < b->fitness; });
        const int nf = std::min<int>(std::max(0, m_cfg.finalists), static_cast<int>(fin.size()));
        for (int i = 0; i < nf && !m_stop.load(); ++i)
            while (fin[i]->refines < m_cfg.lamarck && !m_stop.load()) {
                const int before = fin[i]->refines;
                refine(*fin[i]);
                if (fin[i]->refines == before) break;
            }
    }
    const double searchSeconds = nowSec() - t0;
    if (ec.genotype == ge::Genotype::Probabilistic) m_prior = engine.probs();

    // Collect ranked phenotypes
    std::vector<const CacheEntry*> ranked;
    ranked.reserve(cache.size());
    for (const auto& kv : cache) ranked.push_back(&kv.second);
    std::sort(ranked.begin(), ranked.end(), [](const CacheEntry* a, const CacheEntry* b) {
        if (std::fabs(a->fitness - b->fitness) > 1e-9) return a->fitness < b->fitness;
        return a->params < b->params;
    });

    GEDatasetResult res;
    res.dsName = dsName;
    res.isCls = isCls;
    res.hasTest = hasTest;
    res.trainSamples = train.sampleCount;
    res.testSamples = hasTest ? test->sampleCount : 0;
    res.searchCalls = innerCalls;
    res.uniqueEvals = static_cast<int>(cache.size());
    res.cacheHits = cacheHits;
    res.searchSeconds = searchSeconds;
    res.convergence = convergence;
    res.gensRun = gensRun;
    res.gensPlanned = ec.generations;
    for (int i = 0; i < static_cast<int>(ranked.size()) && i < std::max(m_cfg.topK, 1) * 2; ++i) {
        const CacheEntry& e = *ranked[i];
        GEPhenotype p;
        p.rank = i + 1;
        p.desc = QString::fromStdString(e.spec.describe());
        p.fitness = e.fitness; p.trainErr = e.trainErr; p.valErr = e.valErr;
        p.params = e.params; p.inputs = e.inputs;
        res.top.append(p);
    }

    // Production statistics
    if (ec.genotype == ge::Genotype::Probabilistic) {
        res.probsReport = buildProbsReport(grammar, engine.probs(),
            "Learned production probabilities (PBIL update):");
    } else {
        std::vector<std::vector<double>> freq(grammar.rules().size());
        for (size_t i = 0; i < freq.size(); ++i) freq[i].assign(grammar.rules()[i].prods.size(), 0.0);
        const auto& pop = engine.population();
        const size_t nElite = std::max<size_t>(1, pop.size() / 4);
        for (size_t k = 0; k < nElite && k < pop.size(); ++k)
            for (const auto& c : pop[k].trace) freq[c.nt][c.prod] += 1.0;
        for (auto& f : freq) {
            double s = 0; for (double v : f) s += v;
            if (s > 0) for (double& v : f) v /= s;
        }
        res.probsReport = buildProbsReport(grammar, freq,
            "Production frequencies in the elite quarter of the final population:");
    }
    {
        ge::Node tr;
        if (engine.population().empty() || !engine.derive(engine.best(), tr)) res.bestProgram.clear();
        else res.bestProgram = QString::fromStdString(ge::Grammar::text(tr));
    }

    if (m_stop.load() || ranked.empty()) {
        emit datasetFinished(res);
        return false;
    }

    // ── Final comparison ─────────────────────────────────────────────────────
    TrainParams fp;
    fp.opt = m_cfg.innerOpt; fp.pop = m_cfg.finalPop; fp.iters = m_cfg.finalIters;
    fp.baseBound = m_cfg.baseBound; fp.valSplit = m_cfg.valSplit;
    fp.polish = m_cfg.finalPolish;
    fp.boundMode = m_cfg.boundMode; fp.fanK = m_cfg.fanK;

    // Training parameters that spend (about) `budget` loss evaluations.
    // With a fixed total budget the training is hybrid for EVERY row alike: a global phase
    // (inner optimizer) followed by bound-respecting L-BFGS that gets polishFrac of the budget.
    // The search ranks phenotypes with the same kind of training, so selection and final
    // training agree.
    auto withBudget = [&](qint64 budget) {
        TrainParams t = fp;
        if (m_cfg.totalBudget > 0 && m_cfg.polishFrac > 0) {
            t.polish = true;
            t.polishEvals = static_cast<int>(std::max<qint64>(t.polishEvals,
                                static_cast<qint64>(m_cfg.polishFrac * static_cast<double>(budget))));
        }
        const qint64 pol = t.polish ? t.polishEvals : 0;
        t.iters = static_cast<int>(std::max<qint64>(1, std::min<qint64>(100000, (budget - pol + t.pop - 1) / t.pop)));
        return t;
    };

    const bool fixedT = m_cfg.totalBudget > 0;
    const qint64 T = m_cfg.totalBudget;
    const qint64 rest = fixedT ? std::max<qint64>(static_cast<qint64>(fp.pop) * 10, T - innerCalls) : 0;

    const int K = std::min<int>(std::max(1, m_cfg.topK), static_cast<int>(ranked.size()));
    auto want = [&](const QString& id) { return m_cfg.methods.isEmpty() || m_cfg.methods.contains(id); };
    // reference models asked for explicitly next to the GE rows (Single run page); the batch engine
    // runs them as configuration "baselines" instead and never lists them here
    QStringList refIds;
    for (const QString& id : geBaselineIds()) if (m_cfg.methods.contains(id)) refIds << id;
    int totalRows = 0;
    {
        const QStringList ids = fixedT
            ? QStringList{"standard", "standard_T", "grid_T", "ge1", "ge1_T", "ge1_T_nobounds", "ge1_T_nofeat"}
            : QStringList{"standard", "standard_eq", "grid_eq", "ge1", "ge1_nobounds", "ge1_eq", "ge1_nobounds_eq"};
        for (const QString& id : ids) {
            if (!fixedT && (id == "standard_eq" || id == "ge1_eq" || id == "ge1_nobounds_eq") && !m_cfg.equalBudgetBaseline) continue;
            if (!fixedT && id == "grid_eq" && !m_cfg.gridBaseline) continue;
            if (want(id)) ++totalRows;
        }
        for (int k = 2; k <= K; ++k) if (want(QString("ge%1").arg(k))) ++totalRows;
        if (fixedT) for (const char* id : {"standard_T_ad", "std_ens_T", "ge1_T_ad", "ens_T", "ens_T_nofeat", "ens_T_nobounds",
                                           "ens_T_div", "ens_T_cv", "std_ens_T_cv",
                                           "sage", "sage_noemb", "sage_nofeat", "sage_nobounds", "sage_one", "sage_std"})
            if (m_cfg.methods.contains(id)) ++totalRows;
        totalRows += static_cast<int>(refIds.size());
        totalRows = std::max(1, totalRows);
    }
    emit finalStageStarted(totalRows);
    int rowIdx = 0;

    auto addRow = [&](const QString& id, const QString& label, const ge::NetSpec& spec, const TrainParams& tp,
                      bool isGE, std::shared_ptr<NeuralNetwork>* keepNet,
                      std::shared_ptr<Dataset>* keepDs) {
        const double r0 = nowSec();
        Outcome oc = trainOne(spec, train, test, tp, m_stop);
        GEFinalRow row;
        row.id = id;
        row.label = label;
        row.trainErr = oc.trainErr;
        row.testErr = oc.testErr;
        row.valErr = oc.valErr;
        row.params = oc.params;
        row.evals = oc.calls;
        row.seconds = nowSec() - r0;
        row.isGE = isGE;
        res.rows.append(row);
        if (keepNet) { *keepNet = oc.net; }
        if (keepDs)  { *keepDs = oc.trainDs; }
        emit finalRowDone(++rowIdx, totalRows, label);
    };
    // An ablation whose phenotype is identical to the full one (e.g. no features to remove)
    // repeats the full row instead of adding retraining noise to a paired comparison.
    auto copyRow = [&](const QString& fromId, const QString& id, const QString& label) {
        for (int r = res.rows.size() - 1; r >= 0; --r)
            if (res.rows[r].id == fromId) {
                GEFinalRow row = res.rows[r];
                row.id = id; row.label = label + " (identical phenotype)";
                res.rows.append(row);
                emit finalRowDone(++rowIdx, totalRows, label);
                return true;
            }
        return false;
    };
    auto stopNow = [&]() { if (m_stop.load()) { emit datasetFinished(res); return true; } return false; };

    // Baselines
    ge::NetSpec std0;
    for (int s : parseHidden(m_cfg.baselineHidden)) {
        ge::LayerSpec l; l.size = s; l.act = m_cfg.baselineAct.toStdString();
        l.lbCmd = l.ubCmd = "NOTHING";
        std0.hidden.push_back(l);
    }
    std0.outLbCmd = std0.outUbCmd = "NOTHING";
    const QString stdName = QString("Standard [%1 %2]").arg(m_cfg.baselineHidden, m_cfg.baselineAct);

    // Grid search over classic architectures with the same search budget and the same
    // inner training/fitness as the GE search (restarts until the budget is spent).
    auto gridSearch = [&](qint64& gridCalls) {
        struct Cand { ge::NetSpec spec; double best = kInf; };
        std::vector<Cand> grid;
        auto mk = [&](std::initializer_list<int> sz, const char* act) {
            ge::NetSpec sp0;
            for (int z : sz) { ge::LayerSpec l; l.size = z; l.act = act; l.lbCmd = l.ubCmd = "NOTHING"; sp0.hidden.push_back(l); }
            sp0.outLbCmd = sp0.outUbCmd = "NOTHING";
            Cand c; c.spec = sp0; grid.push_back(c);
        };
        for (int z : {4, 8, 16, 32}) for (const char* a : {"relu", "tanh", "sigmoid"}) mk({z}, a);
        mk({16, 8}, "relu"); mk({32, 16}, "relu"); mk({8, 4}, "tanh"); mk({16, 8}, "tanh");
        gridCalls = 0;
        int rounds = 0;
        bool budgetDone = false;
        while (!budgetDone && !m_stop.load() && rounds < 50) {
            for (auto& c : grid) {
                if (m_stop.load()) break;
                if (rounds > 0 && gridCalls >= innerCalls) { budgetDone = true; break; }
                Outcome oc = trainOne(c.spec, train, nullptr, sp, m_stop);
                gridCalls += oc.calls;
                if (oc.stopped) break;
                c.best = std::min(c.best, fitnessOf(oc));
            }
            ++rounds;
            if (gridCalls >= innerCalls) budgetDone = true;
        }
        size_t bi = 0;
        for (size_t k = 1; k < grid.size(); ++k) if (grid[k].best < grid[bi].best) bi = k;
        return grid[bi].spec;
    };

    std::shared_ptr<NeuralNetwork> bestNet;
    std::shared_ptr<Dataset>       bestDs;
    const ge::NetSpec& g1 = ranked[0]->spec;

    if (fixedT) {
        // Fixed total budget T per run: every "_T" row consumes about T loss evaluations
        // in total (search + training). This is the cost-fair comparison.
        const TrainParams tAll = withBudget(T), tRest = withBudget(rest);
        if (want("standard")) addRow("standard", stdName + " (final budget)", std0, fp, false, nullptr, nullptr);
        if (stopNow()) return false;
        if (want("standard_T")) addRow("standard_T", QString("Standard, total budget T=%1").arg(T), std0, tAll, false, nullptr, nullptr);
        if (stopNow()) return false;
        if (want("grid_T")) {
            qint64 gc = 0;
            const ge::NetSpec gs = gridSearch(gc);
            if (stopNow()) return false;
            addRow("grid_T", QString("Grid search (%1 evals) + training with the rest of T: %2")
                       .arg(gc).arg(QString::fromStdString(gs.describe())), gs, withBudget(std::max<qint64>(fp.pop * 10, T - gc)),
                   false, nullptr, nullptr);
            if (stopNow()) return false;
        }
        if (want("ge1")) addRow("ge1", "GE #1 (final budget)", g1, fp, true, nullptr, nullptr);
        for (int k = 1; k < K && !m_stop.load(); ++k)
            if (want(QString("ge%1").arg(k + 1)))
                addRow(QString("ge%1").arg(k + 1), QString("GE #%1 (final budget)").arg(k + 1), ranked[k]->spec, fp, true, nullptr, nullptr);
        if (stopNow()) return false;
        if (want("ge1_T"))
            addRow("ge1_T", QString("GE #1, search %1 + training %2 = T").arg(innerCalls).arg(rest), g1, tRest, true, &bestNet, &bestDs);
        if (stopNow()) return false;
        if (want("ge1_T_nobounds")) {
            const ge::NetSpec nb = g1.withoutBounds();
            if (!(nb.key() == g1.key() && copyRow("ge1_T", "ge1_T_nobounds", "GE #1 T without bound commands")))
                addRow("ge1_T_nobounds", "GE #1 T without bound commands (ablation)", nb, tRest, true, nullptr, nullptr);
        }
        if (stopNow()) return false;
        if (want("ge1_T_nofeat")) {
            const ge::NetSpec nf = g1.withoutFeatures();
            if (!(nf.key() == g1.key() && copyRow("ge1_T", "ge1_T_nofeat", "GE #1 T without features")))
                addRow("ge1_T_nofeat", "GE #1 T without constructed features (ablation)", nf, tRest, true, nullptr, nullptr);
        }
        if (stopNow()) return false;

        // ── Self-adaptive rows (only when asked for explicitly) ─────────────────
        auto wantAd = [&](const char* id) { return m_cfg.methods.contains(id); };
        const bool anyAd = wantAd("standard_T_ad") || wantAd("std_ens_T") || wantAd("ge1_T_ad") ||
                           wantAd("ens_T") || wantAd("ens_T_nofeat") || wantAd("ens_T_nobounds") ||
                           wantAd("ens_T_div") || wantAd("ens_T_cv") || wantAd("std_ens_T_cv") ||
                           wantAd("sage") || wantAd("sage_noemb") || wantAd("sage_nofeat") || wantAd("sage_nobounds") ||
                           wantAd("sage_one") || wantAd("sage_std");
        if (anyAd) {
            // Candidate pool: the best archive phenotypes with their search-time weights, scored per
            // validation sample (the split of the search), so differences can be compared with their noise.
            struct PoolM { const CacheEntry* e; std::vector<std::vector<double>> p, t; std::vector<double> loss; double mean = kInf; };
            std::vector<PoolM> pool;
            int kind = 0;
            {
                TrainParams ev = sp; ev.evalOnly = true; ev.wantPreds = true;
                const int P = std::min<int>(std::max(1, m_cfg.ensPool), static_cast<int>(ranked.size()));
                for (int i = 0; i < P && !m_stop.load(); ++i) {
                    if (ranked[i]->x.empty()) continue;
                    Outcome oc = trainOne(ranked[i]->spec, train, nullptr, ev, m_stop, &ranked[i]->x);
                    if (oc.valPred.empty()) continue;
                    PoolM pm; pm.e = ranked[i]; pm.p = std::move(oc.valPred); pm.t = std::move(oc.valTarget);
                    kind = oc.kind;
                    pm.loss.resize(pm.p.size());
                    for (size_t r = 0; r < pm.p.size(); ++r) pm.loss[r] = predLossK(kind, pm.p[r], pm.t[r]);
                    double se = 0; meanSE(pm.loss, pm.mean, se);
                    pool.push_back(std::move(pm));
                }
            }
            if (stopNow()) return false;
            int bi = 0;
            for (int c = 1; c < static_cast<int>(pool.size()); ++c) if (pool[c].mean < pool[bi].mean) bi = c;

            // Noise-aware selection of ONE network: every candidate whose validation loss is within seRule
            // standard errors of the best (paired over the same samples) is as good as the best;
            // the simplest of them (fewest parameters) is chosen.
            ge::NetSpec gAd = g1;
            if (!pool.empty()) {
                int ch = bi;
                for (int c = 0; c < static_cast<int>(pool.size()); ++c) {
                    if (c == bi) continue;
                    std::vector<double> d(pool[c].loss.size());
                    for (size_t r = 0; r < d.size(); ++r) d[r] = pool[c].loss[r] - pool[bi].loss[r];
                    double m = 0, se = 0; meanSE(d, m, se);
                    if (m <= m_cfg.seRule * se && pool[c].e->params < pool[ch].e->params) ch = c;
                }
                gAd = pool[ch].e->spec;
            }

            // Greedy ensemble selection with replacement (Caruana et al. 2004): the candidate with the largest
            // validation gain is added while it is not significantly harmful (see below), up to ensMax picks.
            // diverse = true: among the candidates whose gain is within seRule standard errors of the best gain
            // (indistinguishable on this validation split), a network NOT yet in the ensemble is preferred, so
            // the ensemble does not stall on two networks that the small split happens to favour.
            auto greedy = [&](bool diverse) {
                std::vector<int> picks;
                if (pool.empty()) return picks;
                picks.push_back(bi);
                const size_t n = pool[bi].p.size();
                std::vector<std::vector<double>> sum = pool[bi].p;
                std::vector<double> cur = pool[bi].loss;
                while (static_cast<int>(picks.size()) < std::max(1, m_cfg.ensMax)) {
                    const double k = static_cast<double>(picks.size());
                    std::set<int> used(picks.begin(), picks.end());
                    std::vector<double> gm(pool.size()), gs(pool.size());
                    std::vector<std::vector<double>> gl(pool.size());
                    int bc = -1;
                    for (int c = 0; c < static_cast<int>(pool.size()); ++c) {
                        std::vector<double> nl(n), d(n);
                        for (size_t r = 0; r < n; ++r) {
                            std::vector<double> q = sum[r];
                            for (size_t j = 0; j < q.size(); ++j) q[j] = (q[j] + pool[c].p[r][j]) / (k + 1.0);
                            nl[r] = predLossK(kind, q, pool[c].t[r]);
                            d[r] = cur[r] - nl[r];
                        }
                        double m = 0, se = 0; meanSE(d, m, se);
                        gm[c] = m; gs[c] = se; gl[c] = std::move(nl);
                        // larger gain wins; on an equal gain a network not yet in the ensemble (diversity)
                        const bool tie = bc >= 0 && std::fabs(m - gm[bc]) <= 1e-12 * std::max(1.0, std::fabs(gm[bc]));
                        if (bc < 0 || (m > gm[bc] && !tie) || (tie && !used.count(c) && used.count(bc))) bc = c;
                    }
                    if (diverse && used.count(bc)) {
                        // paired comparison of the two resulting ensembles over the validation samples:
                        // an unused candidate is as good as the best when its ensemble loss is not larger
                        // by more than seRule standard errors of the paired difference
                        int alt = -1;
                        for (int c = 0; c < static_cast<int>(pool.size()); ++c) {
                            if (used.count(c)) continue;
                            std::vector<double> dd(n);
                            for (size_t r = 0; r < n; ++r) dd[r] = gl[c][r] - gl[bc][r];
                            double dm = 0, dse = 0; meanSE(dd, dm, dse);
                            if (dm <= m_cfg.seRule * dse && (alt < 0 || gm[c] > gm[alt])) alt = c;
                        }
                        if (alt >= 0) bc = alt;
                    }
                    if (adaptDebug() >= 1)
                        std::fprintf(stderr, "[ens%s] size %zu: candidate %d gain %.4g se %.4g\n", diverse ? "-div" : "", picks.size(), bc, gm[bc], gs[bc]);
                    // non-inferiority: a member is added unless it makes the validation loss worse by more than
                    // seRule standard errors; averaging lowers the variance, which the validation split of the
                    // search (on which the members were selected) cannot show by itself
                    if (gm[bc] < -m_cfg.seRule * gs[bc]) break;
                    picks.push_back(bc);
                    for (size_t r = 0; r < n; ++r) for (size_t j = 0; j < sum[r].size(); ++j) sum[r][j] += pool[bc].p[r][j];
                    cur = gl[bc];
                }
                return picks;
            };
            const std::vector<int> picks = greedy(false);
            const std::vector<int> picksDiv = (wantAd("ens_T_div") || wantAd("ens_T_cv") || wantAd("std_ens_T_cv"))
                                              ? greedy(true) : std::vector<int>();
            if (adaptDebug() >= 1) {
                std::fprintf(stderr, "[pool] %zu candidates, best %d mean %.4g; picks:", pool.size(), bi, pool.empty() ? 0.0 : pool[bi].mean);
                for (int c : picks) std::fprintf(stderr, " %d", c);
                std::fprintf(stderr, "; diverse:");
                for (int c : picksDiv) std::fprintf(stderr, " %d", c);
                std::fprintf(stderr, "\n");
            }
            // distinct members with their weights (number of picks)
            std::vector<std::pair<ge::NetSpec, int>> members;
            {
                std::map<int, int> w;
                for (int c : picks) ++w[c];
                for (const auto& kv : w) members.push_back({pool[kv.first].e->spec, kv.second});
            }
            if (members.empty()) members.push_back({g1, 1});
            std::vector<std::pair<ge::NetSpec, int>> membersDiv;
            {
                std::map<int, int> w;
                for (int c : picksDiv) ++w[c];
                for (const auto& kv : w) membersDiv.push_back({pool[kv.first].e->spec, kv.second});
            }
            if (membersDiv.empty()) membersDiv = members;

            // Train the members (self-adaptive training, budget shared equally) and average their predictions.
            int ensRowIdx = 0;
            // cvK > 1: every member is a cross-validation committee of cvK networks, each validated (early
            // stopping) on its own fold of one cvK-fold partition, so all rows are used for training
            auto ensembleRow = [&](const QString& id, const QString& label, const std::vector<std::pair<ge::NetSpec, int>>& mem,
                                   qint64 budget, bool isGE, int cvK = 0, int seedIdx = -1) {
                // seedIdx >= 0: fixed data splits, so that rows sharing it (the final method and its
                // ablations) differ only in what is removed (paired design)
                const unsigned rowSeed = static_cast<unsigned>(seedIdx >= 0 ? seedIdx : ensRowIdx);
                const double r0 = nowSec();
                const int folds = cvK > 1 ? cvK : 1;
                const qint64 per = std::max<qint64>(fp.pop * 10, budget / std::max<qint64>(1, static_cast<qint64>(mem.size()) * folds));
                std::vector<std::vector<double>> tp, tt, rp, rt;
                double wsum = 0; int params = 0; qint64 calls = 0; int kd = 0;
                for (size_t jf = 0; jf < mem.size() * folds && !m_stop.load(); ++jf) {
                    const size_t j = jf / folds;
                    TrainParams ap = fp;
                    ap.adaptive = true; ap.budget = per;
                    ap.splitSeed = 1000u + 7919u * static_cast<unsigned>(j) + 31u * ec.seed + rowSeed;
                    if (folds > 1) { ap.cvK = folds; ap.cvFold = static_cast<int>(jf % folds); }
                    Outcome oc = trainOne(mem[j].first, train, test, ap, m_stop);
                    if (oc.stopped) return;
                    const double w = mem[j].second;
                    kd = oc.kind;
                    auto acc = [&](std::vector<std::vector<double>>& S, const std::vector<std::vector<double>>& P) {
                        if (S.empty()) { S = P; for (auto& v : S) for (double& x : v) x *= w; return; }
                        for (size_t r = 0; r < S.size() && r < P.size(); ++r) for (size_t q = 0; q < S[r].size(); ++q) S[r][q] += w * P[r][q];
                    };
                    acc(tp, oc.testPred); if (tt.empty()) tt = oc.testTarget;
                    acc(rp, oc.trainPred); if (rt.empty()) rt = oc.trainTarget;
                    wsum += w; params += oc.params; calls += oc.calls;
                }
                if (m_stop.load() || wsum <= 0) return;
                for (auto* S : {&tp, &rp}) for (auto& v : *S) for (double& x : v) x /= wsum;
                GEFinalRow row;
                row.id = id; row.label = label;
                row.trainErr = predError(kd, rp, rt);
                row.testErr = tp.empty() ? -1 : predError(kd, tp, tt);
                row.params = params; row.evals = calls;
                row.seconds = nowSec() - r0; row.isGE = isGE;
                res.rows.append(row);
                ++ensRowIdx;
                emit finalRowDone(++rowIdx, totalRows, label);
            };
            int distinct = static_cast<int>(members.size());
            QString memDesc = QString("%1 picks, %2 distinct").arg(picks.size()).arg(distinct);
            if (wantAd("standard_T_ad"))
                ensembleRow("standard_T_ad", QString("Standard, total budget T, self-adaptive training"), {{std0, 1}}, T, false);
            if (stopNow()) return false;
            if (wantAd("std_ens_T")) {
                std::vector<std::pair<ge::NetSpec, int>> sm;
                for (int j = 0; j < distinct; ++j) sm.push_back({std0, 1});
                ensembleRow("std_ens_T", QString("Ensemble of %1 standard networks, total budget T").arg(distinct), sm, T, false);
            }
            if (stopNow()) return false;
            if (wantAd("ge1_T_ad"))
                ensembleRow("ge1_T_ad", QString("GE #1 adaptive (noise-aware selection): %1").arg(QString::fromStdString(gAd.describe())),
                            {{gAd, 1}}, rest, true);
            if (stopNow()) return false;
            if (wantAd("ens_T"))
                ensembleRow("ens_T", QString("GE ensemble (%1), search %2 + training %3 = T").arg(memDesc).arg(innerCalls).arg(rest),
                            members, rest, true);
            if (stopNow()) return false;
            auto ablation = [&](const char* id, const QString& label, bool feats) {
                std::vector<std::pair<ge::NetSpec, int>> am;
                bool same = true;
                for (const auto& m : members) {
                    ge::NetSpec x = feats ? m.first.withoutFeatures() : m.first.withoutBounds();
                    same = same && x.key() == m.first.key();
                    am.push_back({x, m.second});
                }
                if (same && copyRow("ens_T", id, label)) return;
                ensembleRow(id, label, am, rest, true);
            };
            if (wantAd("ens_T_nofeat")) ablation("ens_T_nofeat", "GE ensemble T without constructed features (ablation)", true);
            if (stopNow()) return false;
            if (wantAd("ens_T_nobounds")) ablation("ens_T_nobounds", "GE ensemble T without bound commands (ablation)", false);
            if (stopNow()) return false;
            // development variants (classification): diversity-aware selection, cross-validation committees
            const int distinctDiv = static_cast<int>(membersDiv.size());
            if (wantAd("ens_T_div"))
                ensembleRow("ens_T_div", QString("GE ensemble, diverse selection (%1 distinct)").arg(distinctDiv), membersDiv, rest, true);
            if (stopNow()) return false;
            if (wantAd("ens_T_cv"))
                ensembleRow("ens_T_cv", QString("GE ensemble, diverse selection, 5-fold CV committees (%1 distinct)").arg(distinctDiv),
                            membersDiv, rest, true, 5);
            if (stopNow()) return false;
            if (wantAd("std_ens_T_cv")) {
                std::vector<std::pair<ge::NetSpec, int>> sm;
                for (int j = 0; j < distinctDiv; ++j) sm.push_back({std0, 1});
                ensembleRow("std_ens_T_cv", QString("Ensemble of %1 standard networks, 5-fold CV committees, T").arg(distinctDiv), sm, T, false, 5);
            }
            // ── SAGE, the final method (frozen after v3_dev) and its ablations ──
            // Diverse noise-aware selection, every member a 5-fold CV committee; all rows below use the
            // same data splits (seedIdx kSageSeed), so each ablation differs from SAGE only in what it removes.
            constexpr int kSageSeed = 100;
            const QString sageDesc = QString("%1 distinct members x 5-fold CV committees").arg(distinctDiv);
            if (stopNow()) return false;
            if (wantAd("sage"))
                ensembleRow("sage", QString("SAGE: %1, search %2 + training %3 = T").arg(sageDesc).arg(innerCalls).arg(rest),
                            membersDiv, rest, true, 5, kSageSeed);
            auto sageAblation = [&](const char* id, const QString& label, ge::NetSpec (ge::NetSpec::*strip)() const) {
                std::vector<std::pair<ge::NetSpec, int>> am;
                bool same = true;
                for (const auto& m : membersDiv) {
                    ge::NetSpec x = (m.first.*strip)();
                    same = same && x.key() == m.first.key();
                    am.push_back({x, m.second});
                }
                if (same && copyRow("sage", id, label)) return;
                ensembleRow(id, label, am, rest, true, 5, kSageSeed);
            };
            if (stopNow()) return false;
            if (wantAd("sage_noemb")) sageAblation("sage_noemb", "SAGE without the periodic input embedding (ablation)", &ge::NetSpec::withoutEmbedding);
            if (stopNow()) return false;
            if (wantAd("sage_nofeat")) sageAblation("sage_nofeat", "SAGE without constructed features (ablation)", &ge::NetSpec::withoutFeatures);
            if (stopNow()) return false;
            if (wantAd("sage_nobounds")) sageAblation("sage_nobounds", "SAGE without bound commands (ablation)", &ge::NetSpec::withoutBounds);
            if (stopNow()) return false;
            if (wantAd("sage_one")) {
                // the best member alone (candidate 0 of the pool is always the first greedy pick)
                if (membersDiv.size() == 1 && copyRow("sage", "sage_one", "SAGE with one member (ablation)")) {}
                else ensembleRow("sage_one", "SAGE with one member, 5-fold CV committee (ablation)",
                                 {{membersDiv.front().first, 1}}, rest, true, 5, kSageSeed);
            }
            if (stopNow()) return false;
            if (wantAd("sage_std")) {
                std::vector<std::pair<ge::NetSpec, int>> sm;
                for (int j = 0; j < distinctDiv; ++j) sm.push_back({std0, 1});
                ensembleRow("sage_std", QString("SAGE with %1 standard networks instead of the grammar, budget T (ablation)").arg(distinctDiv),
                            sm, T, false, 5, kSageSeed);
            }
        }
    } else {
        if (want("standard")) addRow("standard", stdName, std0, fp, false, nullptr, nullptr);
        if (stopNow()) return false;
        TrainParams eq = withBudget(innerCalls + static_cast<qint64>(fp.pop) * fp.iters);
        const qint64 eqBudget = innerCalls + static_cast<qint64>(fp.pop) * fp.iters;
        if (m_cfg.equalBudgetBaseline && want("standard_eq")) {
            addRow("standard_eq", QString("Standard, equal total budget (%1 evals)").arg(eqBudget), std0, eq, false, nullptr, nullptr);
            if (stopNow()) return false;
        }
        if (m_cfg.gridBaseline && want("grid_eq")) {
            qint64 gc = 0;
            const ge::NetSpec gs = gridSearch(gc);
            if (stopNow()) return false;
            addRow("grid_eq", QString("Grid search, equal search budget (%1 evals): %2")
                       .arg(gc).arg(QString::fromStdString(gs.describe())), gs, fp, false, nullptr, nullptr);
            if (stopNow()) return false;
        }
        for (int k = 0; k < K && !m_stop.load(); ++k)
            if (want(QString("ge%1").arg(k + 1)))
                addRow(QString("ge%1").arg(k + 1), QString("GE #%1 (arch + bounds%2)").arg(k + 1)
                           .arg(ranked[k]->spec.features.empty() ? QString() : QString(" + features")),
                       ranked[k]->spec, fp, true, k == 0 ? &bestNet : nullptr, k == 0 ? &bestDs : nullptr);
        if (!m_stop.load() && want("ge1_nobounds"))
            addRow("ge1_nobounds", "GE #1 without bound commands (ablation)", g1.withoutBounds(), fp, true, nullptr, nullptr);
        if (m_cfg.equalBudgetBaseline && !m_stop.load()) {
            if (want("ge1_eq"))
                addRow("ge1_eq", QString("GE #1, retrained with equal total budget (%1 evals)").arg(eqBudget), g1, eq, true, nullptr, nullptr);
            if (!m_stop.load() && want("ge1_nobounds_eq"))
                addRow("ge1_nobounds_eq", "GE #1 without bound commands, equal total budget", g1.withoutBounds(), eq, true, nullptr, nullptr);
        }
    }

    if (!refIds.isEmpty() && !m_stop.load()) {
        BaselineConfig bc;
        bc.seed = m_cfg.seed ? m_cfg.seed + static_cast<quint32>(dsIdx) : makeSeed();
        const int before = rowIdx;
        const QVector<BaselineRow> rows = geRunBaselines(train, test, refIds, bc, m_stop,
            [&](int done, int, const QString& id) {
                if (done > 0) emit finalRowDone(before + done, totalRows, id);
            });
        for (const BaselineRow& b : rows) { res.rows.append(baselineFinalRow(b)); ++rowIdx; }
    }

    // Only a net without constructed features can be saved/exported by the rest
    // of the application (it would need the feature pre-processing otherwise).
    if (bestNet && g1.features.empty() && g1.embSigma <= 0) {
        m_lastNet = bestNet;
        m_lastDs = std::make_shared<Dataset>(train);
    }
    emit datasetFinished(res);
    return !m_stop.load();
}

} // namespace NeuralStudio
