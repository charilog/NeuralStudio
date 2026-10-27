#include "ArchSearchWorker.h"
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
#include <stdexcept>
#include <random>
#include <algorithm>
#include <cmath>
#include <functional>
#include <unordered_map>
#include <numeric>
#include <QFileInfo>
#include <chrono>
#include <atomic>

namespace NeuralStudio {

// ── Shared seed (same pattern as other workers) ───────────────────────────────
static unsigned int makeSeed() {
    static std::atomic<uint64_t> ctr{0};
    const auto t = static_cast<uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    return static_cast<unsigned int>(t ^ (++ctr * 6364136223846793005ULL));
}

// ── Optimizer factory ─────────────────────────────────────────────────────────
static std::unique_ptr<optimsolution::Optimizer> makeOpt(const QString& n) {
    if(n=="lbfgs"){auto p=std::make_unique<optimsolution::LBFGS>();p->setMemory(7);return p;}
    if(n=="bfgs") {auto p=std::make_unique<optimsolution::LBFGS>();p->setMemory(20);return p;}
    if(n=="sa")         return std::make_unique<optimsolution::SimulatedAnnealing>();
    if(n=="neldermead") return std::make_unique<optimsolution::NelderMead>();
    if(n=="de")         return std::make_unique<optimsolution::DE>();
    if(n=="pso")        return std::make_unique<optimsolution::PSO>();
    if(n=="cmaes")      return std::make_unique<optimsolution::CMAES>();
    if(n=="jso")        return std::make_unique<optimsolution::JSO>();
    if(n=="lmcmaes")    return std::make_unique<optimsolution::LMCMAES>();
    if(n=="acor")       return std::make_unique<optimsolution::ACOR>();
    if(n=="clpso")      return std::make_unique<optimsolution::CLPSO>();
    if(n=="mlshaderl")  return std::make_unique<optimsolution::mLSHADE_RL>();
    if(n=="arq3")       return std::make_unique<optimsolution::ARQ3>();
    return std::make_unique<optimsolution::GeneticAlgorithm>();
}

// ── Parse helpers ─────────────────────────────────────────────────────────────
static Activation parseAct(const QString& s) {
    if(s=="sigmoid") return Activation::Sigmoid;
    if(s=="tanh")    return Activation::Tanh;
    if(s=="linear")  return Activation::Linear;
    return Activation::ReLU;
}

static std::shared_ptr<NeuralNetwork> buildNet(const Dataset* ds,
                                                const std::vector<int>& h,
                                                const std::vector<Activation>& acts) {
    NetworkConfig cfg;
    cfg.task = detectTask(ds);
    cfg.hiddenSizes = h;
    cfg.hiddenActivations = acts;
    auto uv = ds->uniqueOutputValues(0);
    // MultiClassClassification needs one output per class, not per output column
    // Output size: multiclass = one node per class; binary = single node
    // (same rule as BatchRunner); regression = dataset output columns.
    const int netOut = (cfg.task == TaskType::MultiClassClassification)
                       ? static_cast<int>(uv.size())
                       : (cfg.task == TaskType::BinaryClassification)
                       ? 1 : ds->outputCount;
    auto net = std::make_shared<NeuralNetwork>();
    net->build(ds->inputCount, netOut, cfg);
    net->setNormalization(ds->inputStats);
    net->fitOutputScaling(ds);   // regression: standardised output layer
    if (cfg.task != TaskType::Regression) {
        net->setClassValues(uv);
        if (!uv.empty()) net->setOutputMapping(uv.front(), uv.back());
    } else if (!ds->outputStats.empty()) {
        net->setOutputMapping(ds->outputStats[0].min, ds->outputStats[0].max);
    }
    return net;
}

// ── ArchProblem ───────────────────────────────────────────────────────────────
// Implements optimsolution::Problem.
// Variables: [numLayers, size0, size1, ..., act0, act1, ...]
// Each evaluate() decodes x → architecture → (optional cache) → train → val error.
// ─────────────────────────────────────────────────────────────────────────────
class ArchProblem : public optimsolution::Problem {
public:
    struct Config {
        int minLayers = 1;
        int maxLayers = 4;
        std::vector<int>        candidateSizes;
        std::vector<Activation> candidateActs;
        std::vector<std::string> actNames;  // parallel to candidateActs
        std::string  innerOpt;
        int          innerPop;
        int          innerIters;
        double       innerWBound;
        double       valSplit;
        int          innerPolishEvals = 0;
        const Dataset* testDs = nullptr;   // companion test file: scored only, never selects anything
    };

    struct ArchEntry {
        std::vector<int>         sizes;
        std::vector<Activation>  acts;
        std::vector<std::string> actNamesList;  // parallel to acts, stored at evaluation time
        std::string  descStr;
        double       trainErr   = 0;
        double       valErr     = 0;
        double       testErr    = -1;
        double       trainLoss  = 0;
        int          numParams  = 0;
        int          evalNum    = 0;
        double       fitness    = 0;
    };

    using ProgressCb = std::function<void(int /*evalNum*/, int /*totalEvals*/,
                                          double /*fitness*/,
                                          const std::string& /*desc*/,
                                          bool /*cacheHit*/)>;

    ArchProblem(const Config& cfg, const Dataset* ds,
                int totalEvals,
                std::atomic<bool>& stop,
                ProgressCb progressCb)
        : m_cfg(cfg), m_ds(ds)
        , m_totalEvals(totalEvals)
        , m_stop(stop)
        , m_progressCb(std::move(progressCb))
    {
        // Dimension: 1 (numLayers) + maxLayers (sizes) + maxLayers (acts)
        m_dim = 1 + 2 * m_cfg.maxLayers;

        m_lb.resize(m_dim);
        m_ub.resize(m_dim);

        // x[0]: numLayers ∈ [minLayers, maxLayers]  (decoded via round + clamp)
        m_lb[0] = m_cfg.minLayers - 0.4999;
        m_ub[0] = m_cfg.maxLayers + 0.4999;

        // x[1..maxLayers]: size index ∈ [0, nSizes-1]
        const int nS = static_cast<int>(m_cfg.candidateSizes.size());
        for (int i = 1; i <= m_cfg.maxLayers; ++i) {
            m_lb[i] = -0.4999;
            m_ub[i] =  nS - 0.5001;
        }

        // x[maxLayers+1..2*maxLayers]: activation index ∈ [0, nActs-1]
        const int nA = static_cast<int>(m_cfg.candidateActs.size());
        for (int i = m_cfg.maxLayers + 1; i < m_dim; ++i) {
            m_lb[i] = -0.4999;
            m_ub[i] =  nA - 0.5001;
        }
    }

    // ── optimsolution::Problem interface ─────────────────────────────────────
    int dimension() const override { return m_dim; }
    const std::vector<double>& lb() const override { return m_lb; }
    const std::vector<double>& ub() const override { return m_ub; }
    int calls()  const override { return m_totalCalls; }

    double evaluate(const std::vector<double>& x) override {
        ++m_totalCalls;

        // Decode x → (sizes, acts)
        const int nSizes = static_cast<int>(m_cfg.candidateSizes.size());
        const int nActs  = static_cast<int>(m_cfg.candidateActs.size());

        int numLayers = std::max(m_cfg.minLayers,
                        std::min(m_cfg.maxLayers,
                                 static_cast<int>(std::round(x[0]))));

        std::vector<int>        sizes(numLayers);
        std::vector<Activation> acts (numLayers);
        for (int l = 0; l < numLayers; ++l) {
            int si = std::max(0, std::min(nSizes-1, static_cast<int>(x[1+l]+0.5)));
            int ai = std::max(0, std::min(nActs-1,
                        static_cast<int>(x[m_cfg.maxLayers + 1 + l] + 0.5)));
            sizes[l] = m_cfg.candidateSizes[si];
            acts [l] = m_cfg.candidateActs [ai];
        }

        const std::string key = makeKey(numLayers, sizes, acts);

        // Check cache
        auto it = m_cache.find(key);
        if (it != m_cache.end()) {
            ++m_cacheHits;
            const double f = it->second.fitness;
            if (f < m_bestF) { m_bestF = f; m_bestKey = key; }
            m_progressCb(m_totalCalls, m_totalEvals, f, it->second.descStr, true);
            return f;
        }

        // Not cached — train the architecture
        ++m_uniqueEvals;
        if (m_stop.load()) return 1e9;  // abort cleanly

        const double f = trainAndMeasure(numLayers, sizes, acts, key);
        if (f < m_bestF) { m_bestF = f; m_bestKey = key; }
        m_progressCb(m_totalCalls, m_totalEvals, f, m_cache[key].descStr, false);
        return f;
    }

    // ── Accessors ─────────────────────────────────────────────────────────────
    int    uniqueEvals() const { return m_uniqueEvals; }
    int    cacheHits()   const { return m_cacheHits;   }
    double bestFitness() const { return m_bestF;        }

    std::string bestDesc() const {
        auto it = m_cache.find(m_bestKey);
        return it != m_cache.end() ? it->second.descStr : "";
    }

    // Top-K unique architectures sorted by fitness (ascending).
    // Tiebreaker: fewer parameters first (Occam's razor — prefer simpler models).
    std::vector<ArchEntry> getTopK(int k) const {
        std::vector<ArchEntry> all;
        all.reserve(m_cache.size());
        for (const auto& kv : m_cache) all.push_back(kv.second);
        std::sort(all.begin(), all.end(),
                  [](const ArchEntry& a, const ArchEntry& b) {
                      if (std::abs(a.fitness - b.fitness) > 1e-8)
                          return a.fitness < b.fitness;
                      return a.numParams < b.numParams;   // tiebreaker: simpler model
                  });
        if (k > 0 && static_cast<int>(all.size()) > k) all.resize(k);
        return all;
    }

    bool isClassification() const { return m_isCls; }

private:
    Config              m_cfg;
    const Dataset*      m_ds;
    int                 m_dim;
    int                 m_totalEvals;
    int                 m_totalCalls  = 0;
    int                 m_uniqueEvals = 0;
    int                 m_cacheHits   = 0;
    double              m_bestF       = std::numeric_limits<double>::infinity();
    std::string         m_bestKey;
    std::atomic<bool>&  m_stop;
    ProgressCb          m_progressCb;
    bool                m_isCls       = false;

    std::vector<double> m_lb, m_ub;
    std::unordered_map<std::string, ArchEntry> m_cache;

    std::string makeKey(int numLayers, const std::vector<int>& sizes,
                        const std::vector<Activation>& acts) const {
        std::string k = std::to_string(numLayers);
        for (int l = 0; l < numLayers; ++l)
            k += ":" + std::to_string(sizes[l]) + "_" + std::to_string(static_cast<int>(acts[l]));
        return k;
    }

    std::string makeDesc(int numLayers, const std::vector<int>& sizes,
                         const std::vector<Activation>& acts) const {
        std::string d = std::to_string(numLayers) + "L: ";
        for (int l = 0; l < numLayers; ++l) {
            if (l > 0) d += " \xE2\x86\x92 ";   // →
            d += std::to_string(sizes[l]) + "\xC2\xB7"; // ·
            // Find the name by searching candidateActs (same approach as actNamesList)
            std::string aname = "?";
            for (int k = 0; k < static_cast<int>(m_cfg.candidateActs.size()); ++k) {
                if (m_cfg.candidateActs[k] == acts[l]) {
                    aname = (k < static_cast<int>(m_cfg.actNames.size()))
                            ? m_cfg.actNames[k] : "?";
                    break;
                }
            }
            d += aname;
        }
        return d;
    }

    double trainAndMeasure(int numLayers, const std::vector<int>& sizes,
                           const std::vector<Activation>& acts,
                           const std::string& key) {
        // Build network
        auto net = buildNet(m_ds, sizes, acts);
        m_isCls = (net->taskType() != TaskType::Regression);

        // Train with inner optimizer
        NSProblem prob(net.get(), m_ds, m_cfg.innerWBound, m_cfg.valSplit);
        auto opt = makeOpt(QString::fromStdString(m_cfg.innerOpt));
        const int maxEvals = m_cfg.innerIters * m_cfg.innerPop;
        opt->setup(&prob, maxEvals, m_cfg.innerPop, makeSeed());
        opt->init();
        for (int it = 1; it <= m_cfg.innerIters && !m_stop.load() && !opt->done(); ++it) {
            opt->one_iteration();
            opt->advanceIter();
        }
        opt->end();
        std::vector<double> best = opt->bestX();
        double trainLoss = opt->bestF();
        // L-BFGS polish inside the same box (the line search projects onto it); kept only if not worse
        if (m_cfg.innerPolishEvals > 0 && !best.empty() && !m_stop.load()) {
            optimsolution::LBFGS bfgs;
            bfgs.setMemory(20);
            bfgs.setup(&prob, prob.calls() + m_cfg.innerPolishEvals, 1, makeSeed());
            bfgs.setInitialPoint(best);
            bfgs.init();
            while (!bfgs.done() && !m_stop.load()) {
                const int c0 = prob.calls();
                bfgs.one_iteration(); bfgs.advanceIter();
                if (prob.calls() == c0) break;   // converged: an iteration that evaluates nothing
            }
            if (!bfgs.bestX().empty() && bfgs.bestF() <= trainLoss) { best = bfgs.bestX(); trainLoss = bfgs.bestF(); }
        }
        if (!best.empty()) NSProblem::unflatten(*net, best);

        // Measure fitness = val error (lower = better)
        double fitness;
        double trainErr;

        if (m_cfg.valSplit > 0) {
            if (m_isCls) {
                trainErr = (1.0 - prob.trainAcc()) * 100.0;
                const double va = prob.valAcc();          // -1: no validation rows (tiny dataset)
                fitness  = va >= 0 ? (1.0 - va) * 100.0 : trainErr;
            } else {
                trainErr = trainLoss;
                const double vl = prob.valLoss();
                fitness  = vl >= 0 ? vl : trainLoss;
            }
        } else {
            // No validation split: use train loss as fitness
            fitness  = trainLoss;
            trainErr = trainLoss;
        }

        // error of this trained network on the companion test file (scored only: the search never
        // sees it; before, Mode 5 reported nothing but validation errors, on which it also selected)
        double testErr = -1;
        if (m_cfg.testDs && m_cfg.testDs->sampleCount > 0 && m_cfg.testDs->inputCount == m_ds->inputCount) {
            NSProblem tp(net.get(), m_cfg.testDs, 1.0, 0.0);
            const double tl = tp.evaluate(NSProblem::flatten(*net));
            testErr = m_isCls ? (1.0 - tp.trainAcc()) * 100.0 : tl;
        }

        // Cache
        ArchEntry e;
        e.sizes      = std::vector<int>(sizes.begin(), sizes.end());
        e.acts       = std::vector<Activation>(acts.begin(), acts.end());
        e.descStr    = makeDesc(numLayers, sizes, acts);
        e.trainErr   = trainErr;
        e.valErr     = fitness;
        e.testErr    = testErr;
        e.trainLoss  = trainLoss;
        e.numParams  = net->totalParams();
        e.evalNum    = m_uniqueEvals;
        e.fitness    = fitness;
        // Store act names by finding each act's position in candidateActs
        for (int l = 0; l < numLayers; ++l) {
            std::string aname = "?";
            for (int k = 0; k < static_cast<int>(m_cfg.candidateActs.size()); ++k) {
                if (m_cfg.candidateActs[k] == acts[l]) {
                    aname = (k < static_cast<int>(m_cfg.actNames.size())) ? m_cfg.actNames[k] : "?";
                    break;
                }
            }
            e.actNamesList.push_back(aname);
        }
        m_cache[key] = e;

        return fitness;
    }
};

// ── ArchSearchWorker ─────────────────────────────────────────────────────────
ArchSearchWorker::ArchSearchWorker(QObject* parent) : QObject(parent) {}

void ArchSearchWorker::run() {
    // An exception escaping a worker-thread slot terminates the whole application (an unreadable
    // dataset file did exactly that); it is reported like any other error instead.
    try {
        for (int i = 0; i < m_cfg.datasetPaths.size() && !m_stop.load(); ++i) {
            if (!runOnDataset(i, m_cfg.datasetPaths[i])) break;
        }
    } catch (const std::exception& ex) {
        emit errorOccurred(QString::fromStdString(ex.what()));
    }
    emit allDone();
}

bool ArchSearchWorker::runOnDataset(int dsIdx, const QString& dsPath) {
    // Load dataset
    // The architecture is chosen on the TRAIN file only (internal validation split).
    // loadCombined merged the companion test file into it, so the test rows took part in
    // the selection, and a later test evaluation of the chosen architecture was biased.
    DatasetBundle bundle = DatasetLoader::loadBundle(dsPath);
    auto ds = bundle.train;
    if (ds) ds->computeStatistics();
    auto testDs = bundle.test;
    if (testDs) testDs->computeStatistics();
    if (!ds) {
        emit errorOccurred(QString("Failed to load dataset: %1").arg(dsPath));
        return false;
    }
    const QString dsName = QFileInfo(dsPath).baseName();
    emit datasetStarted(dsIdx, m_cfg.datasetPaths.size(), dsName);

    // Build ArchProblem config
    ArchProblem::Config pcfg;
    pcfg.minLayers   = m_cfg.minLayers;
    pcfg.maxLayers   = m_cfg.maxLayers;
    pcfg.innerOpt    = m_cfg.innerOpt.toStdString();
    pcfg.innerPop    = m_cfg.innerPop;
    pcfg.innerIters  = m_cfg.innerIters;
    pcfg.innerWBound = m_cfg.innerWBound;
    pcfg.valSplit    = m_cfg.valSplit;
    pcfg.innerPolishEvals = m_cfg.innerPolishEvals;
    pcfg.testDs      = testDs.get();

    for (const int s : m_cfg.candidateSizes)
        pcfg.candidateSizes.push_back(s);
    for (const QString& a : m_cfg.candidateActs) {
        pcfg.candidateActs.push_back(parseAct(a));
        pcfg.actNames.push_back(a.toStdString());
    }

    const int totalEvals = m_cfg.outerPop * m_cfg.outerGens;

    // Progress callback: forward to signal (will be queued across thread boundary)
    auto* self = this;
    ArchProblem prob(pcfg, ds.get(), totalEvals, m_stop,
        [self, totalEvals](int evalNum, int /*total*/, double fitness,
                           const std::string& desc, bool cached) {
            emit self->evalDone(evalNum, totalEvals, fitness,
                                QString::fromStdString(desc), cached);
        });

    // Create outer optimizer
    auto opt = makeOpt(m_cfg.outerOpt);
    opt->setup(&prob, totalEvals, m_cfg.outerPop, makeSeed());
    opt->init();

    QVector<double> convergence;
    convergence.reserve(m_cfg.outerGens);

    for (int gen = 1; gen <= m_cfg.outerGens && !m_stop.load() && !opt->done(); ++gen) {
        opt->one_iteration();
        opt->advanceIter();

        const double best = prob.bestFitness();
        convergence.append(std::isinf(best) ? 100.0 : best);

        emit generationDone(gen, m_cfg.outerGens, best,
                            QString::fromStdString(prob.bestDesc()),
                            prob.uniqueEvals(), prob.cacheHits());
    }
    opt->end();

    // Collect top-K results
    auto entries = prob.getTopK(m_cfg.topK);
    QVector<ArchResult> topResults;
    for (int i = 0; i < static_cast<int>(entries.size()); ++i) {
        const auto& e = entries[i];
        ArchResult r;
        r.rank     = i + 1;
        r.desc     = QString::fromStdString(e.descStr);
        r.trainErr = e.trainErr;
        r.valErr   = e.valErr;
        r.testErr  = e.testErr;
        r.trainLoss= e.trainLoss;
        r.numParams= e.numParams;
        r.evalNum  = e.evalNum;
        for (int s : e.sizes)  r.sizes.append(s);
        for (const auto& aname : e.actNamesList)
            r.actNames.append(QString::fromStdString(aname));
        topResults.append(r);
    }

    emit datasetFinished(dsName, topResults, convergence, prob.isClassification());
    return true;
}

} // namespace NeuralStudio
