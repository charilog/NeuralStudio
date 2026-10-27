#include "MetaTrainer.h"
#include "NSProblem.h"
#include "optimizer.h"

// Concrete optimizer headers
#include "de.h"
#include "pso.h"
#include "cmaes.h"
#include "jso.h"
#include "lmcmaes.h"
#include "acor.h"
#include "clpso.h"
#include "mlshaderl.h"
#include "arq3.h"
#include "lbfgs.h"
#include "genetic.h"
#include "conjgrad.h"
#include "levenberg.h"
#include "simann.h"
#include "neldermead.h"
#include "pbil.h"
#include "umda.h"

#include "utils/Logger.h"
#include <stdexcept>
#include <limits>
#include <random>

namespace NeuralStudio {

MetaTrainer::MetaTrainer(QObject* parent) : QObject(parent) {}

// ─── createOptimizer ─────────────────────────────────────────────────────────
std::unique_ptr<optimsolution::Optimizer> MetaTrainer::createOptimizer() const {
    const QString n = m_name.toLower();
    // Quasi-Newton
    if (n == "lbfgs") { auto p=std::make_unique<optimsolution::LBFGS>(); p->setMemory(7);  return p; }
    if (n == "bfgs")  { auto p=std::make_unique<optimsolution::LBFGS>(); p->setMemory(20); return p; }
    if (n == "cg")    return std::make_unique<optimsolution::ConjugateGradient>();
    if (n == "lm")    return std::make_unique<optimsolution::LevenbergMarquardt>();
    // Single-point derivative-free
    if (n == "sa")         return std::make_unique<optimsolution::SimulatedAnnealing>();
    if (n == "neldermead") return std::make_unique<optimsolution::NelderMead>();
    // EDA
    if (n == "pbil") return std::make_unique<optimsolution::PBIL>();
    if (n == "umda") return std::make_unique<optimsolution::UMDA>();
    // Evolutionary / Swarm
    if (n == "ga")          return std::make_unique<optimsolution::GeneticAlgorithm>();
    if (n == "de")          return std::make_unique<optimsolution::DE>();
    if (n == "pso")         return std::make_unique<optimsolution::PSO>();
    if (n == "cmaes")       return std::make_unique<optimsolution::CMAES>();
    if (n == "jso")         return std::make_unique<optimsolution::JSO>();
    if (n == "lmcmaes")     return std::make_unique<optimsolution::LMCMAES>();
    if (n == "acor")        return std::make_unique<optimsolution::ACOR>();
    if (n == "clpso")       return std::make_unique<optimsolution::CLPSO>();
    if (n == "mlshaderl")   return std::make_unique<optimsolution::mLSHADE_RL>();
    if (n == "arq3")        return std::make_unique<optimsolution::ARQ3>();
    throw std::runtime_error("Unknown meta-optimizer: " + m_name.toStdString());
}

// ─── run ─────────────────────────────────────────────────────────────────────
void MetaTrainer::run() {
    try {
        if (!m_net || !m_ds)    throw std::runtime_error("MetaTrainer: network or dataset not set.");
        if (!m_net->isBuilt())  throw std::runtime_error("MetaTrainer: network not built.");

        const int iterations  = m_cfg.epochs;
        const int popSize     = std::max(4, m_cfg.batchSize);  // batchSize = population
        const int maxEvals    = iterations * popSize;

        NS_INFO << QString("[%1] iterations=%2  popSize=%3  maxEvals=%4  D=%5")
                       .arg(m_name).arg(iterations).arg(popSize)
                       .arg(maxEvals).arg(m_net->totalParams());

        // ── Build problem ────────────────────────────────────────────────────
        NSProblem prob(m_net, m_ds, m_wBound,
                       m_externalValDs ? 0.0 : m_cfg.validationSplit);
        if (m_externalValDs)
            prob.setExternalValDataset(m_externalValDs);

        // ── Build optimizer ──────────────────────────────────────────────────
        auto opt = createOptimizer();

        const QString n = m_name.toLower();
        const bool isQuasiNewton  = (n=="lbfgs"||n=="bfgs"||n=="cg"||n=="lm");
        const bool isSinglePoint  = (n=="sa"||n=="neldermead");
        const bool isEvolutionary = !isQuasiNewton && !isSinglePoint;

        const int budget = isQuasiNewton
                           ? iterations * 100
                           : isEvolutionary
                             ? iterations * popSize
                             : iterations * 20;  // single-point: 20 evals/iter budget

        // Propagate population size from user setting (ignored by L-BFGS)
        {
            optimsolution::MethodConfig mc;
            mc.data["population"] = std::to_string(popSize);
            opt->configure(mc);
        }

        std::random_device rd;
        // single-point methods spend `population` evaluations per iteration: 20, the share of
        // the budget above (with popSize they would exhaust it after 20/popSize of the iterations)
        opt->setup(&prob, budget, isSinglePoint ? 20 : popSize, rd());

        // Warm-start: quasi-Newton and single-point methods begin from current weights
        if (isQuasiNewton || isSinglePoint)
            opt->setInitialPoint(NSProblem::flatten(*m_net));

        // ── Initialise ───────────────────────────────────────────────────────
        m_net->setTrainingMode(false);   // no dropout during evolution
        opt->init();

        if (m_stop.load()) { emit trainingFinished(opt->bestF()); return; }

        // ── Apply initial best to network for val computation ─────────────────
        if (!opt->bestX().empty())
            NSProblem::unflatten(*m_net, opt->bestX());

        const bool isCls = (m_net->taskType() != TaskType::Regression);
        double bestValLoss = std::numeric_limits<double>::infinity();
        int    noImprove   = 0;
        std::vector<double> bestValWeights;   // weights of the best validation iteration (early stopping)

        // ── Main loop ────────────────────────────────────────────────────────
        for (int iter = 1;
             iter <= iterations && !m_stop.load() && !opt->done();
             ++iter)
        {
            opt->one_iteration();
            opt->advanceIter();   // update iters_ for parameter schedules (e.g. CLPSO inertia)

            // Apply best weights so monitoring functions see the best solution
            if (!opt->bestX().empty())
                NSProblem::unflatten(*m_net, opt->bestX());

            EpochResult res;
            res.epoch      = iter;
            res.currentLR  = 0.0;                         // N/A for evolutionary
            res.trainLoss  = opt->bestF();
            // External val dataset always yields a val figure, regardless of
            // the (deliberately zeroed) internal split — see finalVal note.
            const bool hasVal = (m_externalValDs || m_cfg.validationSplit > 0);
            res.valLoss    = hasVal ? prob.valLoss() : -1.0;
            if (isCls) {
                res.trainAcc = prob.trainAcc();
                res.valAcc   = hasVal ? prob.valAcc() : -1.0;
            }
            emit epochCompleted(res);

            // ── Early stopping ────────────────────────────────────────────────
            const int patience = m_valIsTest ? 0 : m_cfg.earlyStoppingPatience;
            if (patience > 0 && res.valLoss >= 0.0) {
                if (res.valLoss < bestValLoss - 1e-6) {
                    bestValLoss = res.valLoss;
                    noImprove   = 0;
                    bestValWeights = NSProblem::flatten(*m_net);
                } else if (++noImprove >= patience) {
                    NS_INFO << QString("[%1] Early stopping at iteration %2 "
                                       "(patience=%3 exceeded).")
                                   .arg(m_name).arg(iter).arg(patience);
                    break;
                }
            }
        }

        // ── End phase (some optimizers do final polishing) ───────────────────
        opt->end();

        // Apply the definitive best to the network: with early stopping the weights of the
        // best validation iteration (as the gradient Trainer does), otherwise the best
        // training fitness
        if (!bestValWeights.empty())
            NSProblem::unflatten(*m_net, bestValWeights);
        else if (!opt->bestX().empty())
            NSProblem::unflatten(*m_net, opt->bestX());

        // Final reported loss: an external val dataset always provides a
        // validation figure (prob.valLoss() evaluates on it), even when the
        // config's internal valSplit is 0 — that split is deliberately zeroed
        // in external-val mode, so gating on it alone would wrongly fall back
        // to the raw optimizer fitness (train loss).
        const double finalVal = (m_externalValDs || m_cfg.validationSplit > 0)
                                ? prob.valLoss() : opt->bestF();
        emit trainingFinished(finalVal);

    } catch (const std::exception& ex) {
        emit trainingError(QString::fromStdString(ex.what()));
    }
}

} // namespace NeuralStudio
