#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  optimizer.h  —  NeuralStudio compatibility shim for the optimsolution
//                  optimizer framework.
//
//  The optimsolution optimizer classes (#include "optimizer.h") expect:
//    • Problem     — abstract objective-function interface
//    • MethodConfig— key/value configuration bag
//    • InitOptions — initialisation settings (ignored here)
//    • Optimizer   — base class with rng_, pop_, prob_, best_x/f_, etc.
//
//  All gradient-based "local search" hooks are stubbed (no-op) because
//  NeuralStudio uses only the global search phase of each method.
// ─────────────────────────────────────────────────────────────────────────────
#include <vector>
#include <stdexcept>
#include <string>
#include <map>
#include <random>
#include <limits>
#include <utility>
#include <cmath>

namespace optimsolution {

// ── Vec — convenience alias used throughout the optimizer files ──────────────
using Vec = std::vector<double>;

// ── MethodConfig ─────────────────────────────────────────────────────────────
struct MethodConfig {
    std::map<std::string, std::string> data;

    std::string getStr(const std::string& k, const std::string& def = "") const {
        auto it = data.find(k);
        return it != data.end() ? it->second : def;
    }
    int getInt(const std::string& k, int def = 0) const {
        auto it = data.find(k);
        if (it == data.end()) return def;
        try { return std::stoi(it->second); } catch (...) { return def; }
    }
    double getDbl(const std::string& k, double def = 0.0) const {
        auto it = data.find(k);
        if (it == data.end()) return def;
        try { return std::stod(it->second); } catch (...) { return def; }
    }
    bool getBool(const std::string& k, bool def = false) const {
        auto it = data.find(k);
        if (it == data.end()) return def;
        const auto& v = it->second;
        if (v=="1"||v=="true"||v=="on"||v=="yes") return true;
        if (v=="0"||v=="false"||v=="off"||v=="no") return false;
        return def;
    }
};

// ── InitOptions — (not used in NeuralStudio; just satisfies the include) ─────
struct InitOptions {};

// ── Problem — abstract interface that optimizers see via prob_* ───────────────
class Problem {
public:
    virtual ~Problem() = default;
    virtual int                         dimension() const = 0;
    virtual const std::vector<double>&  lb()        const = 0;
    virtual const std::vector<double>&  ub()        const = 0;
    virtual double evaluate(const std::vector<double>& x)  = 0;
    virtual int    calls()  const                           = 0;

    // Analytical gradient (loss, gradient_vector).
    // Default throws — override in NSProblem with backprop.
    virtual std::pair<double, std::vector<double>>
    evalAndGrad(const std::vector<double>& x) {
        throw std::runtime_error("evalAndGrad not available for this problem type.");
        return {};
    }
};

// ── Optimizer — base class ────────────────────────────────────────────────────
class Optimizer {
public:
    virtual ~Optimizer() = default;

    virtual std::string methodShortName() const = 0;
    virtual std::string methodFullName()  const = 0;
    virtual void configure(const MethodConfig& /*mc*/) {}
    virtual void setEndLocalFromGlobal(bool enable, const std::string& method) {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

    // ── Framework setup (called by MetaTrainer before init()) ────────────────
    void setup(Problem* prob, int maxEvals, int popSize, unsigned seed) {
        prob_       = prob;
        max_evals_  = maxEvals;
        pop_        = popSize;
        max_iters_  = (popSize > 0) ? (maxEvals / popSize) : maxEvals;
        iters_      = 0;
        rng_.seed(seed);
        best_f_ = std::numeric_limits<double>::infinity();
        // Default start point of the single-point methods (L-BFGS, CG, LM, SA,
        // Nelder-Mead) when no setInitialPoint() follows: a small random point
        // around the centre of the box. (It was the all-zero vector, which for a
        // neural network is a symmetric saddle point: every hidden unit receives
        // the same gradient and they never become different.)
        if (prob_) {
            const int D = prob_->dimension();
            best_x_.assign(D, 0.0);
            const auto& lb = prob_->lb(); const auto& ub = prob_->ub();
            std::uniform_real_distribution<double> u(-1.0, 1.0);
            for (int j = 0; j < D && j < static_cast<int>(lb.size()) && j < static_cast<int>(ub.size()); ++j) {
                const double w = ub[j] - lb[j];
                if (!std::isfinite(w) || w <= 0) continue;
                const double c = 0.5 * (lb[j] + ub[j]);
                const double r = std::min(1.0, 0.05 * w);
                best_x_[j] = std::min(ub[j], std::max(lb[j], c + r * u(rng_)));
            }
        }
        has_initial_point_ = false;
    }

    virtual void init()          = 0;
    virtual void one_iteration() = 0;
    virtual void end() {}

    double                     bestF() const { return best_f_; }
    const std::vector<double>& bestX() const { return best_x_; }
    bool done() const { return prob_ && prob_->calls() >= max_evals_; }

    // Override starting point (e.g. warm-start from previously trained weights).
    // The point is injected as the first population member in init() via
    // injectInitialPoint(). SA / NelderMead / LBFGS read best_x_ directly.
    void setInitialPoint(const std::vector<double>& x) {
        best_x_           = x;
        has_initial_point_ = !x.empty();
        // a warm start outside the box (e.g. weights trained under wider bounds) is
        // projected onto it, so no method evaluates or returns an infeasible point
        if (prob_ && static_cast<int>(best_x_.size()) == prob_->dimension()) {
            const auto& lb = prob_->lb(); const auto& ub = prob_->ub();
            for (size_t j = 0; j < best_x_.size() && j < lb.size() && j < ub.size(); ++j)
                best_x_[j] = std::min(ub[j], std::max(lb[j], best_x_[j]));
        }
    }

    // Called by MetaTrainer after every one_iteration() so that optimizers
    // that use iters_ / max_iters_ for parameter schedules get correct values.
    void advanceIter() { ++iters_; }

protected:
    Problem*            prob_      = nullptr;
    std::mt19937        rng_;
    int                 pop_       = 20;
    int                 max_evals_ = 10000;
    int                 max_iters_ = 0;   // ≈ max_evals_ / pop_; for inertia schedules
    int                 iters_     = 0;   // current iteration counter (0-indexed)
    double              best_f_    = std::numeric_limits<double>::infinity();
    std::vector<double> best_x_;
    InitOptions         initopt_;

    int  population() const    { return pop_; }
    void setPopulation(int n)  { pop_ = n;   }

    // ── Stubs used by several optimizers ─────────────────────────────────────
    void printBest()  {}   // no-op

    // updateStop(fitness_vec): in the original framework this triggers
    // stopping-criterion bookkeeping. Here it is a no-op; MetaTrainer controls
    // stopping via done() / iteration count / early-stopping patience.
    void updateStop(const std::vector<double>&) {}

    bool        finalLocalEnabled() const { return end_local_refine_; }
    const std::string& finalLocalMethod() const { return end_local_method_; }

    // localSearch: gradient-based refinement is not available inside NeuralStudio.
    // Return the candidate unchanged with fitness +inf, so every call site's
    // "floc < f" test fails and nothing is replaced. (Returning best_f_ here, as
    // before, gave an unchanged candidate the fitness of the global best and
    // corrupted the population whenever in-run local search was enabled.)
    std::pair<std::vector<double>, double>
    localSearch(const std::string&, const std::vector<double>& x) {
        return { x, std::numeric_limits<double>::infinity() };
    }

    std::string local_method_;
    double      local_rate_        = 0.0;
    bool        end_local_refine_  = false;
    std::string end_local_method_;
    bool        has_initial_point_ = false;   // set by setInitialPoint()

    // Injects the warm-start (best_x_) into pop[0] when setInitialPoint was
    // called before init().  Population-based optimizers call this immediately
    // after samplePopulation(), before resetting best_f_/best_x_.
    void injectInitialPoint(std::vector<std::vector<double>>& pop) {
        if (!has_initial_point_ || pop.empty()) return;
        if (!prob_ || (int)best_x_.size() != prob_->dimension()) return;
        pop[0] = best_x_;
        has_initial_point_ = false;  // consumed — no re-injection on restart
    }
};

} // namespace optimsolution
