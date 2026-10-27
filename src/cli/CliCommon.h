#pragma once
// CliCommon.h — shared pieces of the NeuralStudio console program (neuralstudio-cli):
// console output and progress, Ctrl+C handling, data-file lists, optimizer names,
// network evaluation, number formatting and result tables (text and .xlsx).
//
// The console program calls the SAME workers as the graphical interface (Trainer,
// MetaTrainer, RBFTrainer, CrossValidator, BatchRunner, PreTrainWorker, EvolBoundsWorker,
// ArchSearchWorker, GEStudioWorker, GEBatchRunner); nothing here changes how a model is
// trained or scored.
#include "CliArgs.h"

#include "core/dataset/Dataset.h"
#include "core/nn/ActivationFunction.h"
#include "core/nn/NeuralNetwork.h"
#include "core/nn/RBFNetwork.h"

#include <QString>
#include <QStringList>
#include <QVector>

#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace NeuralStudio {
class XlsxSheet;
class XlsxWriter;
}

namespace NeuralStudio::cli {

constexpr int kExitOk = 0;
constexpr int kExitError = 1;        // a run failed (bad file, training error, ...)
constexpr int kExitUsage = 2;        // invalid command line
constexpr int kExitInterrupted = 130;

// ── Console ───────────────────────────────────────────────────────────────────
// Results go to stdout, progress / warnings / errors to stderr. Thread-safe: the
// workers report from their own thread.
class Console {
public:
    static Console& get();
    void setQuiet(bool q)   { m_quiet = q; }
    void setVerbose(bool v) { m_verbose = v; }
    bool quiet() const      { return m_quiet; }
    bool verbose() const    { return m_verbose; }

    void out(const QString& text);                 // stdout, newline appended
    void info(const QString& text);                // stderr, hidden by --quiet
    void warn(const QString& text);                // stderr, "warning: "
    void error(const QString& text);               // stderr, "error: "
    // Overwritable progress line (stderr, terminals only; throttled). Hidden by --quiet.
    void progress(const QString& text);
    void endProgress();                            // clears the progress line
private:
    Console();
    void clearLineLocked();
    std::mutex m_mx;
    bool  m_quiet = false, m_verbose = false;
    bool  m_tty = false;
    int   m_lastLen = 0;
    qint64 m_lastMs = 0;
};

// ── Ctrl+C ────────────────────────────────────────────────────────────────────
// Runs `body` on a worker thread. While it runs, the calling thread watches for Ctrl+C:
// the first one calls the stop function registered with StopHandle::set (the worker
// finishes the current step and returns, results so far are kept), a second one exits.
// Exceptions thrown by `body` are re-thrown in the calling thread.
class StopHandle {
public:
    void set(std::function<void()> fn);   // called by the worker thread
    void clear();
    void request();                       // called by the watching thread
    bool requested() const;
private:
    mutable std::mutex    m_mx;
    std::function<void()> m_fn;
    bool                  m_requested = false;
};
void installInterruptHandler();
void runInterruptible(StopHandle& stop, const std::function<void()>& body);

// RAII: registers worker->requestStop() for the lifetime of the guard.
template <class W>
struct StopGuard {
    StopHandle& h;
    StopGuard(StopHandle& handle, W* w) : h(handle) { h.set([w] { w->requestStop(); }); }
    ~StopGuard() { h.clear(); }
    StopGuard(const StopGuard&) = delete;
    StopGuard& operator=(const StopGuard&) = delete;
};

// ── Data files ────────────────────────────────────────────────────────────────
// Collects --data values and positional arguments; expands wildcards (* ?) and folders
// (every training file inside), drops the companion *test* file of a train file that is
// also in the list, removes duplicates and checks that every file exists.
bool collectDataFiles(const Args& a, QStringList& files, QString& error);

// ── Optimizers and activations ───────────────────────────────────────────────
QStringList gradientOptimizers();                 // adam sgd adamw rmsprop nadam adagrad
QStringList allTrainingOptimizers();              // the 24 of Modes 1 and 2
QStringList phaseOptimizers();                    // the 14 the workers of Modes 3-5 build
QStringList archOuterOptimizers();                // the 9 population methods of Mode 5 (outer)
QString     optimizerLabel(const QString& id);    // "DE (Differential Evolution)"
bool        isGradientOptimizer(const QString& id);
// Expands "all", "gradient", "meta" and validates the names (lower-case result).
bool        expandOptimizerList(const QStringList& in, const QStringList& allowed,
                                QStringList& out, QString& error);
bool        parseActivation(const QString& s, Activation& a);    // relu tanh sigmoid linear
bool        parseHidden(const QString& s, std::vector<int>& sizes, QString& error);  // "64,32" / "none"

// ── Formatting ────────────────────────────────────────────────────────────────
QString fmtErr(double v, bool cls);               // % with 2 decimals / 4 significant digits; "-" if < 0
QString fmtNum(double v, int sig = 4);
QString fmtMeanSd(const QVector<double>& v, bool cls);   // "12.34 +/- 1.20" (n>1) or "12.34"
double  mean(const QVector<double>& v);
double  sampleSd(const QVector<double>& v);
QString fmtDuration(double seconds);
QString taskLabel(TaskType t);
// Network descriptions of the workers use the symbols "->" and a middle dot; plain ASCII for consoles
QString asciiText(QString s);
// Plain text table (columns padded); numbers right-aligned.
QString textTable(const QStringList& header, const QVector<QStringList>& rows);

// ── Evaluation (same rules as the Evaluation panel / BatchRunner) ────────────
struct EvalResult {
    bool   valid = false;
    TaskType task = TaskType::Regression;
    int    n = 0;
    // classification
    double accuracy = -1, errorPct = -1, precision = -1, recall = -1, f1 = -1;
    std::vector<std::vector<int>> confusion;   // [true][predicted]
    QStringList classLabels;
    // regression (first output column)
    double mse = -1, rmse = -1, mae = -1, r2 = -1;
};
EvalResult evaluateNetwork(NeuralNetwork& net, const Dataset& ds, const std::vector<int>* rows = nullptr);
EvalResult evaluateRbf(const RBFNetwork& net, const Dataset& ds, const std::vector<int>* rows = nullptr);
// The validation rows of Trainer / NSProblem: shuffle 0..n-1 with mt19937(42), keep the last round(n*split).
std::vector<int> validationRows(int n, double split);
// Lines of text for one evaluation (metrics + confusion matrix)
QString describeEval(const QString& title, const EvalResult& e);
// Appends an evaluation as rows of an .xlsx sheet
void    evalToSheet(XlsxSheet* sh, const QString& set, const EvalResult& e);

// Builds an MLP for a training set exactly like the Network panel (task, output size,
// input normalisation, regression output scaling, class values / output mapping).
std::shared_ptr<NeuralNetwork> buildMlp(const Dataset& train, TaskType task, const std::vector<int>& hidden,
                                        const std::vector<Activation>& acts, double dropout);
TaskType resolveTask(const QString& choice, const Dataset& train);   // "auto" | binary | multiclass | regression

// ── Output files ──────────────────────────────────────────────────────────────
QString ensureSuffix(const QString& path, const QString& suffix);  // adds ".xlsx" when missing
bool    ensureParentDir(const QString& path, QString& error);
QString sheetName(const QString& base);   // max 31 chars, no []:*?/\ ; unique names are the caller's job
// Saves network exports requested with --save-model / --export-cpp / --export-onnx
// (cpp namespace from --cpp-namespace). Returns false when a write failed.
bool    writeModelExports(const Args& a, const NeuralNetwork& net);
QVector<OptSpec> modelExportOptions();

// Common options of every command (--quiet, --verbose, --help)
QVector<OptSpec> commonOptions();

} // namespace NeuralStudio::cli
