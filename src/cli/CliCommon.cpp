#include "CliCommon.h"

#include "core/dataset/CompanionFinder.h"
#include "core/dataset/DatasetLoader.h"
#include "core/export/CppExporter.h"
#include "core/export/OnnxExporter.h"
#include "core/export/XlsxWriter.h"
#include "core/nn/ModelSerializer.h"

#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QSet>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <random>
#include <thread>

#ifdef _WIN32
#  include <io.h>
#  define NS_ISATTY(fd) _isatty(fd)
#  define NS_FILENO(f)  _fileno(f)
#else
#  include <unistd.h>
#  define NS_ISATTY(fd) isatty(fd)
#  define NS_FILENO(f)  fileno(f)
#endif

namespace NeuralStudio::cli {

// ═══════════════════════════════════════════════════════════════════════════════
// Console
// ═══════════════════════════════════════════════════════════════════════════════
Console& Console::get() {
    static Console c;
    return c;
}

Console::Console() {
    m_tty = NS_ISATTY(NS_FILENO(stderr)) != 0;
}

void Console::clearLineLocked() {
    if (m_lastLen > 0) {
        std::fprintf(stderr, "\r%s\r", std::string(static_cast<size_t>(m_lastLen), ' ').c_str());
        std::fflush(stderr);
        m_lastLen = 0;
    }
}

void Console::out(const QString& text) {
    std::lock_guard<std::mutex> lk(m_mx);
    clearLineLocked();
    const QByteArray b = text.toUtf8();
    std::fwrite(b.constData(), 1, static_cast<size_t>(b.size()), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

void Console::info(const QString& text) {
    if (m_quiet) return;
    std::lock_guard<std::mutex> lk(m_mx);
    clearLineLocked();
    std::fprintf(stderr, "%s\n", text.toUtf8().constData());
    std::fflush(stderr);
}

void Console::warn(const QString& text) {
    std::lock_guard<std::mutex> lk(m_mx);
    clearLineLocked();
    std::fprintf(stderr, "warning: %s\n", text.toUtf8().constData());
    std::fflush(stderr);
}

void Console::error(const QString& text) {
    std::lock_guard<std::mutex> lk(m_mx);
    clearLineLocked();
    std::fprintf(stderr, "error: %s\n", text.toUtf8().constData());
    std::fflush(stderr);
}

void Console::progress(const QString& text) {
    if (m_quiet) return;
    std::lock_guard<std::mutex> lk(m_mx);
    const qint64 now = QDateTime::currentMSecsSinceEpoch();
    if (m_tty) {
        if (now - m_lastMs < 100) return;          // at most 10 updates per second
        m_lastMs = now;
        QString t = text;
        if (t.size() > 118) t = t.left(115) + "...";
        const QByteArray b = t.toUtf8();
        const int pad = std::max(0, m_lastLen - static_cast<int>(t.size()));
        std::fprintf(stderr, "\r%s%s", b.constData(), std::string(static_cast<size_t>(pad), ' ').c_str());
        std::fflush(stderr);
        m_lastLen = static_cast<int>(t.size());
    } else {
        // redirected output (log file, CI): one line every 30 s at most
        if (m_lastMs != 0 && now - m_lastMs < 30000) return;
        m_lastMs = now;
        std::fprintf(stderr, "%s\n", text.toUtf8().constData());
        std::fflush(stderr);
    }
}

void Console::endProgress() {
    std::lock_guard<std::mutex> lk(m_mx);
    clearLineLocked();
    if (!m_tty) m_lastMs = 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Ctrl+C
// ═══════════════════════════════════════════════════════════════════════════════
namespace {
volatile std::sig_atomic_t g_sigCount = 0;
extern "C" void onSignal(int sig) {
    g_sigCount = g_sigCount + 1;
    std::signal(sig, onSignal);   // some platforms reset the handler after delivery
}
} // namespace

void installInterruptHandler() {
    std::signal(SIGINT, onSignal);
#ifdef SIGTERM
    std::signal(SIGTERM, onSignal);
#endif
}

void StopHandle::set(std::function<void()> fn) {
    std::lock_guard<std::mutex> lk(m_mx);
    m_fn = std::move(fn);
    if (m_requested && m_fn) m_fn();   // a stop requested before the worker existed
}

void StopHandle::clear() {
    std::lock_guard<std::mutex> lk(m_mx);
    m_fn = nullptr;
}

void StopHandle::request() {
    std::lock_guard<std::mutex> lk(m_mx);
    m_requested = true;
    if (m_fn) m_fn();
}

bool StopHandle::requested() const {
    std::lock_guard<std::mutex> lk(m_mx);
    return m_requested;
}

void runInterruptible(StopHandle& stop, const std::function<void()>& body) {
    std::atomic<bool> done{false};
    std::exception_ptr failure;
    std::thread worker([&] {
        try { body(); } catch (...) { failure = std::current_exception(); }
        done.store(true);
    });
    // The first Ctrl+C asks the worker to stop; another one more than a second later quits at
    // once. (Signals within that second count as one: tools such as 'timeout' or a process
    // manager may deliver the same interrupt to the process and to its process group.)
    int seen = static_cast<int>(g_sigCount);   // signals already handled (or ignored)
    std::chrono::steady_clock::time_point first{};
    bool asked = false;
    while (!done.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const int c = static_cast<int>(g_sigCount);
        if (c == seen) continue;
        const auto now = std::chrono::steady_clock::now();
        if (!asked) {
            asked = true;
            first = now;
            Console::get().warn("interrupted: stopping after the current step "
                                "(results so far are kept; press Ctrl+C again to quit at once)");
            stop.request();
        } else if (now - first > std::chrono::seconds(1)) {
            Console::get().error("interrupted");
            std::_Exit(kExitInterrupted);
        }
        seen = c;   // duplicates within the first second are absorbed here
    }
    worker.join();
    if (failure) std::rethrow_exception(failure);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Data files
// ═══════════════════════════════════════════════════════════════════════════════
namespace {
QString existingPath(const QString& raw) {
    QString p = raw.trimmed();
    if (p.startsWith("~/")) p = QDir::homePath() + p.mid(1);
    if (QFileInfo::exists(p)) return QFileInfo(p).absoluteFilePath();
    QString q = p;
    q.replace('\\', '/');   // Windows-style paths on Linux / macOS
    if (q != p && QFileInfo::exists(q)) return QFileInfo(q).absoluteFilePath();
    return {};
}

bool isDataFile(const QFileInfo& fi) {
    static const QStringList exts = DatasetLoader::supportedExtensions();
    return exts.contains(fi.suffix().toLower());
}
} // namespace

bool collectDataFiles(const Args& a, QStringList& files, QString& error) {
    QStringList raw = a.list("data");
    raw += a.positional();
    if (raw.isEmpty()) { error = "No data file given (use --data <file> or list the files at the end)."; return false; }

    QStringList found;
    for (const QString& r : raw) {
        if (r.contains('*') || r.contains('?')) {
            QString pat = r;
            pat.replace('\\', '/');
            const QFileInfo fi(pat);
            QDir dir(fi.path().isEmpty() ? "." : fi.path());
            const QStringList hits = dir.entryList(QStringList{fi.fileName()}, QDir::Files, QDir::Name);
            if (hits.isEmpty()) { error = QString("No file matches '%1'.").arg(r); return false; }
            for (const QString& h : hits) found << dir.absoluteFilePath(h);
            continue;
        }
        const QString p = existingPath(r);
        if (p.isEmpty()) { error = QString("File not found: %1").arg(r); return false; }
        const QFileInfo fi(p);
        if (fi.isDir()) {
            QDir dir(p);
            const QStringList all = dir.entryList(QDir::Files, QDir::Name);
            int n = 0;
            for (const QString& f : all) {
                const QFileInfo ffi(dir.absoluteFilePath(f));
                if (!isDataFile(ffi)) continue;
                if (CompanionFinder::roleOf(ffi.filePath()) == "test") continue;   // loaded as companions
                found << ffi.absoluteFilePath();
                ++n;
            }
            if (n == 0) { error = QString("No data file in folder %1.").arg(r); return false; }
            continue;
        }
        found << p;
    }

    // remove duplicates, keep the order
    QSet<QString> seen;
    QStringList uniq;
    for (const QString& f : found) {
        const QString key = QFileInfo(f).canonicalFilePath().isEmpty() ? f : QFileInfo(f).canonicalFilePath();
        if (seen.contains(key)) continue;
        seen.insert(key);
        uniq << f;
    }
    // a test file whose training file is also listed is its companion, not a dataset of its own
    QSet<QString> uniqSet(uniq.begin(), uniq.end());
    files.clear();
    for (const QString& f : uniq) {
        if (CompanionFinder::roleOf(f) == "test") {
            const QString comp = CompanionFinder::findCompanion(f);
            if (!comp.isEmpty() && uniqSet.contains(QFileInfo(comp).absoluteFilePath())) continue;
        }
        files << f;
    }
    if (files.isEmpty()) { error = "No data file left after removing test companions."; return false; }
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Optimizers and activations
// ═══════════════════════════════════════════════════════════════════════════════
QStringList gradientOptimizers() { return {"adam", "sgd", "adamw", "rmsprop", "nadam", "adagrad"}; }

QStringList allTrainingOptimizers() {
    return {"adam", "sgd", "adamw", "rmsprop", "nadam", "adagrad",
            "lbfgs", "bfgs", "cg", "lm", "sa", "neldermead",
            "ga", "de", "pso", "clpso", "acor", "cmaes", "lmcmaes", "jso", "mlshaderl", "arq3",
            "pbil", "umda"};
}

QStringList phaseOptimizers() {
    return {"ga", "de", "pso", "cmaes", "jso", "lmcmaes", "acor", "clpso", "mlshaderl", "arq3",
            "lbfgs", "bfgs", "sa", "neldermead"};
}

QStringList archOuterOptimizers() {
    return {"ga", "de", "pso", "cmaes", "jso", "lmcmaes", "acor", "clpso", "mlshaderl"};
}

QString optimizerLabel(const QString& id) {
    static const QMap<QString, QString> m = {
        {"adam", "Adam"}, {"sgd", "SGD"}, {"adamw", "AdamW (decoupled weight decay)"},
        {"rmsprop", "RMSProp"}, {"nadam", "Nadam (Nesterov + Adam)"}, {"adagrad", "AdaGrad"},
        {"lbfgs", "L-BFGS (limited-memory BFGS, m=7)"}, {"bfgs", "BFGS (quasi-Newton, m=20)"},
        {"cg", "Conjugate Gradient (Polak-Ribiere)"}, {"lm", "Levenberg-Marquardt (diagonal)"},
        {"sa", "Simulated Annealing"}, {"neldermead", "Nelder-Mead simplex"},
        {"ga", "GA (Genetic Algorithm)"}, {"de", "DE (Differential Evolution)"},
        {"pso", "PSO (Particle Swarm)"}, {"clpso", "CLPSO (Comprehensive Learning PSO)"},
        {"acor", "ACOR (Ant Colony, continuous)"}, {"cmaes", "CMA-ES"},
        {"lmcmaes", "LM-CMA-ES (limited-memory CMA-ES)"}, {"jso", "jSO"},
        {"mlshaderl", "mLSHADE-RL"}, {"arq3", "ARQ3"},
        {"pbil", "PBIL (Population-Based Incremental Learning)"},
        {"umda", "UMDA (Univariate Marginal Distribution)"},
        {"none", "none (skip the phase)"}, {"ge_grammar", "GE grammar bounds (BDCC 2025)"}};
    return m.value(id, id);
}

bool isGradientOptimizer(const QString& id) { return gradientOptimizers().contains(id.toLower()); }

bool expandOptimizerList(const QStringList& in, const QStringList& allowed, QStringList& out, QString& error) {
    out.clear();
    auto add = [&out](const QString& s) { if (!out.contains(s)) out << s; };
    for (const QString& raw : in) {
        const QString s = raw.trimmed().toLower();
        if (s == "all") { for (const QString& o : allowed) add(o); continue; }
        if (s == "gradient") { for (const QString& o : allowed) if (isGradientOptimizer(o)) add(o); continue; }
        if (s == "meta") { for (const QString& o : allowed) if (!isGradientOptimizer(o)) add(o); continue; }
        if (!allowed.contains(s)) {
            error = QString("Unknown optimizer '%1'. Choose from: %2.").arg(raw, allowed.join(", "));
            return false;
        }
        add(s);
    }
    if (out.isEmpty()) { error = "No optimizer selected."; return false; }
    return true;
}

bool parseActivation(const QString& s, Activation& a) {
    const QString t = s.trimmed().toLower();
    if (t == "relu")    { a = Activation::ReLU;    return true; }
    if (t == "tanh")    { a = Activation::Tanh;    return true; }
    if (t == "sigmoid") { a = Activation::Sigmoid; return true; }
    if (t == "linear")  { a = Activation::Linear;  return true; }
    return false;
}

bool parseHidden(const QString& s, std::vector<int>& sizes, QString& error) {
    sizes.clear();
    const QString t = s.trimmed().toLower();
    if (t.isEmpty() || t == "none" || t == "0") return true;
    for (const QString& part : t.split(',', Qt::SkipEmptyParts)) {
        bool ok = false;
        const int n = part.trimmed().toInt(&ok);
        if (!ok || n < 1 || n > 4096) {
            error = QString("Invalid hidden layer size '%1' (use e.g. 64,32).").arg(part.trimmed());
            return false;
        }
        sizes.push_back(n);
    }
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Formatting
// ═══════════════════════════════════════════════════════════════════════════════
QString fmtNum(double v, int sig) {
    if (!std::isfinite(v)) return "nan";
    return QString::number(v, 'g', sig);
}

QString fmtErr(double v, bool cls) {
    if (!std::isfinite(v) || v < 0) return "-";
    return cls ? QString::number(v, 'f', 2) : fmtNum(v, 4);
}

double mean(const QVector<double>& v) {
    if (v.isEmpty()) return 0.0;
    double s = 0;
    for (double x : v) s += x;
    return s / v.size();
}

double sampleSd(const QVector<double>& v) {
    if (v.size() < 2) return 0.0;
    const double m = mean(v);
    double s = 0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / (v.size() - 1));
}

QString fmtMeanSd(const QVector<double>& v, bool cls) {
    if (v.isEmpty()) return "-";
    const QString m = cls ? QString::number(mean(v), 'f', 2) : fmtNum(mean(v), 4);
    if (v.size() == 1) return m;
    const QString s = cls ? QString::number(sampleSd(v), 'f', 2) : fmtNum(sampleSd(v), 3);
    return m + " +/- " + s;
}

QString fmtDuration(double seconds) {
    if (seconds < 60) return QString::number(seconds, 'f', 1) + " s";
    const qint64 s = static_cast<qint64>(std::llround(seconds));
    if (s < 3600) return QString("%1 min %2 s").arg(s / 60).arg(s % 60);
    return QString("%1 h %2 min").arg(s / 3600).arg((s % 3600) / 60);
}

QString taskLabel(TaskType t) {
    switch (t) {
        case TaskType::BinaryClassification:     return "binary classification";
        case TaskType::MultiClassClassification: return "multi-class classification";
        default:                                 return "regression";
    }
}

QString asciiText(QString s) {
    s.replace(QString::fromUtf8("\xE2\x86\x92"), "->");   // right arrow
    s.replace(QString::fromUtf8("\xC2\xB7"), " ");          // middle dot
    s.replace(QString::fromUtf8("\xE2\x80\x94"), "-");     // em dash
    s.replace(QString::fromUtf8("\xE2\x80\xA6"), "...");   // ellipsis
    s.replace(QString::fromUtf8("\xC2\xB1"), "+/-");        // plus-minus
    return s;
}

QString textTable(const QStringList& header, const QVector<QStringList>& rows) {
    const int nc = header.size();
    QVector<int> w(nc, 0);
    for (int c = 0; c < nc; ++c) w[c] = header[c].size();
    for (const auto& r : rows)
        for (int c = 0; c < nc && c < r.size(); ++c) w[c] = std::max(w[c], static_cast<int>(r[c].size()));
    auto numeric = [](const QString& s) {
        if (s == "-") return true;
        bool ok = false;
        s.section(' ', 0, 0).toDouble(&ok);
        return ok;
    };
    auto line = [&](const QStringList& r, bool head) {
        QString out = " ";
        for (int c = 0; c < nc; ++c) {
            const QString cell = c < r.size() ? r[c] : QString();
            const bool right = !head && c > 0 && numeric(cell);
            out += right ? cell.rightJustified(w[c]) : cell.leftJustified(w[c]);
            if (c + 1 < nc) out += "  ";
        }
        while (out.endsWith(' ')) out.chop(1);
        return out;
    };
    QString t = line(header, true) + "\n";
    int total = 1;
    for (int c = 0; c < nc; ++c) total += w[c] + (c + 1 < nc ? 2 : 0);
    t += " " + QString(total - 1, '-') + "\n";
    for (const auto& r : rows) t += line(r, false) + "\n";
    return t;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Evaluation
// ═══════════════════════════════════════════════════════════════════════════════
std::vector<int> validationRows(int n, double split) {
    std::vector<int> idx(static_cast<size_t>(std::max(0, n)));
    for (int i = 0; i < n; ++i) idx[static_cast<size_t>(i)] = i;
    std::mt19937 rng{42};
    std::shuffle(idx.begin(), idx.end(), rng);
    const int nVal = static_cast<int>(std::round(n * split));
    if (nVal <= 0) return {};
    return std::vector<int>(idx.end() - nVal, idx.end());
}

namespace {
void finishClassification(EvalResult& e, int K, int positive) {
    int correct = 0, total = 0;
    for (int t = 0; t < K; ++t)
        for (int p = 0; p < K; ++p) { total += e.confusion[t][p]; if (t == p) correct += e.confusion[t][p]; }
    e.n = total;
    if (total == 0) return;
    e.accuracy = static_cast<double>(correct) / total;
    e.errorPct = (1.0 - e.accuracy) * 100.0;
    if (K == 2) {
        const int tp = e.confusion[positive][positive];
        const int fp = e.confusion[1 - positive][positive];
        const int fn = e.confusion[positive][1 - positive];
        e.precision = (tp + fp) > 0 ? static_cast<double>(tp) / (tp + fp) : 0.0;
        e.recall    = (tp + fn) > 0 ? static_cast<double>(tp) / (tp + fn) : 0.0;
        e.f1 = (e.precision + e.recall) > 0 ? 2 * e.precision * e.recall / (e.precision + e.recall) : 0.0;
    }
    e.valid = true;
}

template <class PredictFn>
void finishRegression(EvalResult& e, const Dataset& ds, const std::vector<int>& rows, PredictFn predict0) {
    const int n = static_cast<int>(rows.size());
    if (n == 0) return;
    double sumT = 0;
    for (int r : rows) sumT += ds.outputAt(r, 0);
    const double meanT = sumT / n;
    double sse = 0, sae = 0, sst = 0;
    for (int r : rows) {
        const double t = ds.outputAt(r, 0);
        const double d = predict0(r) - t;
        sse += d * d;
        sae += std::abs(d);
        sst += (t - meanT) * (t - meanT);
    }
    e.n = n;
    e.mse = sse / n;
    e.rmse = std::sqrt(e.mse);
    e.mae = sae / n;
    e.r2 = sst > 1e-12 ? 1.0 - sse / sst : 0.0;
    e.valid = true;
}

std::vector<int> allRows(const Dataset& ds, const std::vector<int>* rows) {
    if (rows) return *rows;
    std::vector<int> r(static_cast<size_t>(ds.sampleCount));
    for (int i = 0; i < ds.sampleCount; ++i) r[static_cast<size_t>(i)] = i;
    return r;
}
} // namespace

EvalResult evaluateNetwork(NeuralNetwork& net, const Dataset& ds, const std::vector<int>* rowsIn) {
    EvalResult e;
    e.task = net.taskType();
    const std::vector<int> rows = allRows(ds, rowsIn);
    if (rows.empty() || ds.outputCount < 1) return e;
    net.setTrainingMode(false);
    if (e.task == TaskType::BinaryClassification) {
        // true class: label above the middle of the two train labels (as BatchRunner / Evaluation panel)
        const double mid = 0.5 * (net.outputMin() + net.outputMax());
        e.confusion.assign(2, std::vector<int>(2, 0));
        for (int r : rows) {
            const int tc = ds.outputAt(r, 0) > mid ? 1 : 0;
            const auto p = net.predict(ds.inputRow(r));
            const int pc = p[0] >= 0.5 ? 1 : 0;
            ++e.confusion[tc][pc];
        }
        e.classLabels = {QString::number(net.outputMin(), 'g', 6), QString::number(net.outputMax(), 'g', 6)};
        finishClassification(e, 2, 1);
    } else if (e.task == TaskType::MultiClassClassification) {
        const int K = std::max(1, net.numClasses());
        e.confusion.assign(static_cast<size_t>(K), std::vector<int>(static_cast<size_t>(K), 0));
        for (int r : rows) {
            const int tc = net.classIndex(ds.outputAt(r, 0));
            const auto p = net.predict(ds.inputRow(r));
            const int pc = static_cast<int>(std::max_element(p.begin(), p.end()) - p.begin());
            if (tc >= 0 && tc < K && pc >= 0 && pc < K) ++e.confusion[tc][pc];
        }
        for (double v : net.classValues()) e.classLabels << QString::number(v, 'g', 6);
        finishClassification(e, K, 1);
    } else {
        finishRegression(e, ds, rows, [&](int r) { return net.predict(ds.inputRow(r))[0]; });
    }
    return e;
}

EvalResult evaluateRbf(const RBFNetwork& net, const Dataset& ds, const std::vector<int>* rowsIn) {
    EvalResult e;
    e.task = net.taskType();
    const std::vector<int> rows = allRows(ds, rowsIn);
    if (rows.empty() || ds.outputCount < 1) return e;
    if (e.task == TaskType::Regression) {
        finishRegression(e, ds, rows, [&](int r) { return net.predict(ds.inputs[static_cast<size_t>(r)])[0]; });
        return e;
    }
    const int K = std::max(2, static_cast<int>(net.classValues().size()));
    e.confusion.assign(static_cast<size_t>(K), std::vector<int>(static_cast<size_t>(K), 0));
    for (int r : rows) {
        const auto p = net.predict(ds.inputs[static_cast<size_t>(r)]);
        const int tc = net.classIndex(ds.outputs[static_cast<size_t>(r)][0]);   // index among the train class values
        int pc = 0;
        if (e.task == TaskType::BinaryClassification) pc = p[0] >= 0.5 ? 1 : 0;
        else pc = static_cast<int>(std::max_element(p.begin(), p.end()) - p.begin());
        if (tc >= 0 && tc < K && pc >= 0 && pc < K) ++e.confusion[tc][pc];
    }
    for (double v : net.classValues()) e.classLabels << QString::number(v, 'g', 6);
    while (e.classLabels.size() < K) e.classLabels << QString::number(e.classLabels.size());
    finishClassification(e, K, 1);
    return e;
}

QString describeEval(const QString& title, const EvalResult& e) {
    if (!e.valid) return title + ": no samples\n";
    QString s;
    if (e.task == TaskType::Regression) {
        s = QString("%1 (%2 samples): MSE %3   RMSE %4   MAE %5   R2 %6\n")
                .arg(title).arg(e.n).arg(fmtNum(e.mse)).arg(fmtNum(e.rmse)).arg(fmtNum(e.mae)).arg(fmtNum(e.r2));
        return s;
    }
    s = QString("%1 (%2 samples): accuracy %3 %   error %4 %\n")
            .arg(title).arg(e.n).arg(e.accuracy * 100, 0, 'f', 2).arg(e.errorPct, 0, 'f', 2);
    if (e.precision >= 0)
        s += QString("  precision %1   recall %2   F1 %3   (positive class = %4)\n")
                 .arg(e.precision, 0, 'f', 4).arg(e.recall, 0, 'f', 4).arg(e.f1, 0, 'f', 4)
                 .arg(e.classLabels.value(1, "1"));
    const int K = static_cast<int>(e.confusion.size());
    if (K > 0 && K <= 20) {
        s += "  confusion matrix (rows = true class, columns = predicted class):\n";
        int w = 5;
        for (const QString& l : e.classLabels) w = std::max(w, static_cast<int>(l.size()) + 1);
        for (const auto& row : e.confusion)
            for (int v : row) w = std::max(w, static_cast<int>(QString::number(v).size()) + 1);
        QString head = QString(w + 4, ' ');
        for (int p = 0; p < K; ++p) head += e.classLabels.value(p).rightJustified(w);
        s += head + "\n";
        for (int t = 0; t < K; ++t) {
            QString line = "    " + e.classLabels.value(t).rightJustified(w);
            for (int p = 0; p < K; ++p) line += QString::number(e.confusion[t][p]).rightJustified(w);
            s += line + "\n";
        }
    }
    return s;
}

void evalToSheet(XlsxSheet* sh, const QString& set, const EvalResult& e) {
    if (!sh || !e.valid) return;
    sh->addRow({XlsxCell(set), XlsxCell("samples"), XlsxCell(e.n)});
    if (e.task == TaskType::Regression) {
        sh->addRow({XlsxCell(set), XlsxCell("MSE"), XlsxCell(e.mse)});
        sh->addRow({XlsxCell(set), XlsxCell("RMSE"), XlsxCell(e.rmse)});
        sh->addRow({XlsxCell(set), XlsxCell("MAE"), XlsxCell(e.mae)});
        sh->addRow({XlsxCell(set), XlsxCell("R2"), XlsxCell(e.r2)});
        return;
    }
    sh->addRow({XlsxCell(set), XlsxCell("accuracy %"), XlsxCell(e.accuracy * 100.0)});
    sh->addRow({XlsxCell(set), XlsxCell("error %"), XlsxCell(e.errorPct)});
    if (e.precision >= 0) {
        sh->addRow({XlsxCell(set), XlsxCell("precision"), XlsxCell(e.precision)});
        sh->addRow({XlsxCell(set), XlsxCell("recall"), XlsxCell(e.recall)});
        sh->addRow({XlsxCell(set), XlsxCell("F1"), XlsxCell(e.f1)});
    }
    const int K = static_cast<int>(e.confusion.size());
    for (int t = 0; t < K; ++t)
        for (int p = 0; p < K; ++p)
            sh->addRow({XlsxCell(set),
                        XlsxCell(QString("confusion true=%1 predicted=%2").arg(e.classLabels.value(t), e.classLabels.value(p))),
                        XlsxCell(e.confusion[t][p])});
}

std::shared_ptr<NeuralNetwork> buildMlp(const Dataset& train, TaskType task, const std::vector<int>& hidden,
                                        const std::vector<Activation>& acts, double dropout) {
    NetworkConfig cfg;
    cfg.task = task;
    cfg.dropoutRate = dropout;
    cfg.hiddenSizes = hidden;
    cfg.hiddenActivations = acts;
    auto net = std::make_shared<NeuralNetwork>();
    const auto unique = train.uniqueOutputValues(0);
    const int nOut = task == TaskType::MultiClassClassification ? static_cast<int>(unique.size())
                   : task == TaskType::BinaryClassification     ? 1
                                                                : train.outputCount;
    if (task == TaskType::MultiClassClassification)
        net->setClassValues(std::vector<double>(unique.begin(), unique.end()));
    net->build(train.inputCount, nOut, cfg);
    net->setNormalization(train.inputStats);
    net->fitOutputScaling(&train);   // regression: standardised output layer
    if (unique.size() >= 2) net->setOutputMapping(*unique.begin(), *unique.rbegin());
    return net;
}

TaskType resolveTask(const QString& choice, const Dataset& train) {
    if (choice == "binary") return TaskType::BinaryClassification;
    if (choice == "multiclass") return TaskType::MultiClassClassification;
    if (choice == "regression") return TaskType::Regression;
    return detectTask(&train);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Output files
// ═══════════════════════════════════════════════════════════════════════════════
QString ensureSuffix(const QString& path, const QString& suffix) {
    if (path.endsWith("." + suffix, Qt::CaseInsensitive)) return path;
    return path + "." + suffix;
}

bool ensureParentDir(const QString& path, QString& error) {
    const QFileInfo fi(path);
    QDir d = fi.absoluteDir();
    if (d.exists()) return true;
    if (!QDir().mkpath(d.absolutePath())) { error = "Cannot create folder " + d.absolutePath(); return false; }
    return true;
}

QString sheetName(const QString& base) {
    QString s = base;
    for (QChar c : QString("[]:*?/\\")) s.replace(c, '_');
    if (s.size() > 31) s = s.left(31);
    return s.isEmpty() ? QString("Sheet") : s;
}

QVector<OptSpec> modelExportOptions() {
    const QString g = "Model files";
    return {
        {"save-model", "FILE", "", "save the trained network (.nsmodel, opens in the GUI and in 'predict')", g},
        {"export-cpp", "FILE", "", "export the network as a self-contained C++ header (.h)", g},
        {"cpp-namespace", "NAME", "NeuralStudioModel", "namespace of the exported C++ header", g},
        {"export-onnx", "FILE", "", "export the network in the ONNX format (.onnx)", g},
    };
}

bool writeModelExports(const Args& a, const NeuralNetwork& net) {
    auto& con = Console::get();
    bool ok = true;
    auto attempt = [&](const QString& opt, const std::function<void(const QString&)>& fn) {
        if (!a.has(opt)) return;
        const QString path = a.str(opt);
        QString err;
        if (!ensureParentDir(path, err)) { con.error(err); ok = false; return; }
        try {
            fn(path);
            con.info(QString("Written: %1").arg(QFileInfo(path).absoluteFilePath()));
        } catch (const std::exception& ex) {
            con.error(QString("--%1: %2").arg(opt, QString::fromUtf8(ex.what())));
            ok = false;
        }
    };
    attempt("save-model", [&](const QString& p) { ModelSerializer::save(net, p); });
    attempt("export-cpp", [&](const QString& p) {
        CppExporter::exportHeader(net, p, a.str("cpp-namespace", "NeuralStudioModel"));
    });
    attempt("export-onnx", [&](const QString& p) { OnnxExporter::save(net, p); });
    return ok;
}

QVector<OptSpec> commonOptions() {
    const QString g = "General";
    return {
        {"help", "", "", "show this help", g},
        {"quiet", "", "", "no progress output (results and errors only)", g},
        {"verbose", "", "", "also show the log messages of the library", g},
    };
}

} // namespace NeuralStudio::cli
