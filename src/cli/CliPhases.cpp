// CliPhases.cpp — Mode 3 (Pre-Train + NNC) and Mode 4 (Evolutionary Bounds) of neuralstudio-cli.
//
// Both modes run their worker (PreTrainWorker / EvolBoundsWorker) once per run over the
// queued datasets, exactly as their panels do, and collect the comparison rows of every
// dataset (baseline and phases: training loss, training / validation / test error). With
// --runs N the rows are reported as mean +/- sample standard deviation over the runs. A run
// in which Phase 1+2 was rejected continues Phase 3 from the baseline weights; the row
// keeps the name "Phase 1+2+3" and a column says how often that happened (as the panels).
#include "CliCommon.h"
#include "CliModes.h"

#include "core/bounds/EvolBoundsWorker.h"
#include "core/export/XlsxWriter.h"
#include "core/pretrain/PreTrainWorker.h"

#include <QElapsedTimer>
#include <QFileInfo>
#include <QMap>

namespace NeuralStudio::cli {

namespace {

// ── shared accumulation of the comparison rows ────────────────────────────────
struct PhaseAcc {
    QStringList configs;
    bool cls = false;
    QVector<QVector<double>> trLoss, trErr, vaErr, teErr;
    QVector<int> fromStd;
};
struct PhaseRaw {
    int run;
    QString ds, config;
    double trLoss, trErr, vaErr, teErr;
};
struct BoundsRec {
    QString ds;
    QVector<double> lb, ub;
    double initial = 0;
};

class PhaseCollector {
public:
    void add(int run, const QString& ds, const QStringList& configs, const QVector<double>& trLoss,
             const QVector<double>& trErr, const QVector<double>& vaErr, const QVector<double>& teErr, bool cls) {
        if (!m_acc.contains(ds)) m_order << ds;
        PhaseAcc& a = m_acc[ds];
        if (a.configs.isEmpty()) {
            a.configs = configs;
            a.cls = cls;
            a.trLoss.resize(configs.size()); a.trErr.resize(configs.size());
            a.vaErr.resize(configs.size());  a.teErr.resize(configs.size());
            a.fromStd.resize(configs.size());
        }
        for (int c = 0; c < configs.size() && c < a.configs.size(); ++c) {
            if (configs[c].startsWith("Standard + Phase 3")) { ++a.fromStd[c]; a.configs[c] = "Phase 1+2+3"; }
            a.trLoss[c] << trLoss.value(c, 0.0);
            a.trErr[c]  << trErr.value(c, 0.0);
            a.vaErr[c]  << vaErr.value(c, -1.0);
            a.teErr[c]  << teErr.value(c, -1.0);
            m_raw.push_back({run, ds, configs[c], trLoss.value(c, 0.0), trErr.value(c, 0.0), vaErr.value(c, -1.0),
                             teErr.value(c, -1.0)});
        }
    }
    bool empty() const { return m_order.isEmpty(); }

    QString report() const {
        QString out;
        for (const QString& ds : m_order) {
            const PhaseAcc& a = m_acc[ds];
            const int n = a.trErr.isEmpty() ? 0 : a.trErr[0].size();
            const QString unit = a.cls ? "err %" : "MSE";
            out += QString("Dataset %1 (%2)%3\n").arg(ds, a.cls ? "classification" : "regression")
                       .arg(n > 1 ? QString(", %1 runs, mean +/- std").arg(n) : QString());
            bool anyVal = false, anyTest = false, anyStd = false;
            for (int c = 0; c < a.configs.size(); ++c) {
                for (double v : a.vaErr[c]) anyVal = anyVal || v >= 0;
                for (double v : a.teErr[c]) anyTest = anyTest || v >= 0;
                anyStd = anyStd || a.fromStd[c] > 0;
            }
            QStringList head{"Configuration", "Train " + unit};
            if (anyVal) head << "Val " + unit;
            if (anyTest) head << "Test " + unit;
            head << "Final loss";
            if (anyStd) head << "Phase 3 started from";
            QVector<QStringList> rows;
            for (int c = 0; c < a.configs.size(); ++c) {
                QStringList r{a.configs[c], fmtMeanSd(valid(a.trErr[c]), a.cls)};
                if (anyVal) r << fmtMeanSd(valid(a.vaErr[c]), a.cls);
                if (anyTest) r << fmtMeanSd(valid(a.teErr[c]), a.cls);
                r << fmtMeanSd(a.trLoss[c], false);
                if (anyStd) r << phase3From(a, c);
                rows.push_back(r);
            }
            out += textTable(head, rows) + "\n";
        }
        return out;
    }

    void toXlsx(XlsxWriter& wb) const {
        auto* sc = wb.addSheet("Comparison");
        sc->setHeader({"Dataset", "Task", "Configuration", "Runs", "Train error mean", "Train error std",
                       "Validation error mean", "Validation error std", "Test error mean", "Test error std",
                       "Final loss mean", "Final loss std", "Phase 3 started from", "Error unit"});
        auto mc = [](const QVector<double>& v) { return v.isEmpty() ? XlsxCell() : XlsxCell(mean(v)); };
        auto sd = [](const QVector<double>& v) { return v.size() < 2 ? XlsxCell() : XlsxCell(sampleSd(v)); };
        for (const QString& ds : m_order) {
            const PhaseAcc& a = m_acc[ds];
            for (int c = 0; c < a.configs.size(); ++c) {
                const auto tr = valid(a.trErr[c]), va = valid(a.vaErr[c]), te = valid(a.teErr[c]);
                sc->addRow({XlsxCell(ds), XlsxCell(a.cls ? "classification" : "regression"), XlsxCell(a.configs[c]),
                            XlsxCell(static_cast<int>(a.trErr[c].size())), mc(tr), sd(tr), mc(va), sd(va), mc(te), sd(te),
                            mc(a.trLoss[c]), sd(a.trLoss[c]), XlsxCell(phase3From(a, c)),
                            XlsxCell(a.cls ? "error %" : "MSE")});
            }
        }
        auto* sr = wb.addSheet("Runs");
        sr->setHeader({"Run", "Dataset", "Configuration", "Final loss", "Train error", "Validation error", "Test error"});
        for (const auto& r : m_raw)
            sr->addRow({XlsxCell(r.run), XlsxCell(r.ds), XlsxCell(r.config), XlsxCell(r.trLoss), XlsxCell(r.trErr),
                        r.vaErr >= 0 ? XlsxCell(r.vaErr) : XlsxCell(), r.teErr >= 0 ? XlsxCell(r.teErr) : XlsxCell()});
    }

private:
    static QVector<double> valid(const QVector<double>& v) {
        QVector<double> o;
        for (double x : v) if (x >= 0 && std::isfinite(x)) o << x;
        return o;
    }
    static QString phase3From(const PhaseAcc& a, int c) {
        if (a.configs.value(c) != "Phase 1+2+3") return "-";
        const int k = a.fromStd.value(c), n = a.trErr[c].size();
        return k == 0 ? QString("Phase 1+2") : k == n ? QString("Standard (Phase 1+2 rejected)")
                                                      : QString("Standard in %1 of %2 runs").arg(k).arg(n);
    }
    QStringList m_order;
    QMap<QString, PhaseAcc> m_acc;
    std::vector<PhaseRaw> m_raw;
};

QString hiddenString(const QString& s) { return s.trimmed(); }

// ── Mode 3 options ────────────────────────────────────────────────────────────
QVector<OptSpec> pretrainSpecs() {
    QVector<OptSpec> s = commonOptions();
    const QString gD = "Data and network", gP = "Phases", gO = "Output";
    s += {
        {"data", "FILES", "", "training files, folders or patterns (companion test files are scored only)", gD},
        {"runs", "N", "1", "independent runs (mean +/- std over the runs)", gD},
        {"hidden", "SIZES", "16", "hidden layers of the main network, e.g. 64 or 64,32", gD},
        {"act", "relu|sigmoid|tanh|linear", "relu", "activation of the hidden layers", gD},
        {"val-split", "F", "0.2", "validation fraction of the training file", gD},
        {"iw", "N", "10", "hidden nodes of the small Phase-1 network", gD},
        {"equal-budget", "1|0", "1", "1: the baseline gets the evaluations of all phases; 0: the Phase-2 budget only", gD},
        {"p1-opt", "NAME", "ga", "Phase 1 optimizer (pre-trains the small network)", gP},
        {"p1-pop", "N", "200", "Phase 1 population", gP},
        {"p1-iters", "N", "200", "Phase 1 iterations", gP},
        {"p1-bound", "B", "10", "Phase 1 weight bound", gP},
        {"p2-opt", "NAME", "ga", "Phase 2 optimizer (main network, warm start from Phase 1)", gP},
        {"p2-pop", "N", "200", "Phase 2 population", gP},
        {"p2-iters", "N", "200", "Phase 2 iterations", gP},
        {"p2-bound", "B", "10", "Phase 2 weight bound", gP},
        {"p3-opt", "NAME|none", "none", "Phase 3 optimizer (optional further training)", gP},
        {"p3-pop", "N", "200", "Phase 3 population", gP},
        {"p3-iters", "N", "200", "Phase 3 iterations", gP},
        {"p3-bound", "B", "10", "Phase 3 weight bound", gP},
        {"out", "FILE.xlsx", "", "write the comparison (and every run) to an .xlsx file", gO},
    };
    s += modelExportOptions();
    return s;
}

// ── Mode 4 options ────────────────────────────────────────────────────────────
QVector<OptSpec> boundsSpecs() {
    QVector<OptSpec> s = commonOptions();
    const QString gD = "Data and network", g1 = "Phase 1 (bounds discovery)", g2 = "Phase 2 (training inside the bounds)",
                  g3 = "Phase 3 (final training)", gO = "Output";
    s += {
        {"data", "FILES", "", "training files, folders or patterns (companion test files are scored only)", gD},
        {"runs", "N", "1", "independent runs (mean +/- std over the runs)", gD},
        {"hidden", "SIZES", "16", "hidden layers, e.g. 16 or 64,32", gD},
        {"act", "relu|sigmoid|tanh|linear", "relu", "activation of the hidden layers", gD},
        {"val-split", "F", "0.2", "validation fraction of the training file", gD},
        {"equal-budget", "1|0", "1", "1: the Standard baseline gets the evaluations of all phases; 0: the Phase-2 budget", gD},
        {"opt1", "NAME", "ga", "penalised optimizer of Phase 1", g1},
        {"nc1", "N", "500", "Phase 1 population", g1},
        {"ng1", "N", "200", "Phase 1 generations", g1},
        {"lambda", "L", "2", "penalty multiplier (> 1)", g1},
        {"a", "A", "5", "saturation threshold", g1},
        {"f", "F", "3", "bound factor: lb_i = -f|g*_i|, ub_i = f|g*_i|", g1},
        {"initial-bound", "B", "10", "symmetric search range of Phase 1 and of the baseline", g1},
        {"opt2", "NAME|ge_grammar|none", "ga", "Phase 2 optimizer, or ge_grammar (BDCC 2025)", g2},
        {"nc2", "N", "200", "Phase 2 population", g2},
        {"ng2", "N", "200", "Phase 2 generations", g2},
        {"ns-ge", "N", "30", "ge_grammar: random networks per interval", g2},
        {"nl-ge", "N", "10", "ge_grammar: local search", g2},
        {"pl-ge", "P", "0.05", "ge_grammar: local search rate", g2},
        {"opt3", "NAME|none", "none", "Phase 3 optimizer", g3},
        {"nc3", "N", "500", "Phase 3 population", g3},
        {"ng3", "N", "200", "Phase 3 generations", g3},
        {"bfgs", "1|0", "1", "BFGS polish after Phase 3", g3},
        {"out", "FILE.xlsx", "", "write the comparison, every run and the parameter bounds to an .xlsx file", gO},
    };
    s += modelExportOptions();
    return s;
}

QString checkOpt(const Args& a, const QString& name, const QString& def, bool allowNone, bool allowGE) {
    const QString v = a.str(name, def).trimmed().toLower();
    if ((allowNone && v == "none") || (allowGE && v == "ge_grammar") || phaseOptimizers().contains(v)) return v;
    QStringList ok = phaseOptimizers();
    if (allowNone) ok << "none";
    if (allowGE) ok << "ge_grammar";
    a.addError(QString("--%1: unknown optimizer '%2'. Choose from: %3.").arg(name, v, ok.join(", ")));
    return def;
}

QString phaseLine(const QString& name, const QString& opt, int pop, int it, double bound) {
    if (opt == "none") return QString("%1  none (skipped)").arg(name);
    return QString("%1  %2, population %3, iterations %4, weight bound %5")
        .arg(name, optimizerLabel(opt)).arg(pop).arg(it).arg(fmtNum(bound));
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════════
// Mode 3
// ═══════════════════════════════════════════════════════════════════════════════
int cmdPretrain(const QStringList& argv) {
    auto& con = Console::get();
    Args a;
    QString err;
    const auto specs = pretrainSpecs();
    if (!a.parse(argv, specs, err)) { con.error(err); con.info("Run 'neuralstudio-cli pretrain --help' for the options."); return kExitUsage; }
    if (a.flag("help")) {
        con.out("neuralstudio-cli pretrain  -  Mode 3, Pre-Train + NNC (Symmetry 2025)\n"
                "Phase 1 pre-trains a small network (Iw hidden nodes), which is transferred neuron by neuron into the\n"
                "main network; Phase 2 trains the main network from that start; Phase 3 (optional) trains further.\n"
                "A baseline (the main network without pre-training) is trained for comparison.\n\n"
                "usage: neuralstudio-cli pretrain --data FILES [options]\n"
                + formatOptions(specs) +
                "\nOptimizers: " + phaseOptimizers().join(", ") + ".\n"
                "--save-model / --export-* write the final network of the last dataset of the last run.\n"
                "\nexample:\n"
                "  neuralstudio-cli pretrain --data wine.train,pima.train --hidden 16 --iw 6 --p1-opt de --p2-opt de --runs 10 --out pretrain.xlsx\n");
        return kExitOk;
    }
    PreTrainConfig cfg;
    QStringList files;
    if (!collectDataFiles(a, files, err)) a.addError(err);
    cfg.datasetPaths = files;
    std::vector<int> hs;
    if (!parseHidden(a.str("hidden", "16"), hs, err)) a.addError(err);
    else if (hs.empty()) a.addError("--hidden: at least one hidden layer.");
    cfg.hiddenLayout = hiddenString(a.str("hidden", "16"));
    {
        Activation tmp;
        cfg.activation = a.str("act", "relu").trimmed().toLower();
        if (!parseActivation(cfg.activation, tmp)) a.addError("--act: relu, sigmoid, tanh or linear.");
    }
    cfg.valSplit = a.number("val-split", 0.2, 0.0, 0.9);
    cfg.Iw = a.integer("iw", 10, 1, 4096);
    cfg.equalBudgetBaseline = a.boolean("equal-budget", true);
    auto phase = [&](const QString& p, bool allowNone, const QString& defOpt) {
        PhaseConfig pc;
        pc.optimizer = checkOpt(a, p + "-opt", defOpt, allowNone, false);
        pc.population = a.integer(p + "-pop", 200, 10, 1000000);
        pc.iterations = a.integer(p + "-iters", 200, 1, 10000000);
        pc.wBound = a.number(p + "-bound", 10.0, 1e-6, 1e6);
        return pc;
    };
    cfg.phase1 = phase("p1", false, "ga");
    cfg.phase2 = phase("p2", false, "ga");
    cfg.phase3 = phase("p3", true, "none");
    const int runs = a.integer("runs", 1, 1, 1000);
    if (!a.ok()) { con.error(a.firstError()); return kExitUsage; }

    QString head = "NeuralStudio - Mode 3 (Pre-Train + NNC)\n";
    head += QString("Datasets  %1\n").arg([&] { QStringList n; for (const auto& f : files) n << QFileInfo(f).fileName(); return n.join(", "); }());
    head += QString("Network   hidden %1, %2, validation %3, Iw %4, runs %5\n")
                .arg(cfg.hiddenLayout, cfg.activation).arg(cfg.valSplit).arg(cfg.Iw).arg(runs);
    head += phaseLine("Phase 1 ", cfg.phase1.optimizer, cfg.phase1.population, cfg.phase1.iterations, cfg.phase1.wBound) + "\n";
    head += phaseLine("Phase 2 ", cfg.phase2.optimizer, cfg.phase2.population, cfg.phase2.iterations, cfg.phase2.wBound) + "\n";
    head += phaseLine("Phase 3 ", cfg.phase3.optimizer, cfg.phase3.population, cfg.phase3.iterations, cfg.phase3.wBound) + "\n";
    head += QString("Baseline  %1\n").arg(cfg.equalBudgetBaseline ? "same total number of evaluations as all phases"
                                                                  : "Phase-2 budget only");
    con.out(head);

    StopHandle stop;
    PhaseCollector col;
    std::shared_ptr<NeuralNetwork> lastNet;
    QStringList errors;
    QElapsedTimer timer;
    timer.start();
    try {
        runInterruptible(stop, [&] {
            for (int run = 1; run <= runs && !stop.requested(); ++run) {
                PreTrainWorker w;
                StopGuard<PreTrainWorker> g(stop, &w);
                w.setConfig(cfg);
                QString ds, ph;
                QObject::connect(&w, &PreTrainWorker::datasetStarted, [&](int i, int n, QString name) {
                    ds = QString("run %1/%2  dataset %3/%4 %5").arg(run).arg(runs).arg(i + 1).arg(n).arg(name);
                    con.progress(ds);
                });
                QObject::connect(&w, &PreTrainWorker::phaseStarted, [&](int, int, QString name) { ph = name; con.progress(ds + "  " + ph); });
                QObject::connect(&w, &PreTrainWorker::iterationDone, [&](int, int it, int total, double loss, double) {
                    con.progress(QString("%1  %2  iteration %3/%4  loss %5").arg(ds, ph).arg(it).arg(total).arg(fmtNum(loss)));
                });
                QObject::connect(&w, &PreTrainWorker::datasetFinished,
                                 [&](QString name, QStringList c, QVector<double> tl, QVector<double> te, QVector<double> ve,
                                     QVector<double> tt, bool cls) { col.add(run, name, c, tl, te, ve, tt, cls); });
                QObject::connect(&w, &PreTrainWorker::allDone, [&](std::shared_ptr<NeuralNetwork> n, std::shared_ptr<Dataset>) {
                    if (n) lastNet = n;
                });
                QObject::connect(&w, &PreTrainWorker::errorOccurred, [&](QString m) { errors << QString("run %1: %2").arg(run).arg(m); });
                w.run();
            }
            con.endProgress();
        });
    } catch (const std::exception& ex) {
        con.error(QString::fromUtf8(ex.what()));
        return kExitError;
    }
    for (const QString& e : errors) con.error(e);
    con.out(QString("Finished in %1%2\n").arg(fmtDuration(timer.elapsed() / 1000.0))
                .arg(stop.requested() ? "  (stopped by the user: completed datasets only)" : ""));
    if (!col.empty()) con.out(col.report().trimmed() + "\n");

    int rc = stop.requested() ? kExitInterrupted : (errors.isEmpty() ? kExitOk : kExitError);
    if (a.has("out") && !col.empty()) {
        const QString p = ensureSuffix(a.str("out"), "xlsx");
        try {
            if (!ensureParentDir(p, err)) throw std::runtime_error(err.toStdString());
            XlsxWriter wb;
            col.toXlsx(wb);
            wb.save(p);
            con.info("Written: " + QFileInfo(p).absoluteFilePath());
        } catch (const std::exception& ex) {
            con.error(QString("--out: %1").arg(QString::fromUtf8(ex.what())));
            rc = kExitError;
        }
    }
    if (a.has("save-model") || a.has("export-cpp") || a.has("export-onnx")) {
        if (lastNet) { if (!writeModelExports(a, *lastNet)) rc = kExitError; }
        else con.warn("No final network to save.");
    }
    return rc;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Mode 4
// ═══════════════════════════════════════════════════════════════════════════════
int cmdBounds(const QStringList& argv) {
    auto& con = Console::get();
    Args a;
    QString err;
    const auto specs = boundsSpecs();
    if (!a.parse(argv, specs, err)) { con.error(err); con.info("Run 'neuralstudio-cli bounds --help' for the options."); return kExitUsage; }
    if (a.flag("help")) {
        con.out("neuralstudio-cli bounds  -  Mode 4, Evolutionary Bounds (Foundations 2025 / BDCC 2025)\n"
                "Phase 1: a penalised optimizer finds per-parameter weight bounds. Phase 2: the network is trained\n"
                "inside them (or GE bound programs refine them). Phase 3: optional final training (+ BFGS polish).\n"
                "A Standard baseline is trained for comparison.\n\n"
                "usage: neuralstudio-cli bounds --data FILES [options]\n"
                + formatOptions(specs) +
                "\nOptimizers: " + phaseOptimizers().join(", ") + ".\n"
                "--save-model / --export-* write the final network of the last dataset of the last run.\n"
                "\nexample:\n"
                "  neuralstudio-cli bounds --data sample_data/selected --opt2 de --opt3 ga --runs 10 --out bounds.xlsx\n");
        return kExitOk;
    }
    EvolBoundsConfig cfg;
    QStringList files;
    if (!collectDataFiles(a, files, err)) a.addError(err);
    cfg.datasetPaths = files;
    std::vector<int> hs;
    if (!parseHidden(a.str("hidden", "16"), hs, err)) a.addError(err);
    else if (hs.empty()) a.addError("--hidden: at least one hidden layer.");
    cfg.hiddenLayout = hiddenString(a.str("hidden", "16"));
    {
        Activation tmp;
        cfg.activation = a.str("act", "relu").trimmed().toLower();
        if (!parseActivation(cfg.activation, tmp)) a.addError("--act: relu, sigmoid, tanh or linear.");
    }
    cfg.valSplit = a.number("val-split", 0.2, 0.0, 0.9);
    cfg.equalBudgetBaseline = a.boolean("equal-budget", true);
    cfg.opt1 = checkOpt(a, "opt1", "ga", false, false);
    cfg.Nc1 = a.integer("nc1", 500, 10, 1000000);
    cfg.Ng1 = a.integer("ng1", 200, 1, 10000000);
    cfg.lambda = a.number("lambda", 2.0, 1.0 + 1e-9, 1e6);
    cfg.a = a.number("a", 5.0, 1e-6, 1e6);
    cfg.f = a.number("f", 3.0, 1.0 + 1e-9, 1e6);
    cfg.initialBound = a.number("initial-bound", 10.0, 1e-6, 1e6);
    cfg.opt2 = checkOpt(a, "opt2", "ga", true, true);
    cfg.Nc2 = a.integer("nc2", 200, 10, 1000000);
    cfg.Ng2 = a.integer("ng2", 200, 1, 10000000);
    cfg.Ns_ge = a.integer("ns-ge", 30, 1, 100000);
    cfg.Nl_ge = a.integer("nl-ge", 10, 1, 100000);
    cfg.pl_ge = a.number("pl-ge", 0.05, 0.0, 1.0);
    cfg.opt3 = checkOpt(a, "opt3", "none", true, false);
    cfg.Nc3 = a.integer("nc3", 500, 10, 1000000);
    cfg.Ng3 = a.integer("ng3", 200, 1, 10000000);
    cfg.bfgsEnd = a.boolean("bfgs", true);
    const int runs = a.integer("runs", 1, 1, 1000);
    if (!a.ok()) { con.error(a.firstError()); return kExitUsage; }

    QString head = "NeuralStudio - Mode 4 (Evolutionary Bounds)\n";
    head += QString("Datasets  %1\n").arg([&] { QStringList n; for (const auto& f : files) n << QFileInfo(f).fileName(); return n.join(", "); }());
    head += QString("Network   hidden %1, %2, validation %3, runs %4\n").arg(cfg.hiddenLayout, cfg.activation).arg(cfg.valSplit).arg(runs);
    head += QString("Phase 1   %1, population %2, generations %3, lambda %4, a %5, f %6, initial bound %7\n")
                .arg(optimizerLabel(cfg.opt1)).arg(cfg.Nc1).arg(cfg.Ng1).arg(cfg.lambda).arg(cfg.a).arg(cfg.f).arg(cfg.initialBound);
    if (cfg.opt2 == "none") head += "Phase 2   none (skipped)\n";
    else if (cfg.opt2 == "ge_grammar")
        head += QString("Phase 2   GE grammar bounds, Ns %1, Nl %2, pl %3\n").arg(cfg.Ns_ge).arg(cfg.Nl_ge).arg(cfg.pl_ge);
    else head += QString("Phase 2   %1, population %2, generations %3\n").arg(optimizerLabel(cfg.opt2)).arg(cfg.Nc2).arg(cfg.Ng2);
    if (cfg.opt3 == "none") head += "Phase 3   none (skipped)\n";
    else head += QString("Phase 3   %1, population %2, generations %3, BFGS polish %4\n")
                     .arg(optimizerLabel(cfg.opt3)).arg(cfg.Nc3).arg(cfg.Ng3).arg(cfg.bfgsEnd ? "on" : "off");
    head += QString("Baseline  %1\n").arg(cfg.equalBudgetBaseline ? "same total number of evaluations as all phases"
                                                                  : "Phase-2 budget only");
    con.out(head);

    StopHandle stop;
    PhaseCollector col;
    std::shared_ptr<NeuralNetwork> lastNet;
    QVector<BoundsRec> bounds;   // of the last run
    QStringList errors;
    QElapsedTimer timer;
    timer.start();
    try {
        runInterruptible(stop, [&] {
            for (int run = 1; run <= runs && !stop.requested(); ++run) {
                EvolBoundsWorker w;
                StopGuard<EvolBoundsWorker> g(stop, &w);
                w.setConfig(cfg);
                QString ds, ph;
                QVector<BoundsRec> runBounds;
                QObject::connect(&w, &EvolBoundsWorker::datasetStarted, [&](int i, int n, QString name) {
                    ds = QString("run %1/%2  dataset %3/%4 %5").arg(run).arg(runs).arg(i + 1).arg(n).arg(name);
                    con.progress(ds);
                });
                QObject::connect(&w, &EvolBoundsWorker::phaseStarted, [&](int, int, QString name) { ph = name; con.progress(ds + "  " + ph); });
                QObject::connect(&w, &EvolBoundsWorker::iterationDone, [&](int, int it, int total, double loss, double) {
                    con.progress(QString("%1  %2  iteration %3/%4  loss %5").arg(ds, ph).arg(it).arg(total).arg(fmtNum(loss)));
                });
                QObject::connect(&w, &EvolBoundsWorker::datasetBoundsReady,
                                 [&](QString name, QVector<double> lb, QVector<double> ub, double ib) { runBounds.push_back({name, lb, ub, ib}); });
                QObject::connect(&w, &EvolBoundsWorker::datasetFinished,
                                 [&](QString name, QStringList c, QVector<double> tl, QVector<double> te, QVector<double> ve,
                                     QVector<double> tt, bool cls) { col.add(run, name, c, tl, te, ve, tt, cls); });
                QObject::connect(&w, &EvolBoundsWorker::allDone, [&](std::shared_ptr<NeuralNetwork> n, std::shared_ptr<Dataset>) {
                    if (n) lastNet = n;
                });
                QObject::connect(&w, &EvolBoundsWorker::errorOccurred, [&](QString m) { errors << QString("run %1: %2").arg(run).arg(m); });
                w.run();
                if (!runBounds.isEmpty()) bounds = runBounds;
            }
            con.endProgress();
        });
    } catch (const std::exception& ex) {
        con.error(QString::fromUtf8(ex.what()));
        return kExitError;
    }
    for (const QString& e : errors) con.error(e);
    con.out(QString("Finished in %1%2\n").arg(fmtDuration(timer.elapsed() / 1000.0))
                .arg(stop.requested() ? "  (stopped by the user: completed datasets only)" : ""));
    if (!col.empty()) con.out(col.report().trimmed() + "\n");
    for (const auto& b : bounds) {
        if (b.lb.isEmpty()) continue;
        double wMin = 1e300, wMax = 0, wSum = 0;
        for (int i = 0; i < b.lb.size() && i < b.ub.size(); ++i) {
            const double w = b.ub[i] - b.lb[i];
            wMin = std::min(wMin, w); wMax = std::max(wMax, w); wSum += w;
        }
        con.out(QString("Phase-1 bounds of %1 (last run): %2 parameters, interval width min %3 / mean %4 / max %5 "
                        "(initial interval width %6)")
                    .arg(b.ds).arg(b.lb.size()).arg(fmtNum(wMin)).arg(fmtNum(wSum / b.lb.size())).arg(fmtNum(wMax))
                    .arg(fmtNum(2 * b.initial)));
    }

    int rc = stop.requested() ? kExitInterrupted : (errors.isEmpty() ? kExitOk : kExitError);
    if (a.has("out") && !col.empty()) {
        const QString p = ensureSuffix(a.str("out"), "xlsx");
        try {
            if (!ensureParentDir(p, err)) throw std::runtime_error(err.toStdString());
            XlsxWriter wb;
            col.toXlsx(wb);
            if (!bounds.isEmpty()) {
                auto* sb = wb.addSheet("Bounds");
                sb->setHeader({"Dataset", "Parameter", "Lower bound", "Upper bound", "Initial bound"});
                for (const auto& b : bounds)
                    for (int i = 0; i < b.lb.size() && i < b.ub.size(); ++i)
                        sb->addRow({XlsxCell(b.ds), XlsxCell(i + 1), XlsxCell(b.lb[i]), XlsxCell(b.ub[i]), XlsxCell(b.initial)});
            }
            wb.save(p);
            con.info("Written: " + QFileInfo(p).absoluteFilePath());
        } catch (const std::exception& ex) {
            con.error(QString("--out: %1").arg(QString::fromUtf8(ex.what())));
            rc = kExitError;
        }
    }
    if (a.has("save-model") || a.has("export-cpp") || a.has("export-onnx")) {
        if (lastNet) { if (!writeModelExports(a, *lastNet)) rc = kExitError; }
        else con.warn("No final network to save.");
    }
    return rc;
}

} // namespace NeuralStudio::cli
