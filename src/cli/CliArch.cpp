// CliArch.cpp — Mode 5 (Architecture Search) of neuralstudio-cli.
//
// Runs the ArchSearchWorker of the Architecture Search panel: an outer evolutionary optimizer
// searches the number of layers, their sizes and activations, an inner optimizer (+ bounded
// L-BFGS polish) trains every candidate, a cache avoids training an architecture twice. The
// search selects on the validation split of the training file; the companion test file is
// scored only. With --runs N the search is repeated and the best architectures are compared.
#include "CliCommon.h"
#include "CliModes.h"

#include "core/arch/ArchSearchWorker.h"
#include "core/export/XlsxWriter.h"

#include <QElapsedTimer>
#include <QFileInfo>
#include <QMap>

namespace NeuralStudio::cli {

namespace {

QVector<OptSpec> archSpecs() {
    QVector<OptSpec> s = commonOptions();
    const QString gD = "Data", gS = "Search space", gO = "Outer optimizer (architectures)",
                  gI = "Inner optimizer (weights of every candidate)", gR = "Output";
    s += {
        {"data", "FILES", "", "training files, folders or patterns (companion test files are scored only)", gD},
        {"runs", "N", "1", "independent searches", gD},
        {"val-split", "F", "0.2", "validation fraction used to select the architecture", gD},
        {"min-layers", "N", "1", "fewest hidden layers", gS},
        {"max-layers", "N", "4", "most hidden layers", gS},
        {"sizes", "LIST", "4,8,16,32,64,128", "candidate layer sizes", gS},
        {"acts", "LIST", "relu,tanh,sigmoid", "candidate activations: relu, tanh, sigmoid, linear", gS},
        {"outer", "NAME", "ga", "outer optimizer", gO},
        {"outer-pop", "N", "30", "outer population", gO},
        {"outer-gens", "N", "50", "outer generations", gO},
        {"inner", "NAME", "de", "inner optimizer", gI},
        {"inner-pop", "N", "50", "inner population", gI},
        {"inner-iters", "N", "50", "inner iterations", gI},
        {"inner-bound", "B", "10", "weight bound of the inner training", gI},
        {"polish", "EVALS", "400", "bound-respecting L-BFGS evaluations after the inner optimizer (0 = off)", gI},
        {"topk", "K", "10", "architectures reported per dataset and run", gR},
        {"out", "FILE.xlsx", "", "write the top architectures of every run (and the convergence) to an .xlsx file", gR},
    };
    return s;
}

QString plainDesc(const QString& d) { return asciiText(d); }

struct ArchRun {
    int run;
    QString ds;
    bool cls;
    QVector<ArchResult> top;
    QVector<double> conv;
};

} // namespace

int cmdArch(const QStringList& argv) {
    auto& con = Console::get();
    Args a;
    QString err;
    const auto specs = archSpecs();
    if (!a.parse(argv, specs, err)) { con.error(err); con.info("Run 'neuralstudio-cli archsearch --help' for the options."); return kExitUsage; }
    if (a.flag("help")) {
        con.out("neuralstudio-cli archsearch  -  Mode 5, Architecture Search\n"
                "An outer evolutionary optimizer searches the number of hidden layers, their sizes and activation\n"
                "functions; an inner optimizer trains every candidate on the training part of the file and the\n"
                "validation error selects. The companion test file is scored for every reported architecture.\n\n"
                "usage: neuralstudio-cli archsearch --data FILES [options]\n"
                + formatOptions(specs) +
                "\nOuter optimizers: " + archOuterOptimizers().join(", ") +
                ".\nInner optimizers: " + phaseOptimizers().join(", ") + ".\n"
                "\nexample:\n"
                "  neuralstudio-cli archsearch --data wine.train --max-layers 3 --sizes 4,8,16,32 --outer-gens 30 --runs 5 --out arch.xlsx\n");
        return kExitOk;
    }
    ArchSearchConfig cfg;
    QStringList files;
    if (!collectDataFiles(a, files, err)) a.addError(err);
    cfg.datasetPaths = files;
    cfg.minLayers = a.integer("min-layers", 1, 1, 6);
    cfg.maxLayers = a.integer("max-layers", 4, 1, 6);
    if (cfg.maxLayers < cfg.minLayers) a.addError("--max-layers must not be smaller than --min-layers.");
    cfg.candidateSizes.clear();
    for (const QString& t : a.has("sizes") ? a.list("sizes") : QStringList{"4", "8", "16", "32", "64", "128"}) {
        bool ok = false;
        const int n = t.toInt(&ok);
        if (!ok || n < 1 || n > 4096) { a.addError(QString("--sizes: invalid size '%1'.").arg(t)); break; }
        if (!cfg.candidateSizes.contains(n)) cfg.candidateSizes << n;
    }
    cfg.candidateActs.clear();
    for (const QString& t : a.has("acts") ? a.list("acts") : QStringList{"relu", "tanh", "sigmoid"}) {
        Activation tmp;
        if (!parseActivation(t, tmp)) { a.addError(QString("--acts: unknown activation '%1'.").arg(t)); break; }
        if (!cfg.candidateActs.contains(t.toLower())) cfg.candidateActs << t.toLower();
    }
    if (cfg.candidateSizes.isEmpty()) a.addError("--sizes is empty.");
    if (cfg.candidateActs.isEmpty()) a.addError("--acts is empty.");
    cfg.valSplit = a.number("val-split", 0.2, 0.0, 0.9);
    cfg.topK = a.integer("topk", 10, 1, 1000);
    cfg.outerOpt = a.str("outer", "ga").trimmed().toLower();
    if (!archOuterOptimizers().contains(cfg.outerOpt))
        a.addError(QString("--outer: '%1' is not one of %2.").arg(cfg.outerOpt, archOuterOptimizers().join(", ")));
    cfg.outerPop = a.integer("outer-pop", 30, 5, 100000);
    cfg.outerGens = a.integer("outer-gens", 50, 1, 1000000);
    cfg.innerOpt = a.str("inner", "de").trimmed().toLower();
    if (!phaseOptimizers().contains(cfg.innerOpt))
        a.addError(QString("--inner: '%1' is not one of %2.").arg(cfg.innerOpt, phaseOptimizers().join(", ")));
    cfg.innerPop = a.integer("inner-pop", 50, 4, 100000);
    cfg.innerIters = a.integer("inner-iters", 50, 1, 10000000);
    cfg.innerWBound = a.number("inner-bound", 10.0, 1e-6, 1e6);
    cfg.innerPolishEvals = a.integer("polish", 400, 0, 100000000);
    const int runs = a.integer("runs", 1, 1, 1000);
    if (!a.ok()) { con.error(a.firstError()); return kExitUsage; }

    QString head = "NeuralStudio - Mode 5 (Architecture Search)\n";
    head += QString("Datasets      %1\n").arg([&] { QStringList n; for (const auto& f : files) n << QFileInfo(f).fileName(); return n.join(", "); }());
    QStringList sz; for (int s : cfg.candidateSizes) sz << QString::number(s);
    head += QString("Search space  %1-%2 hidden layers, sizes {%3}, activations {%4}\n")
                .arg(cfg.minLayers).arg(cfg.maxLayers)
                .arg(sz.join(", "), QStringList(cfg.candidateActs.begin(), cfg.candidateActs.end()).join(", "));
    head += QString("Outer         %1, population %2, generations %3\n").arg(optimizerLabel(cfg.outerOpt)).arg(cfg.outerPop).arg(cfg.outerGens);
    head += QString("Inner         %1, population %2, iterations %3, weight bound %4, L-BFGS polish %5 evaluations\n")
                .arg(optimizerLabel(cfg.innerOpt)).arg(cfg.innerPop).arg(cfg.innerIters).arg(fmtNum(cfg.innerWBound))
                .arg(cfg.innerPolishEvals);
    head += QString("Selection     validation split %1 of the training file, runs %2\n").arg(cfg.valSplit).arg(runs);
    con.out(head);

    StopHandle stop;
    QVector<ArchRun> results;
    QStringList errors;
    QElapsedTimer timer;
    timer.start();
    try {
        runInterruptible(stop, [&] {
            for (int run = 1; run <= runs && !stop.requested(); ++run) {
                ArchSearchWorker w;
                StopGuard<ArchSearchWorker> g(stop, &w);
                w.setConfig(cfg);
                QString ds;
                QObject::connect(&w, &ArchSearchWorker::datasetStarted, [&](int i, int n, QString name) {
                    ds = QString("run %1/%2  dataset %3/%4 %5").arg(run).arg(runs).arg(i + 1).arg(n).arg(name);
                    con.progress(ds);
                });
                QObject::connect(&w, &ArchSearchWorker::generationDone,
                                 [&](int gen, int total, double best, QString desc, int uniq, int hits) {
                    con.progress(QString("%1  generation %2/%3  best %4  %5  (%6 trained, %7 cache hits)")
                                     .arg(ds).arg(gen).arg(total).arg(fmtNum(best)).arg(plainDesc(desc)).arg(uniq).arg(hits));
                });
                QObject::connect(&w, &ArchSearchWorker::datasetFinished,
                                 [&](QString name, QVector<ArchResult> top, QVector<double> conv, bool cls) {
                    results.push_back({run, name, cls, top, conv});
                });
                QObject::connect(&w, &ArchSearchWorker::errorOccurred, [&](QString m) { errors << QString("run %1: %2").arg(run).arg(m); });
                w.run();
            }
            con.endProgress();
        });
    } catch (const std::exception& ex) {
        con.error(QString::fromUtf8(ex.what()));
        return kExitError;
    }
    for (const QString& e : errors) con.error(e);

    QString rep = QString("Finished in %1%2\n\n").arg(fmtDuration(timer.elapsed() / 1000.0))
                      .arg(stop.requested() ? "  (stopped by the user)" : "");
    QStringList dsOrder;
    for (const auto& r : results) if (!dsOrder.contains(r.ds)) dsOrder << r.ds;
    for (const QString& ds : dsOrder) {
        bool cls = false;
        QVector<const ArchRun*> rr;
        for (const auto& r : results) if (r.ds == ds) { rr << &r; cls = r.cls; }
        const QString unit = cls ? "err %" : "MSE";
        for (const ArchRun* r : rr) {
            rep += QString("Dataset %1 (%2), run %3 - top %4 architectures (ranked by validation %5)\n")
                       .arg(ds, cls ? "classification" : "regression").arg(r->run).arg(r->top.size()).arg(unit);
            QVector<QStringList> rows;
            for (const auto& x : r->top)
                rows.push_back({QString::number(x.rank), plainDesc(x.desc), QString::number(x.numParams),
                                fmtErr(x.trainErr, cls), fmtErr(x.valErr, cls), fmtErr(x.testErr, cls)});
            rep += textTable({"#", "Architecture", "Params", "Train " + unit, "Val " + unit, "Test " + unit}, rows) + "\n";
        }
        if (rr.size() > 1) {
            QVector<double> va, te, tr;
            QMap<QString, int> freq;
            for (const ArchRun* r : rr) {
                if (r->top.isEmpty()) continue;
                const auto& b = r->top.first();
                va << b.valErr;
                if (b.testErr >= 0) te << b.testErr;
                tr << b.trainErr;
                ++freq[plainDesc(b.desc)];
            }
            rep += QString("Dataset %1 - best architecture over %2 runs: train %3, validation %4, test %5 (%6)\n")
                       .arg(ds).arg(rr.size()).arg(fmtMeanSd(tr, cls), fmtMeanSd(va, cls), fmtMeanSd(te, cls), unit);
            QStringList f;
            for (auto it = freq.begin(); it != freq.end(); ++it) f << QString("%1 (%2x)").arg(it.key()).arg(it.value());
            rep += "  chosen: " + f.join("; ") + "\n\n";
        }
        // the best architecture of the last run as a Mode 1 command (the GUI's "Apply to Mode 1")
        const ArchRun* last = rr.isEmpty() ? nullptr : rr.last();
        if (last && !last->top.isEmpty()) {
            const auto& b = last->top.first();
            QString path;
            for (const QString& f : files) if (QFileInfo(f).baseName() == ds) { path = f; break; }
            QStringList h, ac;
            for (int s : b.sizes) h << QString::number(s);
            for (const QString& x : b.actNames) ac << x.toLower();
            rep += QString("To train it in Mode 1:  neuralstudio-cli train --data \"%1\" --hidden %2 --act %3\n\n")
                       .arg(path.isEmpty() ? ds : path, h.join(","), ac.join(","));
        }
    }
    con.out(rep.trimmed() + "\n");

    int rc = stop.requested() ? kExitInterrupted : (errors.isEmpty() ? kExitOk : kExitError);
    if (a.has("out") && !results.isEmpty()) {
        const QString p = ensureSuffix(a.str("out"), "xlsx");
        try {
            if (!ensureParentDir(p, err)) throw std::runtime_error(err.toStdString());
            XlsxWriter wb;
            auto* st = wb.addSheet("Top architectures");
            st->setHeader({"Run", "Dataset", "Task", "Rank", "Architecture", "Hidden sizes", "Activations", "Parameters",
                           "Train error", "Validation error", "Test error", "Train loss", "Evaluation #"});
            for (const auto& r : results)
                for (const auto& x : r.top) {
                    QStringList h, ac;
                    for (int s : x.sizes) h << QString::number(s);
                    for (const QString& n : x.actNames) ac << n;
                    st->addRow({XlsxCell(r.run), XlsxCell(r.ds), XlsxCell(r.cls ? "classification" : "regression"),
                                XlsxCell(x.rank), XlsxCell(plainDesc(x.desc)), XlsxCell(h.join(",")), XlsxCell(ac.join(",")),
                                XlsxCell(x.numParams), XlsxCell(x.trainErr), XlsxCell(x.valErr),
                                x.testErr >= 0 ? XlsxCell(x.testErr) : XlsxCell(), XlsxCell(x.trainLoss), XlsxCell(x.evalNum)});
                }
            auto* sc = wb.addSheet("Convergence");
            sc->setHeader({"Run", "Dataset", "Generation", "Best validation error"});
            for (const auto& r : results)
                for (int i = 0; i < r.conv.size(); ++i)
                    sc->addRow({XlsxCell(r.run), XlsxCell(r.ds), XlsxCell(i + 1), XlsxCell(r.conv[i])});
            wb.save(p);
            con.info("Written: " + QFileInfo(p).absoluteFilePath());
        } catch (const std::exception& ex) {
            con.error(QString("--out: %1").arg(QString::fromUtf8(ex.what())));
            rc = kExitError;
        }
    }
    return rc;
}

} // namespace NeuralStudio::cli
