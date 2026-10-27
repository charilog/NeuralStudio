// CliBatch.cpp — Mode 2 (Batch Run) of neuralstudio-cli.
//
// Same engine as the Batch panel: one BatchRunner job per (dataset, optimizer), repeated
// --runs times; the network template and training settings are shared by all jobs. The
// statistics use the shared matrix builder of the Wilcoxon / Friedman tabs
// (ui/panels/BatchMetricMatrix.h: task-based validity, repeats and folds name_1..name_K
// averaged into one value per dataset), Holm-adjusted pairwise Wilcoxon signed-rank tests
// and the Friedman test with the Nemenyi post-hoc analysis.
#include "CliCommon.h"
#include "CliModes.h"

#include "core/batch/BatchRunner.h"
#include "core/export/XlsxWriter.h"
#include "core/stats/Statistics.h"
#include "ui/panels/BatchMetricMatrix.h"

#include <QElapsedTimer>
#include <QFileInfo>
#include <QMap>

#include <algorithm>
#include <cmath>

namespace NeuralStudio::cli {

namespace {

QVector<OptSpec> batchSpecs() {
    QVector<OptSpec> s = commonOptions();
    const QString gD = "Data and optimizers", gN = "Network and training", gS = "Statistics", gO = "Output";
    s += {
        {"data", "FILES", "", "training files, folders or patterns (comma separated or repeated; also positional)", gD},
        {"optimizers", "LIST", "adam,sgd", "optimizers to compare: names, or all | gradient | meta", gD},
        {"runs", "N", "1", "independent runs of every (dataset, optimizer) job", gD},
        {"hidden", "SIZES", "64,32", "hidden layer sizes (same network for every job)", gN},
        {"act", "relu|tanh|sigmoid", "relu", "activation of the hidden layers", gN},
        {"dropout", "P", "0", "dropout rate", gN},
        {"epochs", "N", "200", "epochs (gradient) or iterations (other methods)", gN},
        {"lr", "RATE", "0.001", "learning rate (gradient optimizers)", gN},
        {"batch", "N", "32", "mini-batch size (gradient) or population size (other methods)", gN},
        {"val-split", "F", "0.2", "validation fraction of every training file", gN},
        {"patience", "N", "20", "early-stopping patience (0 = off)", gN},
        {"weight-bound", "B", "5", "weight interval [-B, B] of the non-gradient methods", gN},
        {"stat-metrics", "LIST", "auto", "metrics tested: auto | none | testErrorPct, valErrorPct, trainErrorPct, "
                                         "testRMSE, valRMSE, testMAE, valMAE, trainMAE, testR2, valR2", gS},
        {"stat-task", "all|binary|multiclass|regression", "all", "only datasets of this task enter the tests", gS},
        {"out", "FILE.xlsx", "", "write all runs, the summary and the statistical tests to an .xlsx file", gO},
    };
    return s;
}

struct RunResult {
    int run = 0;
    BatchJobResult r;
};

const QStringList kMetricKeys = {"testErrorPct", "valErrorPct", "trainErrorPct", "testRMSE", "valRMSE",
                                 "testMAE", "valMAE", "trainMAE", "testR2", "valR2"};

QString metricLabel(const QString& k) {
    static const QMap<QString, QString> m = {
        {"testErrorPct", "Test error %"}, {"valErrorPct", "Validation error %"}, {"trainErrorPct", "Train error %"},
        {"testRMSE", "Test RMSE"}, {"valRMSE", "Validation RMSE"}, {"testMAE", "Test MAE"}, {"valMAE", "Validation MAE"},
        {"trainMAE", "Train MAE"}, {"testR2", "Test R2"}, {"valR2", "Validation R2"}};
    return m.value(k, k);
}

QString starsOf(double p) {
    return p < 0.001 ? "***" : p < 0.01 ? "**" : p < 0.05 ? "*" : "ns";
}

void printHelpBatch() {
    Console::get().out(
        "neuralstudio-cli batch  -  Mode 2, Batch Run\n"
        "Train the same network with several optimizers on several datasets, repeat the runs,\n"
        "and compare the optimizers with Holm-adjusted Wilcoxon signed-rank tests and the\n"
        "Friedman test with the Nemenyi post-hoc analysis.\n\n"
        "usage: neuralstudio-cli batch --data FILES --optimizers LIST [options]\n"
        + formatOptions(batchSpecs()) +
        "\nThe folds name_1 ... name_K of one dataset and repeated runs are averaged into one value per\n"
        "dataset before testing (as in the Wilcoxon / Friedman tabs). The Friedman test needs at least\n"
        "three optimizers, every test at least two datasets with a value for every optimizer.\n"
        "\nexamples:\n"
        "  neuralstudio-cli batch --data sample_data/selected --optimizers adam,sgd,de,pso --runs 5 --out cmp.xlsx\n"
        "  neuralstudio-cli batch wine.train pima.train housing.train --optimizers gradient --epochs 300\n");
}

} // namespace

int cmdBatch(const QStringList& argv) {
    auto& con = Console::get();
    Args a;
    QString err;
    const auto specs = batchSpecs();
    if (!a.parse(argv, specs, err)) { con.error(err); con.info("Run 'neuralstudio-cli batch --help' for the options."); return kExitUsage; }
    if (a.flag("help")) { printHelpBatch(); return kExitOk; }

    QStringList files, optims;
    if (!collectDataFiles(a, files, err)) a.addError(err);
    if (!expandOptimizerList(a.has("optimizers") ? a.list("optimizers") : QStringList{"adam", "sgd"},
                             allTrainingOptimizers(), optims, err))
        a.addError(err);
    const int runs = a.integer("runs", 1, 1, 1000);
    std::vector<int> hidden;
    if (!parseHidden(a.str("hidden", "64,32"), hidden, err)) a.addError(err);
    if (hidden.empty()) hidden = {64, 32};   // as the Batch panel
    Activation act = Activation::ReLU;
    if (a.has("act") && !parseActivation(a.str("act"), act)) a.addError("--act: relu, tanh, sigmoid or linear.");
    const double dropout = a.number("dropout", 0.0, 0.0, 0.95);
    const int epochs = a.integer("epochs", 200, 1, 1000000);
    const double lr = a.number("lr", 0.001, 1e-9, 10.0);
    const int batch = a.integer("batch", 32, 1, 1000000);
    const double valSplit = a.number("val-split", 0.2, 0.0, 0.9);
    const int patience = a.integer("patience", 20, 0, 1000000);
    const double wBound = a.number("weight-bound", 5.0, 1e-6, 1e6);
    const QString statTask = a.choice("stat-task", "all", {"all", "binary", "multiclass", "regression"});
    QStringList statMetrics;
    {
        const QStringList req = a.has("stat-metrics") ? a.list("stat-metrics") : QStringList{"auto"};
        for (const QString& m : req) {
            if (m.toLower() == "auto" || m.toLower() == "none") { statMetrics << m.toLower(); continue; }
            bool found = false;
            for (const QString& k : kMetricKeys)
                if (k.compare(m, Qt::CaseInsensitive) == 0) { statMetrics << k; found = true; }
            if (!found) a.addError(QString("--stat-metrics: unknown metric '%1' (%2).").arg(m, kMetricKeys.join(", ")));
        }
    }
    if (!a.ok()) { con.error(a.firstError()); return kExitUsage; }

    // ── jobs (as BatchPanel::makeJobTemplate) ────────────────────────────────
    std::vector<BatchJobConfig> jobs;
    for (const QString& ds : files)
        for (const QString& opt : optims) {
            BatchJobConfig j;
            j.datasetPath = ds;
            j.optimizerName = opt;
            j.weightBound = wBound;
            j.hiddenSizes = hidden;
            j.hiddenActivations.assign(hidden.size(), act);
            j.dropoutRate = dropout;
            j.trainerCfg.optimizer = opt == "sgd" ? OptimizerType::SGD : OptimizerType::Adam;   // BatchRunner sets the exact one
            j.trainerCfg.learningRate = lr;
            j.trainerCfg.epochs = epochs;
            j.trainerCfg.batchSize = batch;
            j.trainerCfg.validationSplit = valSplit;
            j.trainerCfg.earlyStoppingPatience = patience;
            jobs.push_back(j);
        }

    QString head = "NeuralStudio - Mode 2 (Batch Run)\n";
    head += QString("Datasets    %1: %2\n").arg(files.size()).arg([&] {
        QStringList n; for (const QString& f : files) n << QFileInfo(f).fileName(); return n.join(", "); }());
    head += QString("Optimizers  %1: %2\n").arg(optims.size()).arg(optims.join(", "));
    QString hs; for (size_t i = 0; i < hidden.size(); ++i) hs += (i ? "," : "") + QString::number(hidden[i]);
    head += QString("Network     hidden %1, %2, dropout %3\n").arg(hs, activationName(act)).arg(dropout);
    head += QString("Training    %1 epochs/iterations, lr %2, batch/population %3, validation %4, patience %5, weight bound %6\n")
                .arg(epochs).arg(fmtNum(lr)).arg(batch).arg(valSplit).arg(patience).arg(fmtNum(wBound));
    head += QString("Jobs        %1 x %2 run(s) = %3 trainings\n").arg(jobs.size()).arg(runs).arg(jobs.size() * runs);
    con.out(head);

    // ── run ──────────────────────────────────────────────────────────────────
    StopHandle stop;
    std::vector<RunResult> results;
    QString fatal;
    QElapsedTimer timer;
    timer.start();
    try {
        runInterruptible(stop, [&] {
            for (int run = 1; run <= runs && !stop.requested(); ++run) {
                BatchRunner br;
                StopGuard<BatchRunner> g(stop, &br);
                br.setJobs(jobs);
                const int M = static_cast<int>(jobs.size());
                QString cur;
                QObject::connect(&br, &BatchRunner::jobStarted, [&](int i, QString name) {
                    cur = QString("run %1/%2  job %3/%4  %5 / %6").arg(run).arg(runs).arg(i + 1).arg(M)
                              .arg(name, jobs[static_cast<size_t>(i)].optimizerName);
                    con.progress(cur);
                });
                QObject::connect(&br, &BatchRunner::jobProgress, [&](int, int ep, int total, double loss) {
                    con.progress(QString("%1  epoch %2/%3  loss %4").arg(cur).arg(ep).arg(total).arg(fmtNum(loss)));
                });
                QObject::connect(&br, &BatchRunner::jobCompleted, [&](int, BatchJobResult r) {
                    results.push_back({run, r});
                    if (!r.success) con.warn(QString("%1 / %2: %3").arg(r.datasetName, r.optimizerName, r.errorMsg));
                });
                QObject::connect(&br, &BatchRunner::batchError, [&](QString m) { fatal = m; });
                br.run();
                if (!fatal.isEmpty()) break;
            }
            con.endProgress();
        });
    } catch (const std::exception& ex) {
        con.error(QString::fromUtf8(ex.what()));
        return kExitError;
    }
    if (!fatal.isEmpty()) { con.error(fatal); return kExitError; }

    // ── summary per (dataset, optimizer) ────────────────────────────────────
    struct Acc {
        QString ds, opt; TaskType task = TaskType::Regression; int ok = 0, fail = 0;
        QVector<double> trE, vaE, teE, trMAE, vaMAE, teMAE, vaRMSE, teRMSE, teR2, secs;
    };
    QMap<QString, Acc> acc;
    QStringList order;
    for (const auto& rr : results) {
        const auto& r = rr.r;
        const QString key = r.datasetName + "|" + r.optimizerName;
        if (!acc.contains(key)) { order << key; acc[key].ds = r.datasetName; acc[key].opt = r.optimizerName; }
        Acc& x = acc[key];
        if (!r.success) { ++x.fail; continue; }
        ++x.ok;
        x.task = r.task;
        x.secs << r.durationMs / 1000.0;
        auto put = [](QVector<double>& v, double d) { if (d >= 0 && std::isfinite(d)) v << d; };
        if (r.task == TaskType::Regression) {
            put(x.trMAE, r.trainMAE); put(x.vaMAE, r.valMAE); put(x.teMAE, r.testMAE);
            put(x.vaRMSE, r.valRMSE); put(x.teRMSE, r.testRMSE);
            if (r.hasTestSet && std::isfinite(r.testR2)) x.teR2 << r.testR2;   // R2 may be negative
        } else {
            put(x.trE, r.trainErrorPct); put(x.vaE, r.valErrorPct); put(x.teE, r.testErrorPct);
        }
    }
    QVector<QStringList> clsRows, regRows;
    for (const QString& k : order) {
        const Acc& x = acc[k];
        const QString okS = x.fail ? QString("%1 (%2 failed)").arg(x.ok).arg(x.fail) : QString::number(x.ok);
        if (x.ok == 0) { clsRows.push_back({x.ds, x.opt, okS, "-", "-", "-", "-"}); continue; }
        if (x.task == TaskType::Regression)
            regRows.push_back({x.ds, x.opt, okS, fmtMeanSd(x.trMAE, false), fmtMeanSd(x.vaRMSE, false),
                               fmtMeanSd(x.teRMSE, false), fmtMeanSd(x.teMAE, false), fmtMeanSd(x.teR2, false),
                               QString::number(mean(x.secs), 'f', 2)});
        else
            clsRows.push_back({x.ds, x.opt, okS, fmtMeanSd(x.trE, true), fmtMeanSd(x.vaE, true), fmtMeanSd(x.teE, true),
                               QString::number(mean(x.secs), 'f', 2)});
    }
    QString rep = QString("Finished in %1%2\n\n").arg(fmtDuration(timer.elapsed() / 1000.0))
                      .arg(stop.requested() ? "  (stopped by the user: the table holds the completed jobs)" : "");
    if (!clsRows.isEmpty())
        rep += "Classification (mean +/- std over the runs)\n" +
               textTable({"Dataset", "Optimizer", "Runs", "Train err %", "Val err %", "Test err %", "Time s"}, clsRows) + "\n";
    if (!regRows.isEmpty())
        rep += "Regression (mean +/- std over the runs)\n" +
               textTable({"Dataset", "Optimizer", "Runs", "Train MAE", "Val RMSE", "Test RMSE", "Test MAE", "Test R2", "Time s"},
                         regRows) + "\n";
    con.out(rep.trimmed() + "\n");

    // ── statistics ───────────────────────────────────────────────────────────
    std::vector<BatchJobResult> plain;
    for (const auto& rr : results) plain.push_back(rr.r);
    QStringList metrics;
    if (statMetrics.contains("auto")) {
        bool hasCls = false, hasReg = false, clsTest = true, regTest = true;
        for (const auto& r : plain) {
            if (!r.success) continue;
            if (r.task == TaskType::Regression) { hasReg = true; regTest = regTest && r.hasTestSet; }
            else { hasCls = true; clsTest = clsTest && r.hasTestSet; }
        }
        if (hasCls) metrics << (clsTest ? "testErrorPct" : "valErrorPct");
        if (hasReg) metrics << (regTest ? "testRMSE" : "valRMSE");
    } else if (!statMetrics.contains("none")) {
        metrics = statMetrics;
    }
    const QString taskFilter = statTask == "binary" ? "Binary Classification" : statTask == "multiclass" ? "Multi-class"
                             : statTask == "regression" ? "Regression" : "";

    struct WRow { QString a, b; int n; double wp, wm, p, pHolm; double ma, mb; };
    struct FRes { FriedmanResult fr; NemenyiResult nem; QVector<QString> names; int N = 0; bool ok = false; };
    QMap<QString, QVector<WRow>> wilcox;
    QMap<QString, FRes> fried;
    QMap<QString, QPair<int, int>> shape;   // metric -> (optimizers, datasets)
    QString stat;
    for (const QString& m : metrics) {
        QVector<QString> names, dsNames;
        QVector<QVector<double>> mat;
        int averaged = 0;
        if (!buildBatchMetricMatrix(plain, m, taskFilter, 2, names, mat, dsNames, &averaged)) {
            stat += QString("%1: not tested (needs >= 2 optimizers and >= 2 datasets with a value for every optimizer).\n\n")
                        .arg(metricLabel(m));
            continue;
        }
        const int k = names.size(), N = dsNames.size();
        shape[m] = {k, N};
        const bool higherBetter = m.contains("R2");
        stat += QString("Statistics on %1: %2 optimizers x %3 datasets%4\n").arg(metricLabel(m)).arg(k).arg(N)
                    .arg(averaged > 0 ? QString(" (%1 repeated / fold results averaged, one value per dataset)").arg(averaged)
                                      : QString());
        if (N < 5) stat += QString("  note: only %1 datasets - the tests have very little power.\n").arg(N);

        // Wilcoxon, all pairs, Holm step-down
        QVector<WRow> rows;
        struct P { int i; double p; };
        std::vector<P> ps;
        for (int x = 0; x < k; ++x)
            for (int y = x + 1; y < k; ++y) {
                std::vector<double> va(mat[x].begin(), mat[x].end()), vb(mat[y].begin(), mat[y].end());
                const auto w = Statistics::wilcoxon(va, vb);
                WRow r{names[x], names[y], w.n, w.wPlus, w.wMinus, w.valid ? w.pValue : 1.0, -1.0,
                       mean(QVector<double>(mat[x].begin(), mat[x].end())), mean(QVector<double>(mat[y].begin(), mat[y].end()))};
                if (w.valid) ps.push_back({static_cast<int>(rows.size()), w.pValue});
                rows.push_back(r);
            }
        std::sort(ps.begin(), ps.end(), [](const P& u, const P& v) { return u.p < v.p; });
        double run = 0.0;
        const int mtests = static_cast<int>(ps.size());
        for (int i = 0; i < mtests; ++i) {
            run = std::max(run, std::min(1.0, (mtests - i) * ps[static_cast<size_t>(i)].p));
            rows[ps[static_cast<size_t>(i)].i].pHolm = run;
        }
        QVector<QStringList> trow;
        for (const auto& r : rows) {
            const bool aBetter = higherBetter ? r.ma > r.mb : r.ma < r.mb;
            trow.push_back({r.a + " vs " + r.b, QString::number(r.n), fmtNum(r.wp), fmtNum(r.wm),
                            QString::number(r.p, 'f', 4), r.pHolm >= 0 ? QString::number(r.pHolm, 'f', 4) : QString("-"),
                            r.pHolm >= 0 ? starsOf(r.pHolm) : QString("n/a"),
                            r.ma == r.mb ? QString("-") : (aBetter ? r.a : r.b)});
        }
        stat += "  Wilcoxon signed-rank (two-tailed), Holm-adjusted over the pairs:\n" +
                textTable({"Pair", "n", "W+", "W-", "p", "p Holm", "", higherBetter ? "Higher mean" : "Lower mean"}, trow);
        wilcox[m] = rows;

        // Friedman + Nemenyi (k >= 3)
        if (k >= 3) {
            std::vector<std::string> nm;
            for (const auto& n : names) nm.push_back(n.toStdString());
            std::vector<std::vector<double>> data;
            for (const auto& v : mat) data.push_back(std::vector<double>(v.begin(), v.end()));
            if (higherBetter) for (auto& row : data) for (double& v : row) v = -v;   // rank R2 as higher = better
            FRes f;
            f.fr = Statistics::friedman(nm, data);
            f.nem = Statistics::nemenyi(data);
            f.names = names;
            f.N = N;
            f.ok = f.fr.valid;
            if (f.ok) {
                stat += QString("  Friedman chi2(%1) = %2, p = %3 %4\n").arg(f.fr.df).arg(f.fr.chi2, 0, 'f', 3)
                            .arg(f.fr.pValue < 0.001 ? QString("< 0.001") : QString::number(f.fr.pValue, 'f', 4))
                            .arg(QString::fromStdString(f.fr.stars()));
                QVector<QStringList> rr;
                for (int ri = 0; ri < static_cast<int>(f.fr.rankOrder.size()); ++ri) {
                    const int j = f.fr.rankOrder[static_cast<size_t>(ri)];
                    rr.push_back({QString::number(ri + 1), names[j], QString::number(f.fr.avgRanks[static_cast<size_t>(j)], 'f', 3)});
                }
                stat += textTable({"#", "Optimizer", "Average rank"}, rr);
                if (f.nem.valid) {
                    stat += QString("  Nemenyi critical difference: CD(0.05) = %1, CD(0.10) = %2\n")
                                .arg(f.nem.cd05, 0, 'f', 3).arg(f.nem.cd10, 0, 'f', 3);
                    QStringList sig;
                    for (int x = 0; x < k; ++x)
                        for (int y = x + 1; y < k; ++y) {
                            const double gap = std::abs(f.nem.avgRanks[static_cast<size_t>(x)] - f.nem.avgRanks[static_cast<size_t>(y)]);
                            if (gap > f.nem.cd05)
                                sig << QString("%1 vs %2 (rank gap %3, p %4)").arg(names[x], names[y])
                                           .arg(gap, 0, 'f', 2).arg(f.nem.pPair[static_cast<size_t>(x)][static_cast<size_t>(y)], 0, 'f', 4);
                        }
                    stat += sig.isEmpty() ? QString("  no pair differs significantly (Nemenyi, 0.05)\n")
                                          : "  significant pairs (Nemenyi, 0.05): " + sig.join("; ") + "\n";
                }
            } else {
                stat += "  Friedman test could not be computed.\n";
            }
            fried[m] = f;
        } else {
            stat += "  Friedman test: needs at least 3 optimizers.\n";
        }
        stat += "\n";
    }
    if (!stat.isEmpty()) con.out(stat.trimmed() + "\n");

    // ── xlsx ─────────────────────────────────────────────────────────────────
    int rc = stop.requested() ? kExitInterrupted : kExitOk;
    if (a.has("out")) {
        const QString p = ensureSuffix(a.str("out"), "xlsx");
        try {
            if (!ensureParentDir(p, err)) throw std::runtime_error(err.toStdString());
            XlsxWriter wb;
            auto* sr = wb.addSheet("Runs");
            sr->setHeader({"Run", "Dataset", "Optimizer", "Task", "Success", "Error", "Samples", "Inputs", "Classes",
                           "Test file", "Epochs run", "Seconds", "Train err %", "Val err %", "Test err %",
                           "Train MAE", "Val MAE", "Test MAE", "Train RMSE", "Val RMSE", "Test RMSE", "Train R2", "Val R2", "Test R2"});
            auto num = [](double v) { return std::isfinite(v) && v != -1.0 ? XlsxCell(v) : XlsxCell(); };
            for (const auto& rr : results) {
                const auto& r = rr.r;
                const bool reg = r.task == TaskType::Regression;
                sr->addRow({XlsxCell(rr.run), XlsxCell(r.datasetName), XlsxCell(r.optimizerName), XlsxCell(taskName(r.task)),
                            XlsxCell(r.success ? "yes" : "no"), XlsxCell(r.errorMsg), XlsxCell(r.samples), XlsxCell(r.inputs),
                            XlsxCell(r.numClasses), XlsxCell(r.hasTestSet ? "yes" : "no"), XlsxCell(r.epochsRun),
                            XlsxCell(r.durationMs / 1000.0),
                            num(r.trainErrorPct), num(r.valErrorPct), num(r.testErrorPct),
                            num(r.trainMAE), num(r.valMAE), num(r.testMAE), num(r.trainRMSE), num(r.valRMSE), num(r.testRMSE),
                            reg ? XlsxCell(r.trainR2) : XlsxCell(), reg && r.valR2 != -1.0 ? XlsxCell(r.valR2) : XlsxCell(),
                            reg && r.hasTestSet ? XlsxCell(r.testR2) : XlsxCell()});
            }
            auto* ss = wb.addSheet("Summary");
            ss->setHeader({"Dataset", "Optimizer", "Task", "Successful runs", "Failed runs",
                           "Train err % mean", "Train err % std", "Val err % mean", "Val err % std", "Test err % mean", "Test err % std",
                           "Val RMSE mean", "Test RMSE mean", "Test RMSE std", "Test MAE mean", "Test R2 mean", "Seconds mean"});
            auto mc = [](const QVector<double>& v) { return v.isEmpty() ? XlsxCell() : XlsxCell(mean(v)); };
            auto sc = [](const QVector<double>& v) { return v.size() < 2 ? XlsxCell() : XlsxCell(sampleSd(v)); };
            for (const QString& k : order) {
                const Acc& x = acc[k];
                ss->addRow({XlsxCell(x.ds), XlsxCell(x.opt), XlsxCell(taskName(x.task)), XlsxCell(x.ok), XlsxCell(x.fail),
                            mc(x.trE), sc(x.trE), mc(x.vaE), sc(x.vaE), mc(x.teE), sc(x.teE),
                            mc(x.vaRMSE), mc(x.teRMSE), sc(x.teRMSE), mc(x.teMAE), mc(x.teR2), mc(x.secs)});
            }
            for (auto it = wilcox.begin(); it != wilcox.end(); ++it) {
                auto* sw = wb.addSheet(sheetName("Wilcoxon " + it.key()));
                sw->setHeader({"Optimizer A", "Optimizer B", "n (non-zero differences)", "W+", "W-", "p", "p Holm",
                               "Significance (Holm)", "Mean A", "Mean B"});
                for (const auto& r : *it)
                    sw->addRow({XlsxCell(r.a), XlsxCell(r.b), XlsxCell(r.n), XlsxCell(r.wp), XlsxCell(r.wm), XlsxCell(r.p),
                                r.pHolm >= 0 ? XlsxCell(r.pHolm) : XlsxCell(), XlsxCell(r.pHolm >= 0 ? starsOf(r.pHolm) : QString("n/a")),
                                XlsxCell(r.ma), XlsxCell(r.mb)});
                sw->addRow({XlsxCell(QString("%1 optimizers x %2 datasets, metric %3")
                                         .arg(shape[it.key()].first).arg(shape[it.key()].second).arg(metricLabel(it.key())))});
            }
            for (auto it = fried.begin(); it != fried.end(); ++it) {
                const FRes& f = *it;
                if (!f.ok) continue;
                auto* sf = wb.addSheet(sheetName("Friedman " + it.key()));
                sf->setHeader({"Rank", "Optimizer", "Average rank"});
                for (int ri = 0; ri < static_cast<int>(f.fr.rankOrder.size()); ++ri) {
                    const int j = f.fr.rankOrder[static_cast<size_t>(ri)];
                    sf->addRow({XlsxCell(ri + 1), XlsxCell(f.names[j]), XlsxCell(f.fr.avgRanks[static_cast<size_t>(j)])});
                }
                sf->addRow({});
                sf->addRow({XlsxCell("chi2"), XlsxCell(f.fr.chi2)});
                sf->addRow({XlsxCell("df"), XlsxCell(f.fr.df)});
                sf->addRow({XlsxCell("p"), XlsxCell(f.fr.pValue)});
                sf->addRow({XlsxCell("datasets"), XlsxCell(f.N)});
                if (f.nem.valid) {
                    sf->addRow({XlsxCell("Nemenyi CD 0.05"), XlsxCell(f.nem.cd05)});
                    sf->addRow({XlsxCell("Nemenyi CD 0.10"), XlsxCell(f.nem.cd10)});
                    sf->addRow({});
                    sf->addRow({XlsxCell("Optimizer A"), XlsxCell("Optimizer B"), XlsxCell("Rank gap"), XlsxCell("Nemenyi p"),
                                XlsxCell("Significant (0.05)")});
                    const int k = f.names.size();
                    for (int x = 0; x < k; ++x)
                        for (int y = x + 1; y < k; ++y) {
                            const double gap = std::abs(f.nem.avgRanks[static_cast<size_t>(x)] - f.nem.avgRanks[static_cast<size_t>(y)]);
                            sf->addRow({XlsxCell(f.names[x]), XlsxCell(f.names[y]), XlsxCell(gap),
                                        XlsxCell(f.nem.pPair[static_cast<size_t>(x)][static_cast<size_t>(y)]),
                                        XlsxCell(gap > f.nem.cd05 ? "yes" : "no")});
                        }
                }
            }
            wb.save(p);
            con.info("Written: " + QFileInfo(p).absoluteFilePath());
        } catch (const std::exception& ex) {
            con.error(QString("--out: %1").arg(QString::fromUtf8(ex.what())));
            rc = kExitError;
        }
    }
    bool anyOk = false;
    for (const auto& rr : results) anyOk = anyOk || rr.r.success;
    if (!anyOk && rc == kExitOk) rc = kExitError;
    return rc;
}

} // namespace NeuralStudio::cli
