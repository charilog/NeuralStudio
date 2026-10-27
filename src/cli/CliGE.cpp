// CliGE.cpp — Mode 6 (GE Studio) of neuralstudio-cli.
//
// Two forms, the two pages of the GE Studio panel:
//   * single run:  neuralstudio-cli ge --data FILES [options]
//       GEStudioWorker on the queued files (the "Single run" page): grammar search, final
//       comparison on the companion test file, phenotypes, convergence.
//   * experiment:  neuralstudio-cli ge experiment <ge_batch options>
//                  (or any command with --experiment / --dir)
//       the batch engine GEBatchRunner (the "Experiment" page and tools/ge_batch): datasets
//       x folds x configurations x seeds, resumable raw CSV, overall tables and tests. The
//       options are exactly those of ge_batch.
#include "CliCommon.h"
#include "CliModes.h"

#include "core/export/XlsxWriter.h"
#include "core/ge/Baselines.h"
#include "core/ge/GEBatch.h"
#include "core/ge/GEReport.h"
#include "core/ge/GEStudioWorker.h"
#include "core/ge/NetSpec.h"

#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>

namespace NeuralStudio::cli {

namespace {

QVector<OptSpec> geSpecs() {
    QVector<OptSpec> s = commonOptions();
    const QString gD = "Data and models", gG = "Grammar", gE = "Outer evolution", gI = "Inner training of every phenotype",
                  gF = "Final comparison", gB = "Fixed total budget and self-adaptive rows", gO = "Output";
    s += {
        {"data", "FILES", "", "training files (.train with a .test companion), folders or patterns", gD},
        {"config", "NAME", "unified-ge", "<preset>-<genotype>: unified|unifiedpe|archbounds|archonly x ge|pge|gelocal|sage, "
                                         "or 'baselines' (reference models only)", gD},
        {"methods", "LIST", "all GE rows", "rows of the final comparison (ids of 'list methods'); reference models are "
                                          "computed only when listed", gD},
        {"grammar", "FILE.bnf", "", "custom BNF grammar (replaces the preset grammar)", gG},
        {"max-layers", "N", "4", "most hidden layers a derivation may produce", gG},
        {"max-features", "N", "3", "most constructed features (unified presets)", gG},
        {"feature-size", "N", "15", "most operations + operands of one feature", gG},
        {"pop", "N", "40", "population", gE},
        {"gens", "N", "40", "generations", gE},
        {"replace-rate", "F", "0.3", "share replaced every generation", gE},
        {"mutation-rate", "F", "0.05", "mutation rate", gE},
        {"tournament", "N", "4", "tournament size", gE},
        {"genome-len", "N", "120", "chromosome length (codons)", gE},
        {"local-rate", "F", "0.1", "Local Crossover rate (gelocal)", gE},
        {"local-ncr", "N", "3", "Local Crossover trials", gE},
        {"pge-alpha", "F", "0.3", "PGE learning rate", gE},
        {"pge-elite", "F", "0.25", "PGE elite fraction", gE},
        {"pge-floor", "F", "0.02", "PGE probability floor", gE},
        {"transfer-prior", "", "", "PGE: carry the learned probabilities to the next dataset", gE},
        {"patience", "N", "8", "stop after N generations without improvement (0 = off)", gE},
        {"seed", "N", "0", "random seed (0 = random; a seed makes the run reproducible)", gE},
        {"optimizer", "NAME", "de", "inner optimizer", gI},
        {"inner-pop", "N", "30", "inner population", gI},
        {"inner-iters", "N", "60", "inner iterations", gI},
        {"bound", "B", "10", "base bound (bound-mode abs)", gI},
        {"bound-mode", "fan|abs", "fan", "base interval per layer: fan-in scaled or +/-B", gI},
        {"fan-k", "K", "3", "multiplier of the fan-in scaled base interval", gI},
        {"inner-polish", "1|0", "1", "bound-respecting L-BFGS after the inner optimizer", gI},
        {"inner-polish-evals", "N", "300", "evaluations of that L-BFGS", gI},
        {"lamarck", "N", "3", "a revisited phenotype resumes from its stored weights up to N times", gI},
        {"finalists", "N", "5", "best phenotypes refined at the end of the search", gI},
        {"val-split", "F", "0.2", "validation fraction (fitness)", gI},
        {"metric", "loss|error", "loss", "fitness: validation loss or validation error %", gI},
        {"parsimony", "L", "0", "fitness *= 1 + L * params / 1000", gI},
        {"topk", "K", "5", "best phenotypes retrained in the final comparison", gF},
        {"final-pop", "N", "80", "population of the final training", gF},
        {"final-iters", "N", "200", "iterations of the final training", gF},
        {"final-polish", "1|0", "1", "L-BFGS polish of every final row", gF},
        {"baseline-hidden", "SIZES", "16", "hidden layers of the Standard network", gF},
        {"baseline-act", "relu|tanh|sigmoid", "relu", "activation of the Standard network", gF},
        {"total-budget", "T", "0", "fixed total budget in loss evaluations (0 = off)", gB},
        {"search-frac", "F", "0.5", "share of T for the search", gB},
        {"polish-frac", "F", "0.5", "share of every row's budget for L-BFGS", gB},
        {"se-rule", "S", "1", "standard errors of the noise-aware rules", gB},
        {"ens-pool", "N", "12", "ensemble candidates", gB},
        {"ens-max", "N", "8", "most ensemble picks", gB},
        {"out", "FILE.xlsx", "", "write comparison, phenotypes, convergence and best programs to an .xlsx file", gO},
    };
    s += modelExportOptions();
    return s;
}

bool configFromNameCli(const QString& name, GEStudioConfig& c) {
    // the same mapping as the batch engine (GEBatch.cpp)
    if (name == "baselines") { c.mode = "baselines"; return true; }
    const QStringList p = name.split('-');
    if (p.size() != 2) return false;
    if (p[0] == "unified") c.grammarText.clear();
    else if (p[0] == "unifiedpe") c.grammarText = QString::fromStdString(ge::presetGrammar(ge::GrammarPreset::UnifiedEmbedding));
    else if (p[0] == "archbounds") c.grammarText = QString::fromStdString(ge::presetGrammar(ge::GrammarPreset::ArchAndBounds));
    else if (p[0] == "archonly") c.grammarText = QString::fromStdString(ge::presetGrammar(ge::GrammarPreset::ArchOnly));
    else return false;
    if (p[1] == "ge") { c.genotype = "ge"; c.crossover = "onepoint"; }
    else if (p[1] == "sage") { c.genotype = "ge"; c.crossover = "onepoint"; c.adaptiveSearch = true; }
    else if (p[1] == "gelocal") { c.genotype = "ge"; c.crossover = "local"; }
    else if (p[1] == "pge") { c.genotype = "pge"; }
    else return false;
    if (p[0] != "unified" && p[0] != "unifiedpe") c.maxFeatures = 0;
    return true;
}

bool isExperimentCall(const QStringList& args) {
    if (!args.isEmpty() && (args.first() == "experiment" || args.first() == "batch")) return true;
    for (const QString& s : args)
        if (s == "--experiment" || s.startsWith("--experiment=") || s == "--dir" || s.startsWith("--dir=")
            || s == "--summarize-only")
            return true;
    return false;
}

// ── experiment (ge_batch engine) ─────────────────────────────────────────────
int runExperiment(QStringList args) {
    auto& con = Console::get();
    if (!args.isEmpty() && (args.first() == "experiment" || args.first() == "batch")) args.removeFirst();
    // our own general flags are not ge_batch options
    args.removeAll("--quiet");
    args.removeAll("--verbose");
    // --key=value -> --key value (ge_batch syntax)
    QStringList norm;
    for (const QString& s : args) {
        const int eq = s.indexOf('=');
        if (s.startsWith("--") && eq > 2) norm << s.left(eq) << s.mid(eq + 1);
        else norm << s;
    }
    if (norm.isEmpty() || norm.contains("--help") || norm.contains("-h")) {
        con.out("neuralstudio-cli ge experiment  -  Mode 6, GE Studio experiments (datasets x folds x models)\n"
                "The same engine and options as the Experiment page of GE Studio and tools/ge_batch.\n\n" +
                batchUsage().replace("ge_batch", "neuralstudio-cli ge experiment"));
        return norm.isEmpty() ? kExitUsage : kExitOk;
    }
    GEBatchOptions opt;
    QString err;
    if (!parseBatchArgs(norm, QDir::currentPath(), opt, err)) { con.error(err); return kExitUsage; }
    for (const QString& w : opt.warnings) con.warn(w);

    StopHandle stop;
    QString failure;
    bool stopped = false;
    try {
        runInterruptible(stop, [&] {
            GEBatchRunner runner;
            StopGuard<GEBatchRunner> g(stop, &runner);
            runner.setOptions(opt);
            QObject::connect(&runner, &GEBatchRunner::message, [&](QString m) { con.out(m); });
            QObject::connect(&runner, &GEBatchRunner::progress, [&](int job, int todo, double within, QString label) {
                con.progress(QString("job %1/%2  %3%  %4").arg(job + 1).arg(todo).arg(static_cast<int>(within * 100)).arg(label));
            });
            QObject::connect(&runner, &GEBatchRunner::jobFinished, [&](int, int, QString line, bool ok) {
                if (ok) con.info(line); else con.warn(line);
            });
            QObject::connect(&runner, &GEBatchRunner::finished, [&](bool s, QString) { stopped = s; });
            QObject::connect(&runner, &GEBatchRunner::errorOccurred, [&](QString m) { failure = m; });
            runner.run();
            con.endProgress();
        });
    } catch (const std::exception& ex) {
        con.error(QString::fromUtf8(ex.what()));
        return kExitError;
    }
    if (!failure.isEmpty()) { con.error(failure); return kExitError; }
    if (stopped || stop.requested()) {
        con.warn("stopped: the unfinished run was discarded; run the same command again to resume.");
        return kExitInterrupted;
    }
    return kExitOk;
}

void printHelpGE() {
    Console::get().out(
        "neuralstudio-cli ge  -  Mode 6, Grammatical Evolution Studio\n"
        "One BNF grammar derives the architecture, per-layer weight-bound commands and optional constructed\n"
        "features of a network; an inner optimizer trains the weights inside the derived bounds. The final\n"
        "table compares the best phenotypes with a Standard network on the companion test file.\n\n"
        "usage: neuralstudio-cli ge --data FILES [options]               single run (the 'Single run' page)\n"
        "       neuralstudio-cli ge experiment <experiment options>      experiment (see 'ge experiment --help')\n"
        + formatOptions(geSpecs()) +
        "\nexamples:\n"
        "  neuralstudio-cli ge --data wine.train --config unified-ge --gens 30 --seed 7 --out ge_wine.xlsx\n"
        "  neuralstudio-cli ge --data wine.train --methods standard,ge1,ge1_nobounds,rf,svm\n"
        "  neuralstudio-cli ge experiment --experiment experiments/v4_smoke.json\n"
        "  neuralstudio-cli ge experiment --dir sample_data/tenfolding --datasets wine,pima --folds 1-10 --out results/run1\n");
}

} // namespace

int cmdGE(const QStringList& argv) {
    auto& con = Console::get();
    if (isExperimentCall(argv)) return runExperiment(argv);

    Args a;
    QString err;
    const auto specs = geSpecs();
    if (!a.parse(argv, specs, err)) { con.error(err); con.info("Run 'neuralstudio-cli ge --help' for the options."); return kExitUsage; }
    if (a.flag("help")) { printHelpGE(); return kExitOk; }

    GEStudioConfig c;
    QStringList files;
    if (!collectDataFiles(a, files, err)) a.addError(err);
    c.datasetPaths = files;
    const QString cfgName = a.str("config", "unified-ge").trimmed().toLower();
    if (!configFromNameCli(cfgName, c))
        a.addError(QString("--config: unknown configuration '%1' (e.g. unified-ge, archbounds-pge, unifiedpe-sage, baselines).").arg(cfgName));
    if (a.has("grammar")) {
        QFile f(a.str("grammar"));
        if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) a.addError("Cannot read the grammar file " + a.str("grammar"));
        else c.grammarText = QString::fromUtf8(f.readAll());
    }
    c.maxLayers = a.integer("max-layers", c.maxLayers, 1, 12);
    if (a.has("max-features")) c.maxFeatures = a.integer("max-features", c.maxFeatures, 0, 32);
    c.maxFeatureSize = a.integer("feature-size", c.maxFeatureSize, 1, 1000);
    c.pop = a.integer("pop", c.pop, 4, 100000);
    c.gens = a.integer("gens", c.gens, 1, 1000000);
    c.replaceRate = a.number("replace-rate", c.replaceRate, 0.0, 1.0);
    c.mutationRate = a.number("mutation-rate", c.mutationRate, 0.0, 1.0);
    c.tournament = a.integer("tournament", c.tournament, 2, 1000);
    c.genomeLen = a.integer("genome-len", c.genomeLen, 4, 100000);
    c.localRate = a.number("local-rate", c.localRate, 0.0, 1.0);
    c.localNcr = a.integer("local-ncr", c.localNcr, 1, 1000);
    c.pgeAlpha = a.number("pge-alpha", c.pgeAlpha, 0.0, 1.0);
    c.pgeElite = a.number("pge-elite", c.pgeElite, 0.0, 1.0);
    c.pgeFloor = a.number("pge-floor", c.pgeFloor, 0.0, 1.0);
    c.transferPrior = a.flag("transfer-prior");
    c.patience = a.integer("patience", 8, 0, 1000000);   // the panel's default
    c.seed = static_cast<quint32>(a.bigInt("seed", 0, 0, 4294967295LL));
    c.innerOpt = a.str("optimizer", c.innerOpt).trimmed().toLower();
    if (!phaseOptimizers().contains(c.innerOpt))
        a.addError(QString("--optimizer: '%1' is not one of %2.").arg(c.innerOpt, phaseOptimizers().join(", ")));
    c.innerPop = a.integer("inner-pop", c.innerPop, 4, 100000);
    c.innerIters = a.integer("inner-iters", c.innerIters, 1, 10000000);
    c.baseBound = a.number("bound", c.baseBound, 1e-6, 1e6);
    c.boundMode = a.choice("bound-mode", c.boundMode, {"fan", "abs"});
    c.fanK = a.number("fan-k", c.fanK, 1e-6, 1e6);
    c.innerPolish = a.boolean("inner-polish", c.innerPolish);
    c.innerPolishEvals = a.integer("inner-polish-evals", c.innerPolishEvals, 0, 100000000);
    c.lamarck = a.integer("lamarck", c.lamarck, 0, 1000);
    c.finalists = a.integer("finalists", c.finalists, 0, 1000);
    c.valSplit = a.number("val-split", c.valSplit, 0.0, 0.9);
    c.fitnessMetric = a.choice("metric", c.fitnessMetric, {"loss", "error"});
    c.parsimony = a.number("parsimony", c.parsimony, 0.0, 1e6);
    c.topK = a.integer("topk", c.topK, 1, 100);
    c.finalPop = a.integer("final-pop", c.finalPop, 4, 100000);
    c.finalIters = a.integer("final-iters", c.finalIters, 1, 10000000);
    c.finalPolish = a.boolean("final-polish", c.finalPolish);
    c.baselineHidden = a.str("baseline-hidden", c.baselineHidden);
    {
        std::vector<int> tmp;
        if (!parseHidden(c.baselineHidden, tmp, err) || tmp.empty()) a.addError("--baseline-hidden: e.g. 16 or 32,16.");
    }
    c.baselineAct = a.choice("baseline-act", c.baselineAct, {"relu", "tanh", "sigmoid"});
    c.totalBudget = a.bigInt("total-budget", c.totalBudget, 0, std::numeric_limits<qint64>::max());
    c.searchFrac = a.number("search-frac", c.searchFrac, 0.0, 1.0);
    c.polishFrac = a.number("polish-frac", c.polishFrac, 0.0, 1.0);
    c.seRule = a.number("se-rule", c.seRule, 0.0, 100.0);
    c.ensPool = a.integer("ens-pool", c.ensPool, 1, 10000);
    c.ensMax = a.integer("ens-max", c.ensMax, 1, 10000);
    if (a.has("methods")) {
        for (const QString& m : a.list("methods")) {
            bool known = false;
            geMethodInfo(m, &known);
            if (!known) { a.addError(QString("--methods: unknown model '%1' (see 'neuralstudio-cli list methods').").arg(m)); break; }
            c.methods << m;
        }
        if (c.mode == "ge") {
            // as the panel: reference models only -> no GE search
            bool anyGE = false;
            for (const QString& m : c.methods) anyGE = anyGE || !geBaselineIds().contains(m);
            if (!anyGE) c.mode = "baselines";
        }
    }
    if (!a.ok()) { con.error(a.firstError()); return kExitUsage; }

    QString head = "NeuralStudio - Mode 6 (GE Studio, single run)\n";
    head += QString("Datasets      %1\n").arg([&] { QStringList n; for (const auto& f : files) n << QFileInfo(f).fileName(); return n.join(", "); }());
    if (c.mode == "baselines") {
        head += QString("Models        reference models only: %1\n").arg(c.methods.isEmpty() ? geBaselineIds().join(", ") : c.methods.join(", "));
    } else {
        head += QString("Grammar       %1%2, max %3 layers, %4 features\n")
                    .arg(cfgName, a.has("grammar") ? " (custom grammar " + QFileInfo(a.str("grammar")).fileName() + ")" : QString())
                    .arg(c.maxLayers).arg(c.maxFeatures);
        head += QString("Evolution     %1, population %2, generations %3, patience %4, seed %5\n")
                    .arg(c.genotype.toUpper()).arg(c.pop).arg(c.gens).arg(c.patience)
                    .arg(c.seed ? QString::number(c.seed) : QString("random"));
        head += QString("Inner         %1, population %2, iterations %3, bounds %4, fitness validation %5\n")
                    .arg(optimizerLabel(c.innerOpt)).arg(c.innerPop).arg(c.innerIters).arg(c.boundMode).arg(c.fitnessMetric);
        head += QString("Final         top %1, population %2, iterations %3, Standard %4 %5%6\n")
                    .arg(c.topK).arg(c.finalPop).arg(c.finalIters).arg(c.baselineHidden, c.baselineAct)
                    .arg(c.totalBudget > 0 ? QString(", total budget T = %1").arg(c.totalBudget) : QString());
        if (!c.methods.isEmpty()) head += QString("Rows          %1\n").arg(c.methods.join(", "));
    }
    con.out(head);

    StopHandle stop;
    QVector<GEDatasetResult> results;
    QString failure;
    std::shared_ptr<NeuralNetwork> lastNet;
    QElapsedTimer timer;
    timer.start();
    try {
        runInterruptible(stop, [&] {
            GEStudioWorker w;
            StopGuard<GEStudioWorker> g(stop, &w);
            w.setConfig(c);
            QString ds;
            QObject::connect(&w, &GEStudioWorker::datasetStarted, [&](int i, int n, QString name, QString info) {
                ds = QString("dataset %1/%2 %3").arg(i + 1).arg(n).arg(name);
                if (!info.isEmpty()) con.info(QString("%1: %2").arg(name, asciiText(info)));
                con.progress(ds);
            });
            QObject::connect(&w, &GEStudioWorker::generationDone,
                             [&](int gen, int total, double best, QString, double, double, int uniq, int hits, qint64 calls) {
                con.progress(QString("%1  generation %2/%3  best fitness %4  (%5 phenotypes trained, %6 cache hits, %7 evaluations)")
                                 .arg(ds).arg(gen).arg(total).arg(fmtNum(best)).arg(uniq).arg(hits).arg(calls));
            });
            QObject::connect(&w, &GEStudioWorker::finalStageStarted, [&](int n) { con.progress(QString("%1  final comparison: %2 rows").arg(ds).arg(n)); });
            QObject::connect(&w, &GEStudioWorker::finalRowDone, [&](int row, int n, QString label) {
                con.progress(QString("%1  final comparison %2/%3  %4").arg(ds).arg(row + 1).arg(n).arg(label));
            });
            QObject::connect(&w, &GEStudioWorker::datasetFinished, [&](GEDatasetResult r) { results.push_back(r); });
            QObject::connect(&w, &GEStudioWorker::allDone, [&](std::shared_ptr<NeuralNetwork> n, std::shared_ptr<Dataset>) { lastNet = n; });
            QObject::connect(&w, &GEStudioWorker::errorOccurred, [&](QString m) { failure = m; });
            w.run();
            con.endProgress();
        });
    } catch (const std::exception& ex) {
        con.error(QString::fromUtf8(ex.what()));
        return kExitError;
    }
    if (!failure.isEmpty()) con.error(failure);

    QString rep = QString("Finished in %1%2\n\n").arg(fmtDuration(timer.elapsed() / 1000.0))
                      .arg(stop.requested() ? "  (stopped by the user)" : "");
    for (const auto& r : results) {
        const QString unit = r.isCls ? "err %" : "MSE";
        rep += QString("Dataset %1 (%2, %3 training rows%4)\n").arg(r.dsName, r.isCls ? "classification" : "regression")
                   .arg(r.trainSamples).arg(r.hasTest ? QString(", %1 test rows").arg(r.testSamples) : QString(", no test file"));
        if (c.mode != "baselines")
            rep += QString("  search: %1/%2 generations, %3 phenotypes trained, %4 cache hits, %5 loss evaluations, %6\n")
                       .arg(r.gensRun).arg(r.gensPlanned).arg(r.uniqueEvals).arg(r.cacheHits).arg(r.searchCalls)
                       .arg(fmtDuration(r.searchSeconds));
        QVector<QStringList> rows;
        for (const auto& x : r.rows)
            rows.push_back({asciiText(x.label), fmtErr(x.trainErr, r.isCls), fmtErr(x.valErr, r.isCls), fmtErr(x.testErr, r.isCls),
                            QString::number(x.params), QString::number(x.evals), QString::number(x.seconds, 'f', 1)});
        if (!rows.isEmpty())
            rep += textTable({"Model", "Train " + unit, "Val " + unit, "Test " + unit, "Params", "Evaluations", "Seconds"}, rows);
        if (!r.top.isEmpty()) {
            QVector<QStringList> pr;
            for (const auto& p : r.top)
                pr.push_back({QString::number(p.rank), asciiText(p.desc), fmtNum(p.fitness), fmtErr(p.trainErr, r.isCls),
                              fmtErr(p.valErr, r.isCls), QString::number(p.params), QString::number(p.inputs)});
            rep += "  best phenotypes:\n" + textTable({"#", "Phenotype", "Fitness", "Train " + unit, "Val " + unit, "Params", "Inputs"}, pr);
        }
        if (!r.bestProgram.isEmpty()) rep += "  best program: " + r.bestProgram.simplified() + "\n";
        rep += "\n";
    }
    con.out(rep.trimmed() + "\n");

    int rc = stop.requested() ? kExitInterrupted : (failure.isEmpty() ? kExitOk : kExitError);
    if (a.has("out") && !results.isEmpty()) {
        const QString p = ensureSuffix(a.str("out"), "xlsx");
        try {
            if (!ensureParentDir(p, err)) throw std::runtime_error(err.toStdString());
            XlsxWriter wb;
            auto* sc = wb.addSheet("Comparison");
            sc->setHeader({"Dataset", "Task", "Model id", "Model", "Uses BNF", "Train error", "Validation error", "Test error",
                           "Parameters", "Loss evaluations", "Seconds", "Error unit"});
            for (const auto& r : results)
                for (const auto& x : r.rows)
                    sc->addRow({XlsxCell(r.dsName), XlsxCell(r.isCls ? "classification" : "regression"), XlsxCell(x.id),
                                XlsxCell(x.label), XlsxCell(geMethodInfo(x.id).bnf ? "yes" : "no"), XlsxCell(x.trainErr),
                                x.valErr >= 0 ? XlsxCell(x.valErr) : XlsxCell(), x.testErr >= 0 ? XlsxCell(x.testErr) : XlsxCell(),
                                XlsxCell(x.params), XlsxCell(x.evals), XlsxCell(x.seconds), XlsxCell(r.isCls ? "error %" : "MSE")});
            auto* sp = wb.addSheet("Phenotypes");
            sp->setHeader({"Dataset", "Rank", "Phenotype", "Fitness", "Train error", "Validation error", "Parameters", "Inputs"});
            for (const auto& r : results)
                for (const auto& x : r.top)
                    sp->addRow({XlsxCell(r.dsName), XlsxCell(x.rank), XlsxCell(x.desc), XlsxCell(x.fitness), XlsxCell(x.trainErr),
                                XlsxCell(x.valErr), XlsxCell(x.params), XlsxCell(x.inputs)});
            auto* sv = wb.addSheet("Convergence");
            sv->setHeader({"Dataset", "Generation", "Best fitness"});
            for (const auto& r : results)
                for (int i = 0; i < r.convergence.size(); ++i)
                    sv->addRow({XlsxCell(r.dsName), XlsxCell(i), XlsxCell(r.convergence[i])});
            auto* ss = wb.addSheet("Search");
            ss->setHeader({"Dataset", "Generations run", "Generations planned", "Phenotypes trained", "Cache hits",
                           "Loss evaluations", "Search seconds", "Best program"});
            for (const auto& r : results)
                ss->addRow({XlsxCell(r.dsName), XlsxCell(r.gensRun), XlsxCell(r.gensPlanned), XlsxCell(r.uniqueEvals),
                            XlsxCell(r.cacheHits), XlsxCell(r.searchCalls), XlsxCell(r.searchSeconds), XlsxCell(r.bestProgram)});
            wb.save(p);
            con.info("Written: " + QFileInfo(p).absoluteFilePath());
        } catch (const std::exception& ex) {
            con.error(QString("--out: %1").arg(QString::fromUtf8(ex.what())));
            rc = kExitError;
        }
    }
    if (a.has("save-model") || a.has("export-cpp") || a.has("export-onnx")) {
        if (lastNet) { if (!writeModelExports(a, *lastNet)) rc = kExitError; }
        else con.warn("No network to save: the best phenotype uses constructed features or an input embedding, "
                      "which the model files cannot hold (as in the GUI).");
    }
    return rc;
}

} // namespace NeuralStudio::cli
