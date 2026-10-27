// CliMain.cpp — entry point of neuralstudio-cli, the console program of NeuralStudio.
//
//   neuralstudio-cli <command> [options]
//
// One command per mode of the graphical application (train = Mode 1, batch = Mode 2,
// pretrain = Mode 3, bounds = Mode 4, archsearch = Mode 5, ge = Mode 6), plus 'predict'
// and 'list'. The modes can also be called by number ("neuralstudio-cli 4 ...").
// Results are written to standard output (and to .xlsx / model files on request), progress
// and messages to standard error. Exit codes: 0 success, 1 a run failed, 2 invalid command
// line, 130 interrupted with Ctrl+C.
#include "CliCommon.h"
#include "CliModes.h"

#include "core/ge/Baselines.h"
#include "core/ge/GEReport.h"

#include <QCoreApplication>
#include <QMap>

#include <cstdio>

#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#endif

#ifndef NS_VERSION
#  define NS_VERSION "1.0.6"
#endif

using namespace NeuralStudio;
using namespace NeuralStudio::cli;

namespace {

void messageHandler(QtMsgType type, const QMessageLogContext&, const QString& msg) {
    auto& con = Console::get();
    switch (type) {
        case QtDebugMsg:
        case QtInfoMsg:
            if (con.verbose()) con.info(msg);
            break;
        case QtWarningMsg:
            if (!con.quiet() || con.verbose()) con.warn(msg);
            break;
        case QtCriticalMsg:
        case QtFatalMsg:
            con.error(msg);
            break;
    }
}

QString mainUsage() {
    return QString(
        "NeuralStudio %1 - command-line interface\n"
        "\n"
        "usage: neuralstudio-cli <command> [options]\n"
        "       neuralstudio-cli <command> --help\n"
        "\n"
        "commands (the modes of the graphical application):\n"
        "  train       Mode 1  Single Workflow: train one MLP or RBF network, evaluate, cross-validate, export\n"
        "  batch       Mode 2  Batch Run: datasets x optimizers x runs, Wilcoxon / Friedman / Nemenyi tests\n"
        "  pretrain    Mode 3  Pre-Train + NNC: pre-trained small network transferred into the main network\n"
        "  bounds      Mode 4  Evolutionary Bounds: per-parameter weight bounds, training inside them\n"
        "  archsearch  Mode 5  Architecture Search: evolutionary search of layers, sizes and activations\n"
        "  ge          Mode 6  GE Studio: grammatical evolution (single run, or 'ge experiment ...')\n"
        "  predict             apply a saved .nsmodel network to a data file\n"
        "  list                optimizers, GE configurations and models\n"
        "\n"
        "The modes can also be given by number: 'neuralstudio-cli 4 --data wine.train' = 'bounds'.\n"
        "Every command uses the same workers and defaults as the corresponding panel of the GUI.\n"
        "Data files: Neural Designer (.train/.test/.data) and CSV/TSV; the companion test file\n"
        "(wine.train -> wine.test, a_train.csv -> a_test.csv) is found automatically and is never\n"
        "used for training or selection.\n"
        "\n"
        "general options: --help, --quiet (results only), --verbose (library log), --version\n"
        "Ctrl+C stops after the current step and reports what was completed; press it twice to quit.\n"
        "\n"
        "examples:\n"
        "  neuralstudio-cli train --data wine.train --hidden 16 --optimizer adam --out wine.xlsx --save-model wine.nsmodel\n"
        "  neuralstudio-cli predict --model wine.nsmodel --data new_rows.csv --out predictions.csv\n"
        "  neuralstudio-cli batch --data sample_data/selected --optimizers adam,de,pso --runs 5 --out batch.xlsx\n"
        "  neuralstudio-cli bounds --data pima.train --runs 10 --out bounds.xlsx\n"
        "  neuralstudio-cli ge experiment --experiment experiments/v4_smoke.json\n").arg(NS_VERSION);
}

int cmdListImpl(const QStringList& args) {
    auto& con = Console::get();
    const QString what = args.isEmpty() ? QString("all") : args.first().toLower();
    if (what == "--help" || what == "help") {
        con.out("usage: neuralstudio-cli list [optimizers|configs|methods|all]");
        return kExitOk;
    }
    QString out;
    if (what == "all" || what == "optimizers") {
        out += "Optimizers of Modes 1 and 2 (train, batch: --optimizer / --optimizers):\n";
        for (const QString& o : allTrainingOptimizers())
            out += QString("  %1%2%3\n").arg(o.leftJustified(12), optimizerLabel(o),
                                             isGradientOptimizer(o) ? QString("  [gradient]") : QString());
        out += "  groups: all, gradient, meta\n\n";
        out += "Optimizers of Modes 3 and 4 (pretrain --p1-opt..., bounds --opt1...) and of the inner training of Modes 5 and 6:\n  "
               + phaseOptimizers().join(", ") + "\n";
        out += "  bounds --opt2 also accepts ge_grammar; --p3-opt, --opt2 and --opt3 accept none\n";
        out += "Outer optimizers of Mode 5 (archsearch --outer):\n  " + archOuterOptimizers().join(", ") + "\n\n";
    }
    if (what == "all" || what == "configs") {
        out += "GE Studio configurations (ge --config, ge experiment --configs):\n"
               "  <preset>-<genotype>\n"
               "  preset:   unified (architecture + bounds + features), unifiedpe (+ periodic input embedding),\n"
               "            archbounds (architecture + bounds), archonly (architecture)\n"
               "  genotype: ge (codons), gelocal (codons + Local Crossover), pge (probabilistic GE),\n"
               "            sage (GE with the noise-aware search stop)\n"
               "  baselines: the reference models only\n\n";
    }
    if (what == "all" || what == "methods") {
        out += "GE Studio models (ge --methods, experiment files):\n";
        for (const auto& m : geMethodCatalog())
            out += QString("  %1%2%3\n").arg(m.id.leftJustified(18), m.bnf ? QString("[BNF] ") : QString("      "), m.label);
        out += "  ge2 .. geK: the next best phenotypes\n"
               "Reference models (computed only when listed): " + geBaselineIds().join(", ") + "\n";
    }
    if (out.isEmpty()) { con.error("list: unknown topic '" + what + "' (optimizers, configs, methods, all)."); return kExitUsage; }
    con.out(out.trimmed());
    return kExitOk;
}

} // namespace

namespace NeuralStudio::cli {
int cmdList(const QStringList& args) { return cmdListImpl(args); }
}

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    QCoreApplication::setApplicationName("neuralstudio-cli");
    QCoreApplication::setApplicationVersion(NS_VERSION);
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    qInstallMessageHandler(messageHandler);

    QStringList args = QCoreApplication::arguments();
    if (!args.isEmpty()) args.removeFirst();
    for (QString& s : args) if (s == "-h") s = "--help";

    auto& con = Console::get();
    con.setQuiet(args.contains("--quiet"));
    con.setVerbose(args.contains("--verbose"));

    if (args.isEmpty() || args.first() == "--help" || args.first() == "help") {
        if (args.size() >= 2 && args.first() == "help") {
            args = QStringList{args[1], "--help"};
        } else {
            con.out(mainUsage());
            return args.isEmpty() ? kExitUsage : kExitOk;
        }
    }
    if (args.first() == "--version" || args.first() == "version") {
        con.out(QString("neuralstudio-cli %1").arg(NS_VERSION));
        return kExitOk;
    }

    static const QMap<QString, QString> alias = {
        {"train", "train"}, {"single", "train"}, {"1", "train"}, {"mode1", "train"},
        {"batch", "batch"}, {"2", "batch"}, {"mode2", "batch"},
        {"pretrain", "pretrain"}, {"pre-train", "pretrain"}, {"3", "pretrain"}, {"mode3", "pretrain"},
        {"bounds", "bounds"}, {"evol-bounds", "bounds"}, {"4", "bounds"}, {"mode4", "bounds"},
        {"archsearch", "archsearch"}, {"arch", "archsearch"}, {"5", "archsearch"}, {"mode5", "archsearch"},
        {"ge", "ge"}, {"ge-studio", "ge"}, {"6", "ge"}, {"mode6", "ge"},
        {"predict", "predict"}, {"list", "list"}};
    const QString cmd = alias.value(args.first().toLower());
    if (cmd.isEmpty()) {
        con.error(QString("Unknown command '%1'.").arg(args.first()));
        con.info("Commands: train, batch, pretrain, bounds, archsearch, ge, predict, list (or the mode numbers 1-6).");
        return kExitUsage;
    }
    const QStringList rest = args.mid(1);
    installInterruptHandler();
    try {
        if (cmd == "train")      return cmdTrain(rest);
        if (cmd == "batch")      return cmdBatch(rest);
        if (cmd == "pretrain")   return cmdPretrain(rest);
        if (cmd == "bounds")     return cmdBounds(rest);
        if (cmd == "archsearch") return cmdArch(rest);
        if (cmd == "ge")         return cmdGE(rest);
        if (cmd == "predict")    return cmdPredict(rest);
        if (cmd == "list")       return cmdList(rest);
    } catch (const std::exception& ex) {
        con.error(QString::fromUtf8(ex.what()));
        return kExitError;
    }
    return kExitUsage;
}
