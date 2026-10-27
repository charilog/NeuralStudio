// CliTrain.cpp — Mode 1 (Single Workflow) and 'predict' of neuralstudio-cli.
//
// Mirrors the Data / Network / Training / Evaluation / Prediction panels:
//   * the companion test file is found as in the Data panel; during training it is the
//     monitoring curve only (all training rows are used and early stopping is off), exactly
//     as the Training panel does - unless --no-monitor-test is given, in which case the
//     validation split drives early stopping and the test file is scored only at the end;
//   * the network is built as in the Network panel (same output size, normalisation,
//     regression output scaling, class values);
//   * gradient optimizers run the Trainer, all other methods the MetaTrainer, RBF networks
//     the analytic RBFTrainer; K-fold cross-validation runs the CrossValidator (gradient
//     training: a population method falls back to Adam, as in the GUI).
#include "CliCommon.h"
#include "CliModes.h"

#include "core/dataset/DatasetLoader.h"
#include "core/export/XlsxWriter.h"
#include "core/nn/ModelSerializer.h"
#include "core/nn/RBFTrainer.h"
#include "core/nn/Trainer.h"
#include "core/optimizers/MetaTrainer.h"
#include "core/validation/CrossValidator.h"

#include <QDateTime>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QRegularExpression>
#include <QTextStream>

#include <cmath>

namespace NeuralStudio::cli {

namespace {

QVector<OptSpec> trainSpecs() {
    QVector<OptSpec> s = commonOptions();
    const QString gD = "Data", gN = "Network (MLP)", gR = "Network (RBF)", gT = "Training",
                  gC = "Cross-validation", gO = "Output";
    s += {
        {"data", "FILE", "", "training file (.train/.csv/...); its companion test file is loaded automatically", gD},
        {"test", "FILE", "", "use this test file instead of the companion", gD},
        {"no-test", "", "", "do not load a companion test file", gD},
        {"no-monitor-test", "", "", "train with the validation split (early stopping) and score the test file only at the end", gD},
        {"task", "auto|binary|multiclass|regression", "auto", "task (auto = detected from the target column)", gD},
        {"model", "mlp|rbf", "mlp", "network type", gN},
        {"hidden", "SIZES", "64,32", "hidden layer sizes, e.g. 64,32 ('none' = no hidden layer)", gN},
        {"act", "ACT[,ACT...]", "relu", "activation: relu|tanh|sigmoid|linear, one for all layers or one per layer", gN},
        {"dropout", "P", "0", "dropout rate of the hidden layers (gradient optimizers)", gN},
        {"centers", "K", "20", "RBF centres", gR},
        {"kernel", "NAME", "gaussian", "gaussian|multiquadric|inv-multiquadric", gR},
        {"width", "NAME", "maxdist", "width rule: maxdist|nearest|mean3", gR},
        {"width-scale", "S", "1", "multiplier of the computed widths", gR},
        {"ridge", "L", "1e-4", "ridge regularisation of the output layer", gR},
        {"kmeans-iter", "N", "150", "k-means iterations", gR},
        {"optimizer", "NAME", "adam", "one of 'neuralstudio-cli list optimizers' (24 methods)", gT},
        {"epochs", "N", "300", "epochs (gradient) or iterations (other methods)", gT},
        {"lr", "RATE", "0.001", "learning rate (gradient optimizers)", gT},
        {"lr-schedule", "constant|step|exp", "constant", "learning-rate schedule", gT},
        {"lr-decay", "F", "0.5", "decay factor of the step / exponential schedule", gT},
        {"lr-step", "N", "50", "epochs per step of the step schedule", gT},
        {"weight-decay", "W", "0.01", "AdamW weight decay", gT},
        {"batch", "N", "32 / 30", "mini-batch size (gradient) or population size (other methods)", gT},
        {"val-split", "F", "0.2", "validation fraction of the training file", gT},
        {"patience", "N", "20", "early-stopping patience in epochs (0 = off)", gT},
        {"weight-bound", "B", "5", "search interval [-B, B] of the weights (non-gradient methods)", gT},
        {"cv", "K", "", "also run K-fold cross-validation on the training file (2..20)", gC},
        {"cv-only", "", "", "run only the cross-validation (no final training)", gC},
        {"out", "FILE.xlsx", "", "write settings, metrics, training history and CV folds to an .xlsx file", gO},
        {"history", "FILE.csv", "", "write the per-epoch losses to a .csv file", gO},
        {"predict", "FILE", "", "after training, predict the rows of FILE", gO},
        {"predict-out", "FILE.csv", "<FILE>_predictions.csv", "where the predictions are written", gO},
    };
    s += modelExportOptions();
    return s;
}

QVector<OptSpec> predictSpecs() {
    QVector<OptSpec> s = commonOptions();
    const QString g = "Prediction";
    s += {
        {"model", "FILE.nsmodel", "", "a network saved with --save-model or File > Save Model", g},
        {"data", "FILE", "", "rows to predict (inputs only, or inputs followed by the target)", g},
        {"out", "FILE.csv", "", "write the predictions here (default: standard output)", g},
    };
    return s;
}

QString netDescription(const NeuralNetwork& net) {
    QString s = QString("MLP %1").arg(net.inputSize());
    auto& layers = const_cast<NeuralNetwork&>(net).layers();
    for (size_t i = 0; i < layers.size(); ++i) {
        const bool out = i + 1 == layers.size();
        s += QString(" -> %1").arg(layers[i].outputSize());
        if (!out) s += " " + activationName(layers[i].activation());
    }
    s += QString("  (%1 parameters)").arg(net.totalParams());
    return s;
}

// Rows of a prediction file: inputs (+ optional targets)
struct PredictRows {
    std::vector<std::vector<double>> x;
    std::vector<double> target;   // first target column, empty when the file has none
};

bool parsePlainRows(const QString& path, int nIn, PredictRows& rows, QString& error) {
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) { error = "Cannot open " + path; return false; }
    QTextStream ts(&f);
    static const QRegularExpression sep("[,;\\t ]+");
    int lineNo = 0;
    bool first = true;
    while (!ts.atEnd()) {
        const QString line = ts.readLine().trimmed();
        ++lineNo;
        if (line.isEmpty() || line.startsWith('#')) continue;
        const QStringList tok = line.split(sep, Qt::SkipEmptyParts);
        std::vector<double> v;
        bool numeric = true;
        for (const QString& t : tok) {
            bool ok = false;
            const double d = t.toDouble(&ok);
            if (!ok) { numeric = false; break; }
            v.push_back(d);
        }
        if (!numeric) {
            if (first) { first = false; continue; }   // header line
            error = QString("%1, line %2: non-numeric value.").arg(path).arg(lineNo);
            return false;
        }
        first = false;
        if (static_cast<int>(v.size()) == nIn) {
            rows.x.push_back(v);
        } else if (static_cast<int>(v.size()) == nIn + 1) {
            rows.target.push_back(v.back());
            v.pop_back();
            rows.x.push_back(v);
        } else {
            error = QString("%1, line %2: %3 values, the network expects %4 inputs (optionally followed by the target).")
                        .arg(path).arg(lineNo).arg(v.size()).arg(nIn);
            return false;
        }
    }
    if (!rows.target.empty() && rows.target.size() != rows.x.size()) {
        error = "Some rows have a target value and some do not.";
        return false;
    }
    if (rows.x.empty()) { error = "No data rows in " + path; return false; }
    return true;
}

bool readPredictRows(const QString& path, int nIn, PredictRows& rows, QString& error) {
    // 1. the dataset loaders (Neural Designer format, CSV with header detection)
    try {
        auto ds = DatasetLoader::load(path);
        if (ds && ds->sampleCount > 0) {
            if (ds->inputCount == nIn) {
                for (int r = 0; r < ds->sampleCount; ++r) {
                    rows.x.push_back(ds->inputRow(r));
                    if (ds->outputCount > 0) rows.target.push_back(ds->outputAt(r, 0));
                }
                return true;
            }
            if (ds->inputCount + ds->outputCount == nIn) {   // inputs only: the loader took the last column as target
                for (int r = 0; r < ds->sampleCount; ++r) {
                    auto v = ds->inputRow(r);
                    for (int o = 0; o < ds->outputCount; ++o) v.push_back(ds->outputAt(r, o));
                    rows.x.push_back(v);
                }
                return true;
            }
        }
    } catch (const std::exception&) {
        // fall through to the plain reader
    }
    rows = PredictRows();
    return parsePlainRows(path, nIn, rows, error);
}

// Writes the predictions of `net` for the rows of `inPath`; prints metrics when targets exist.
bool predictToFile(NeuralNetwork& net, const QString& inPath, const QString& outPath, QString& error) {
    auto& con = Console::get();
    PredictRows rows;
    if (!readPredictRows(inPath, net.inputSize(), rows, error)) return false;
    net.setTrainingMode(false);
    const TaskType task = net.taskType();
    const bool hasT = !rows.target.empty();

    QString csv;
    QTextStream ts(&csv);
    ts.setRealNumberPrecision(10);
    // header
    ts << "row";
    if (task == TaskType::Regression) {
        for (int o = 0; o < net.outputSize(); ++o) ts << (net.outputSize() == 1 ? ",prediction" : QString(",prediction_%1").arg(o + 1));
    } else if (task == TaskType::BinaryClassification) {
        ts << ",predicted_class,score_class_" << QString::number(net.outputMax(), 'g', 6);
    } else {
        ts << ",predicted_class";
        for (double v : net.classValues()) ts << ",p_class_" << QString::number(v, 'g', 6);
    }
    if (hasT) ts << ",target";
    ts << "\n";

    int correct = 0;
    double sse = 0, sae = 0, sumT = 0, sst = 0;
    for (const double t : rows.target) sumT += t;
    const double meanT = hasT ? sumT / rows.target.size() : 0.0;
    for (size_t i = 0; i < rows.x.size(); ++i) {
        const auto p = net.predict(rows.x[i]);
        ts << (i + 1);
        if (task == TaskType::Regression) {
            for (double v : p) ts << "," << v;
            if (hasT) {
                const double d = p[0] - rows.target[i];
                sse += d * d; sae += std::abs(d);
                sst += (rows.target[i] - meanT) * (rows.target[i] - meanT);
            }
        } else if (task == TaskType::BinaryClassification) {
            const bool one = p[0] >= 0.5;
            ts << "," << (one ? net.outputMax() : net.outputMin()) << "," << p[0];
            if (hasT) {
                const bool tOne = rows.target[i] > 0.5 * (net.outputMin() + net.outputMax());
                if (tOne == one) ++correct;
            }
        } else {
            const int best = static_cast<int>(std::max_element(p.begin(), p.end()) - p.begin());
            ts << "," << net.classValues().at(static_cast<size_t>(best));
            for (double v : p) ts << "," << v;
            if (hasT && net.classIndex(rows.target[i]) == best) ++correct;
        }
        if (hasT) ts << "," << rows.target[i];
        ts << "\n";
    }
    ts.flush();

    if (outPath.isEmpty()) {
        QByteArray b = csv.toUtf8();
        if (b.endsWith('\n')) b.chop(1);
        con.out(QString::fromUtf8(b));
    } else {
        QString err;
        if (!ensureParentDir(outPath, err)) { error = err; return false; }
        QFile f(outPath);
        if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) { error = "Cannot write " + outPath; return false; }
        f.write(csv.toUtf8());
        con.info(QString("Predictions of %1 rows written to %2").arg(rows.x.size()).arg(QFileInfo(outPath).absoluteFilePath()));
    }
    if (hasT) {
        const double n = static_cast<double>(rows.x.size());
        if (task == TaskType::Regression)
            con.info(QString("Against the target column: MSE %1  RMSE %2  MAE %3  R2 %4")
                         .arg(fmtNum(sse / n)).arg(fmtNum(std::sqrt(sse / n))).arg(fmtNum(sae / n))
                         .arg(fmtNum(sst > 1e-12 ? 1.0 - sse / sst : 0.0)));
        else
            con.info(QString("Against the target column: accuracy %1 %  error %2 %")
                         .arg(100.0 * correct / n, 0, 'f', 2).arg(100.0 - 100.0 * correct / n, 0, 'f', 2));
    }
    return true;
}

void printHelpTrain() {
    Console::get().out(
        "neuralstudio-cli train  -  Mode 1, Single Workflow\n"
        "Train one MLP or RBF network on one data file, evaluate it on the training, validation\n"
        "and test data, optionally run K-fold cross-validation, and save / export the model.\n\n"
        "usage: neuralstudio-cli train --data FILE [options]\n"
        "       neuralstudio-cli train FILE [options]\n"
        + formatOptions(trainSpecs()) +
        "\nexamples:\n"
        "  neuralstudio-cli train --data wine.train --hidden 16 --act tanh --optimizer adam --epochs 500\n"
        "  neuralstudio-cli train wine.train --optimizer de --batch 50 --epochs 200 --save-model wine.nsmodel\n"
        "  neuralstudio-cli train housing.train --model rbf --centers 30 --out housing_rbf.xlsx\n"
        "  neuralstudio-cli train pima.train --cv 10 --cv-only\n");
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════════
// train
// ═══════════════════════════════════════════════════════════════════════════════
int cmdTrain(const QStringList& argv) {
    auto& con = Console::get();
    Args a;
    QString err;
    const auto specs = trainSpecs();
    if (!a.parse(argv, specs, err)) { con.error(err); con.info("Run 'neuralstudio-cli train --help' for the options."); return kExitUsage; }
    if (a.flag("help")) { printHelpTrain(); return kExitOk; }

    // ── options ──────────────────────────────────────────────────────────────
    const QString model = a.choice("model", "mlp", {"mlp", "rbf"});
    const QString taskChoice = a.choice("task", "auto", {"auto", "binary", "multiclass", "regression"});
    const QString optName = a.str("optimizer", "adam").trimmed().toLower();
    if (!allTrainingOptimizers().contains(optName))
        a.addError(QString("Unknown optimizer '%1'. Choose from: %2.").arg(optName, allTrainingOptimizers().join(", ")));
    const bool gd = isGradientOptimizer(optName);
    const int epochs = a.integer("epochs", 300, 1, 1000000);
    const double lr = a.number("lr", 0.001, 1e-9, 10.0);
    const QString sched = a.choice("lr-schedule", "constant", {"constant", "step", "exp"});
    const double lrDecay = a.number("lr-decay", 0.5, 1e-6, 1.0);
    const int lrStep = a.integer("lr-step", 50, 1, 1000000);
    const double wDecay = a.number("weight-decay", 0.01, 0.0, 10.0);
    const int batch = a.integer("batch", gd ? 32 : 30, 1, 1000000);
    const double valSplit = a.number("val-split", 0.2, 0.0, 0.9);
    const int patience = a.integer("patience", 20, 0, 1000000);
    const double wBound = a.number("weight-bound", 5.0, 1e-6, 1e6);
    const double dropout = a.number("dropout", 0.0, 0.0, 0.95);
    const int cvK = a.has("cv") ? a.integer("cv", 5, 2, 20) : 0;
    const bool cvOnly = a.flag("cv-only");
    if (cvOnly && cvK == 0) a.addError("--cv-only needs --cv K.");
    if (model == "rbf" && cvK > 0) a.addError("Cross-validation is available for MLP networks (as in the GUI).");
    std::vector<int> hidden;
    if (!parseHidden(a.str("hidden", "64,32"), hidden, err)) a.addError(err);
    std::vector<Activation> acts;
    {
        const QStringList an = a.has("act") ? a.list("act") : QStringList{"relu"};
        std::vector<Activation> parsed;
        for (const QString& s : an) {
            Activation x;
            if (!parseActivation(s, x)) { a.addError(QString("Unknown activation '%1' (relu, tanh, sigmoid, linear).").arg(s)); break; }
            parsed.push_back(x);
        }
        if (parsed.size() == 1) acts.assign(hidden.size(), parsed.front());
        else if (parsed.size() == hidden.size()) acts = parsed;
        else if (!parsed.empty())
            a.addError(QString("--act: give one activation or one per hidden layer (%1 layers).").arg(hidden.size()));
    }
    RBFConfig rcfg;
    rcfg.nCenters = a.integer("centers", 20, 2, 100000);
    {
        const QString k = a.choice("kernel", "gaussian", {"gaussian", "multiquadric", "inv-multiquadric"});
        rcfg.kernel = k == "multiquadric" ? RBFKernel::Multiquadric
                    : k == "inv-multiquadric" ? RBFKernel::InvMultiquadric : RBFKernel::Gaussian;
        const QString w = a.choice("width", "maxdist", {"maxdist", "nearest", "mean3"});
        rcfg.widthStrategy = w == "nearest" ? RBFWidthStrategy::NearestNeighbor
                           : w == "mean3" ? RBFWidthStrategy::MeanNN : RBFWidthStrategy::MaxDist;
    }
    rcfg.widthScale = a.number("width-scale", 1.0, 1e-6, 1e6);
    rcfg.ridgeLambda = a.number("ridge", 1e-4, 0.0, 1e6);
    rcfg.kmeansMaxIter = a.integer("kmeans-iter", 150, 1, 100000);
    rcfg.valSplit = valSplit;
    if (model == "rbf" && (a.has("save-model") || a.has("export-cpp") || a.has("export-onnx")))
        a.addError("Saving and exporting are available for MLP networks (as in the GUI).");
    if (a.flag("no-test") && a.has("test")) a.addError("--test and --no-test cannot be combined.");
    QStringList files;
    if (!collectDataFiles(a, files, err)) a.addError(err);
    else if (files.size() != 1) a.addError(QString("Mode 1 trains on one data file (%1 given); use 'batch' for several.").arg(files.size()));
    if (!a.ok()) { con.error(a.firstError()); return kExitUsage; }

    // ── data ─────────────────────────────────────────────────────────────────
    std::shared_ptr<Dataset> train, test;
    try {
        if (a.has("test")) {
            train = std::shared_ptr<Dataset>(DatasetLoader::load(files.first()).release());
            test = std::shared_ptr<Dataset>(DatasetLoader::load(a.str("test")).release());
        } else {
            DatasetBundle b = DatasetLoader::loadBundle(files.first());
            train = b.train;
            if (!a.flag("no-test")) test = b.test;
        }
    } catch (const std::exception& ex) {
        con.error(QString::fromUtf8(ex.what()));
        return kExitError;
    }
    if (!train || train->sampleCount == 0) { con.error("The training file has no rows."); return kExitError; }
    if (test && (test->inputCount != train->inputCount || test->outputCount != train->outputCount)) {
        con.error(QString("The test file has %1 inputs / %2 outputs, the training file %3 / %4.")
                      .arg(test->inputCount).arg(test->outputCount).arg(train->inputCount).arg(train->outputCount));
        return kExitError;
    }
    const TaskType task = resolveTask(taskChoice, *train);
    const bool cls = task != TaskType::Regression;
    // the test file is the monitoring curve of an MLP training (Training panel); RBF networks and
    // the cross-validation do not use it during training
    const bool monitorTest = test && !a.flag("no-monitor-test") && model == "mlp" && !cvOnly;

    // ── report header ────────────────────────────────────────────────────────
    QString head = "NeuralStudio - Mode 1 (Single Workflow)\n";
    {
        const auto uv = train->uniqueOutputValues(0);
        head += QString("Data      %1  (%2 samples, %3 inputs, %4%5)\n")
                    .arg(QFileInfo(files.first()).fileName()).arg(train->sampleCount).arg(train->inputCount)
                    .arg(taskLabel(task))
                    .arg(cls ? QString(", %1 classes").arg(uv.size()) : QString());
        if (test)
            head += QString("Test      %1  (%2 samples)%3\n").arg(QFileInfo(test->sourcePath).fileName()).arg(test->sampleCount)
                        .arg(monitorTest ? "  - training curve only, never used to stop or select (as in the GUI)"
                             : cvOnly ? "  - not used by the cross-validation"
                                      : "  - scored after training");
        else head += "Test      none\n";
    }

    std::shared_ptr<NeuralNetwork> net;
    std::shared_ptr<RBFNetwork> rbf;
    if (model == "mlp") {
        net = buildMlp(*train, task, hidden, acts, dropout);
        head += "Network   " + netDescription(*net) + "\n";
    } else {
        rbf = std::make_shared<RBFNetwork>();
        const auto uv = train->uniqueOutputValues(0);
        const int nOut = task == TaskType::MultiClassClassification ? static_cast<int>(uv.size())
                       : task == TaskType::BinaryClassification ? 1 : train->outputCount;
        rbf->init(train->inputCount, nOut, task, rcfg);
        head += QString("Network   RBF %1 -> %2 centres -> %3, %4 kernel, ridge %5\n")
                    .arg(train->inputCount).arg(rcfg.nCenters).arg(nOut)
                    .arg(a.str("kernel", "gaussian")).arg(fmtNum(rcfg.ridgeLambda));
    }
    if (model == "mlp" && !cvOnly) {
        if (gd)
            head += QString("Training  %1, %2 epochs, learning rate %3 (%4), batch %5, validation %6, patience %7\n")
                        .arg(optimizerLabel(optName)).arg(epochs).arg(fmtNum(lr)).arg(sched).arg(batch)
                        .arg(monitorTest ? QString("none (all rows train)") : QString::number(valSplit))
                        .arg(monitorTest ? QString("off") : QString::number(patience));
        else {
            // MetaTrainer: quasi-Newton methods take one step per iteration, single-point methods
            // 20 evaluations per iteration, population methods `batch` (the population size)
            const bool qn = optName == "lbfgs" || optName == "bfgs" || optName == "cg" || optName == "lm";
            const bool sp = optName == "sa" || optName == "neldermead";
            const QString budget = qn ? QString("up to %1 steps").arg(epochs)
                                 : sp ? QString("%1 iterations x 20 evaluations").arg(epochs)
                                      : QString("%1 iterations, population %2").arg(epochs).arg(batch);
            head += QString("Training  %1, %2, weights in [-%3, %3], validation %4, patience %5\n")
                        .arg(optimizerLabel(optName), budget).arg(fmtNum(wBound))
                        .arg(monitorTest ? QString("none (all rows train)") : QString::number(valSplit))
                        .arg(monitorTest ? QString("off") : QString::number(patience));
        }
    }
    con.out(head);

    if (!gd && model == "mlp" && !cvOnly) {
        // the search dimension against the evaluation budget, as the banner of the Training panel
        const int D = net->totalParams();
        const bool qn = optName == "lbfgs" || optName == "bfgs" || optName == "cg" || optName == "lm";
        const bool sp = optName == "sa" || optName == "neldermead";
        if (!qn) {
            const qint64 budget = static_cast<qint64>(epochs) * (sp ? 20 : std::max(4, batch));
            const double perW = static_cast<double>(budget) / std::max(1, D);
            if (perW < 20)
                con.warn(QString("%1 weights, budget %2 evaluations (%3 per weight): small budget for this many weights; "
                                 "consider a smaller network, more iterations, or L-BFGS / Adam.")
                             .arg(D).arg(budget).arg(perW, 0, 'f', 1));
        }
    }

    // ── training ─────────────────────────────────────────────────────────────
    StopHandle stop;
    QVector<EpochResult> history;
    QString trainError;
    double rbfTr = -1, rbfVal = -1, rbfTrA = -1, rbfValA = -1;
    CVSummary cv;
    bool cvDone = false;
    QElapsedTimer timer;
    timer.start();
    double trainSeconds = 0, cvSeconds = 0;
    try {
        runInterruptible(stop, [&] {
            if (model == "rbf") {
                RBFTrainer t;
                t.setNetwork(rbf);
                t.setDataset(train.get());
                QObject::connect(&t, &RBFTrainer::progressMessage, [&](QString m) { con.progress(asciiText(m)); });
                QObject::connect(&t, &RBFTrainer::trainingCompleted, [&](double te, double ve, double ta, double va) {
                    rbfTr = te; rbfVal = ve; rbfTrA = ta; rbfValA = va;
                });
                QObject::connect(&t, &RBFTrainer::trainingError, [&](QString m) { trainError = m; });
                t.run();
                trainSeconds = timer.elapsed() / 1000.0;
                con.endProgress();
                return;
            }
            if (!cvOnly) {
                TrainerConfig tc;
                tc.epochs = epochs;
                tc.batchSize = batch;
                tc.validationSplit = valSplit;
                tc.earlyStoppingPatience = patience;
                tc.learningRate = lr;
                tc.lrSchedule = sched == "step" ? LRSchedule::StepDecay : sched == "exp" ? LRSchedule::ExponentialDecay
                                                                                         : LRSchedule::Constant;
                tc.lrDecayRate = lrDecay;
                tc.lrDecayStepSize = lrStep;
                tc.weightDecay = wDecay;
                tc.optimizer = optName == "sgd" ? OptimizerType::SGD : optName == "adamw" ? OptimizerType::AdamW
                             : optName == "rmsprop" ? OptimizerType::RMSProp : optName == "nadam" ? OptimizerType::Nadam
                             : optName == "adagrad" ? OptimizerType::AdaGrad : OptimizerType::Adam;
                auto onEpoch = [&](EpochResult e) {
                    history.push_back(e);
                    QString p = QString("epoch %1/%2  train loss %3").arg(e.epoch).arg(epochs).arg(fmtNum(e.trainLoss));
                    if (e.valLoss >= 0) p += QString("  %1 loss %2").arg(monitorTest ? "test" : "val", fmtNum(e.valLoss));
                    if (e.valAcc >= 0) p += QString("  %1 acc %2 %").arg(monitorTest ? "test" : "val").arg(e.valAcc * 100, 0, 'f', 2);
                    con.progress(p);
                };
                if (gd) {
                    Trainer t;
                    StopGuard<Trainer> g(stop, &t);
                    t.setConfig(tc);
                    t.setNetwork(net.get());
                    t.setDataset(train.get());
                    if (monitorTest) t.setValDataset(test.get(), /*isTestSet=*/true);
                    QObject::connect(&t, &Trainer::epochCompleted, onEpoch);
                    QObject::connect(&t, &Trainer::trainingError, [&](QString m) { trainError = m; });
                    t.run();
                } else {
                    MetaTrainer t;
                    StopGuard<MetaTrainer> g(stop, &t);
                    t.setConfig(tc);
                    t.setOptimizerName(optName);
                    t.setNetwork(net.get());
                    t.setDataset(train.get());
                    t.setWeightBound(wBound);
                    if (monitorTest) t.setExternalValDataset(test.get(), /*isTestSet=*/true);
                    QObject::connect(&t, &MetaTrainer::epochCompleted, onEpoch);
                    QObject::connect(&t, &MetaTrainer::trainingError, [&](QString m) { trainError = m; });
                    t.run();
                }
                con.endProgress();
            }
            trainSeconds = timer.elapsed() / 1000.0;
            if (cvK > 0 && trainError.isEmpty() && !stop.requested()) {
                NetworkConfig nc;
                nc.task = task;
                nc.hiddenSizes = hidden;
                nc.hiddenActivations = acts;
                nc.dropoutRate = dropout;
                TrainerConfig tc;
                tc.optimizer = optName == "sgd" ? OptimizerType::SGD : optName == "adamw" ? OptimizerType::AdamW
                             : optName == "rmsprop" ? OptimizerType::RMSProp : optName == "nadam" ? OptimizerType::Nadam
                             : optName == "adagrad" ? OptimizerType::AdaGrad : OptimizerType::Adam;
                tc.learningRate = lr;
                tc.lrSchedule = sched == "step" ? LRSchedule::StepDecay : sched == "exp" ? LRSchedule::ExponentialDecay
                                                                                         : LRSchedule::Constant;
                tc.lrDecayRate = lrDecay;
                tc.lrDecayStepSize = lrStep;
                tc.weightDecay = wDecay;
                tc.epochs = epochs;
                tc.batchSize = batch;   // the GUI passes the same spin box (population size for non-gradient methods)
                if (!gd) con.info(QString("Cross-validation uses gradient training: %1 is replaced by Adam (as in the GUI).")
                                      .arg(optimizerLabel(optName)));
                CrossValidator v;
                StopGuard<CrossValidator> g(stop, &v);
                v.setK(cvK);
                v.setDataset(train.get());
                v.setNetworkConfig(nc);
                v.setTrainerConfig(tc);
                QObject::connect(&v, &CrossValidator::foldCompleted, [&](FoldResult f) {
                    con.progress(QString("cross-validation: fold %1/%2 done").arg(f.fold).arg(cvK));   // folds are numbered from 1
                });
                QObject::connect(&v, &CrossValidator::cvFinished, [&](CVSummary s) { cv = s; cvDone = true; });
                QObject::connect(&v, &CrossValidator::cvError, [&](QString m) { trainError = "cross-validation: " + m; });
                v.run();
                cvSeconds = timer.elapsed() / 1000.0 - trainSeconds;
                con.endProgress();
            }
        });
    } catch (const std::exception& ex) {
        con.error(QString::fromUtf8(ex.what()));
        return kExitError;
    }
    const double seconds = trainSeconds;
    if (!trainError.isEmpty()) { con.error(trainError); return kExitError; }

    // ── results ──────────────────────────────────────────────────────────────
    QString rep;
    EvalResult evTrain, evVal, evTest;
    std::vector<int> vRows;
    if (model == "rbf") {
        evTrain = evaluateRbf(*rbf, *train);
        if (test) evTest = evaluateRbf(*rbf, *test);
        rep += QString("Trained in %1 (analytic: k-means++ centres, width rule, ridge regression)\n").arg(fmtDuration(seconds));
        if (rbfVal >= 0)
            rep += QString("Validation split (%1 of the training rows): MSE %2%3\n").arg(valSplit).arg(fmtNum(rbfVal))
                       .arg(rbfValA >= 0 ? QString(", accuracy %1 %").arg(rbfValA * 100, 0, 'f', 2) : QString());
    } else if (!cvOnly) {
        const int last = history.isEmpty() ? 0 : history.last().epoch;
        rep += QString("Trained in %1, %2 %3%4\n").arg(fmtDuration(seconds)).arg(last).arg(gd ? "epochs" : "iterations")
                   .arg(stop.requested() ? "  (stopped by the user)"
                        : (last < epochs && !monitorTest && patience > 0 ? "  (early stopping: best-validation weights restored)" : ""));
        evTrain = evaluateNetwork(*net, *train);
        if (!monitorTest && valSplit > 0) {
            vRows = validationRows(train->sampleCount, valSplit);
            if (!vRows.empty()) evVal = evaluateNetwork(*net, *train, &vRows);
        }
        if (test) evTest = evaluateNetwork(*net, *test);
    }
    if (evTrain.valid) rep += describeEval("Training file", evTrain);
    if (evVal.valid) rep += describeEval("Validation rows", evVal);
    if (evTest.valid) rep += describeEval("Test file", evTest);
    if (cvDone) {
        if (!rep.isEmpty()) rep += "\n";
        rep += QString("%1-fold cross-validation on the training file (%2 folds completed, %3):\n")
                   .arg(cvK).arg(cv.folds.size()).arg(fmtDuration(cvSeconds));
        QVector<QStringList> rows;
        for (const auto& f : cv.folds)
            rows.push_back({QString::number(f.fold), fmtNum(f.finalLoss),
                            f.finalAcc >= 0 ? QString::number(f.finalAcc * 100, 'f', 2) : QString("-")});
        rep += textTable({"Fold", "Validation loss", "Accuracy %"}, rows);
        rep += QString("  mean loss %1 +/- %2").arg(fmtNum(cv.meanLoss)).arg(fmtNum(cv.stdLoss, 3));
        if (cv.meanAcc >= 0) rep += QString("   mean accuracy %1 +/- %2 %").arg(cv.meanAcc * 100, 0, 'f', 2).arg(cv.stdAcc * 100, 0, 'f', 2);
        rep += "\n";
    }
    con.out(rep);

    // ── files ────────────────────────────────────────────────────────────────
    int rc = stop.requested() ? kExitInterrupted : kExitOk;
    if (a.has("history") && !history.isEmpty()) {
        const QString p = a.str("history");
        if (!ensureParentDir(p, err)) { con.error(err); rc = kExitError; }
        else {
            QFile f(p);
            if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) { con.error("Cannot write " + p); rc = kExitError; }
            else {
                QTextStream ts(&f);
                ts.setRealNumberPrecision(10);
                const QString w = monitorTest ? "test" : "val";
                ts << "epoch,train_loss," << w << "_loss,train_acc," << w << "_acc,learning_rate\n";
                for (const auto& e : history)
                    ts << e.epoch << "," << e.trainLoss << "," << e.valLoss << "," << e.trainAcc << "," << e.valAcc
                       << "," << e.currentLR << "\n";
                con.info("Written: " + QFileInfo(p).absoluteFilePath());
            }
        }
    }
    if (a.has("out")) {
        const QString p = ensureSuffix(a.str("out"), "xlsx");
        try {
            if (!ensureParentDir(p, err)) throw std::runtime_error(err.toStdString());
            XlsxWriter wb;
            auto* st = wb.addSheet("Settings");
            st->setHeader({"Setting", "Value"});
            st->addRow({XlsxCell("data"), XlsxCell(QFileInfo(files.first()).absoluteFilePath())});
            st->addRow({XlsxCell("test"), XlsxCell(test ? test->sourcePath : QString("none"))});
            st->addRow({XlsxCell("task"), XlsxCell(taskLabel(task))});
            st->addRow({XlsxCell("network"), XlsxCell(net ? netDescription(*net) : QString("RBF, %1 centres").arg(rcfg.nCenters))});
            if (net && !cvOnly) {
                st->addRow({XlsxCell("optimizer"), XlsxCell(optimizerLabel(optName))});
                st->addRow({XlsxCell(gd ? "epochs" : "iterations"), XlsxCell(epochs)});
                st->addRow({XlsxCell(gd ? "batch size" : "population"), XlsxCell(batch)});
                if (gd) st->addRow({XlsxCell("learning rate"), XlsxCell(lr)});
                else st->addRow({XlsxCell("weight bound"), XlsxCell(wBound)});
                st->addRow({XlsxCell("validation split"), XlsxCell(monitorTest ? 0.0 : valSplit)});
                st->addRow({XlsxCell("patience"), XlsxCell(monitorTest ? 0 : patience)});
            }
            st->addRow({XlsxCell("seconds"), XlsxCell(seconds)});
            auto* sm = wb.addSheet("Metrics");
            sm->setHeader({"Set", "Metric", "Value"});
            evalToSheet(sm, "train", evTrain);
            evalToSheet(sm, "validation", evVal);
            evalToSheet(sm, "test", evTest);
            if (!history.isEmpty()) {
                auto* sh = wb.addSheet("History");
                const QString w = monitorTest ? "test" : "val";
                sh->setHeader({"epoch", "train loss", w + " loss", "train acc", w + " acc", "learning rate"});
                for (const auto& e : history)
                    sh->addRow({XlsxCell(e.epoch), XlsxCell(e.trainLoss), XlsxCell(e.valLoss), XlsxCell(e.trainAcc),
                                XlsxCell(e.valAcc), XlsxCell(e.currentLR)});
            }
            if (cvDone) {
                auto* sc = wb.addSheet("Cross-validation");
                sc->setHeader({"fold", "validation loss", "accuracy"});
                for (const auto& f : cv.folds) sc->addRow({XlsxCell(f.fold), XlsxCell(f.finalLoss), XlsxCell(f.finalAcc)});
                sc->addRow({XlsxCell("mean"), XlsxCell(cv.meanLoss), XlsxCell(cv.meanAcc)});
                sc->addRow({XlsxCell("std"), XlsxCell(cv.stdLoss), XlsxCell(cv.stdAcc)});
            }
            wb.save(p);
            con.info("Written: " + QFileInfo(p).absoluteFilePath());
        } catch (const std::exception& ex) {
            con.error(QString("--out: %1").arg(QString::fromUtf8(ex.what())));
            rc = kExitError;
        }
    }
    if (net && !cvOnly) {
        if (!writeModelExports(a, *net)) rc = kExitError;
        if (a.has("predict")) {
            const QString in = a.str("predict");
            const QString outP = a.has("predict-out") ? a.str("predict-out")
                                 : QFileInfo(in).absolutePath() + "/" + QFileInfo(in).completeBaseName() + "_predictions.csv";
            if (!predictToFile(*net, in, outP, err)) { con.error(err); rc = kExitError; }
        }
    } else if (a.has("predict")) {
        con.warn("--predict needs a trained MLP (it is skipped for RBF networks and with --cv-only).");
    }
    return rc;
}

// ═══════════════════════════════════════════════════════════════════════════════
// predict
// ═══════════════════════════════════════════════════════════════════════════════
int cmdPredict(const QStringList& argv) {
    auto& con = Console::get();
    Args a;
    QString err;
    const auto specs = predictSpecs();
    if (!a.parse(argv, specs, err)) { con.error(err); return kExitUsage; }
    if (a.flag("help")) {
        con.out("neuralstudio-cli predict  -  apply a saved network to new rows\n\n"
                "usage: neuralstudio-cli predict --model FILE.nsmodel --data FILE [--out FILE.csv]\n"
                + formatOptions(specs) +
                "\nThe data file holds one row per sample: the input values, optionally followed by the target\n"
                "(then accuracy / errors against it are reported). CSV, TSV, whitespace-separated and Neural\n"
                "Designer files are read; a non-numeric first line is taken as a header.\n");
        return kExitOk;
    }
    QString dataPath = a.str("data");
    if (dataPath.isEmpty() && !a.positional().isEmpty()) dataPath = a.positional().first();
    if (!a.has("model")) { con.error("--model FILE.nsmodel is required."); return kExitUsage; }
    if (dataPath.isEmpty()) { con.error("--data FILE is required."); return kExitUsage; }
    NeuralNetwork net;
    try {
        ModelSerializer::load(net, a.str("model"));
    } catch (const std::exception& ex) {
        con.error(QString::fromUtf8(ex.what()));
        return kExitError;
    }
    con.info(QString("Model: %1, %2").arg(netDescription(net), taskLabel(net.taskType())));
    if (!predictToFile(net, dataPath, a.str("out"), err)) { con.error(err); return kExitError; }
    return kExitOk;
}

} // namespace NeuralStudio::cli
