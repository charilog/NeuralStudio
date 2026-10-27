// Runs the workers of Modes 2-5 headless on one dataset with a small budget and prints the
// comparison rows (train / validation / TEST error). ./modes_check <file.train>
// (Mode 3 = Pre-Train + NNC, Mode 4 = Evol. Bounds; their output lines used to be labelled M5 / M4.)
#include "core/bounds/EvolBoundsWorker.h"
#include "core/pretrain/PreTrainWorker.h"
#include "core/arch/ArchSearchWorker.h"
#include "core/batch/BatchRunner.h"
#include <QCoreApplication>
#include <cstdio>
using namespace NeuralStudio;
static int bad = 0;
static void show(const char* mode, const QString& ds, const QStringList& cfg, const QVector<double>& tr,
                 const QVector<double>& va, const QVector<double>& te, bool cls) {
    for (int i = 0; i < cfg.size(); ++i) {
        std::printf("%s %s | %-22s train %8.4g  val %8.4g  test %8.4g\n", mode, qPrintable(ds), qPrintable(cfg[i]),
                    tr.value(i), va.value(i), te.value(i, -1));
        if (te.value(i, -1) < 0) ++bad;   // the companion test file must be scored
    }
    (void)cls;
}
int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    if (argc < 2) return 2;
    {
        EvolBoundsConfig c; c.datasetPaths << argv[1]; c.hiddenLayout = "8";
        c.opt1 = "de"; c.Nc1 = 30; c.Ng1 = 30; c.opt2 = "de"; c.Nc2 = 30; c.Ng2 = 30; c.opt3 = "de"; c.Nc3 = 30; c.Ng3 = 30;
        EvolBoundsWorker w; w.setConfig(c);
        QObject::connect(&w, &EvolBoundsWorker::datasetFinished, [](QString ds, QStringList cfg, QVector<double>, QVector<double> tr,
                                                                    QVector<double> va, QVector<double> te, bool cls) { show("M4", ds, cfg, tr, va, te, cls); });
        QObject::connect(&w, &EvolBoundsWorker::errorOccurred, [](QString m) { std::printf("M4 error: %s\n", qPrintable(m)); ++bad; });
        w.run();
    }
    {
        PreTrainConfig c; c.datasetPaths << argv[1]; c.hiddenLayout = "8"; c.Iw = 4;
        c.phase1 = {"de", 30, 30, 10.0}; c.phase2 = {"de", 30, 30, 10.0}; c.phase3 = {"de", 30, 30, 10.0};
        PreTrainWorker w; w.setConfig(c);
        QObject::connect(&w, &PreTrainWorker::datasetFinished, [](QString ds, QStringList cfg, QVector<double>, QVector<double> tr,
                                                                  QVector<double> va, QVector<double> te, bool cls) { show("M3", ds, cfg, tr, va, te, cls); });
        QObject::connect(&w, &PreTrainWorker::errorOccurred, [](QString m) { std::printf("M3 error: %s\n", qPrintable(m)); ++bad; });
        w.run();
    }
    // Mode 5: every architecture also gets a TEST error (scored only), and the run must not crash
    {
        ArchSearchConfig c; c.datasetPaths << argv[1];
        c.minLayers = 1; c.maxLayers = 1; c.candidateSizes = {4, 8}; c.candidateActs = {"relu", "tanh"};
        c.outerOpt = "de"; c.outerPop = 5; c.outerGens = 5; c.innerOpt = "de"; c.innerPop = 10; c.innerIters = 20;
        c.innerPolishEvals = 100; c.topK = 3;
        ArchSearchWorker w; w.setConfig(c);
        QObject::connect(&w, &ArchSearchWorker::datasetFinished, [](QString ds, QVector<ArchResult> top, QVector<double>, bool) {
            for (const auto& r : top) {
                std::printf("M5 %s | #%d %-24s val %8.4g  test %8.4g  train %8.4g\n", qPrintable(ds), r.rank, qPrintable(r.desc),
                            r.valErr, r.testErr, r.trainErr);
                if (r.testErr < 0) ++bad;
            }
            if (top.isEmpty()) ++bad;
        });
        QObject::connect(&w, &ArchSearchWorker::errorOccurred, [](QString m) { std::printf("M5 error: %s\n", qPrintable(m)); ++bad; });
        w.run();
        // an unreadable file is an error message, not a crash
        ArchSearchConfig c2 = c; c2.datasetPaths = QStringList{QString(argv[1]) + ".does_not_exist"};
        ArchSearchWorker w2; w2.setConfig(c2);
        bool reported = false;
        QObject::connect(&w2, &ArchSearchWorker::errorOccurred, [&reported](QString) { reported = true; });
        w2.run();
        std::printf("M5 missing file -> %s\n", reported ? "error reported" : "NOT REPORTED");
        if (!reported) ++bad;
    }
    // Mode 2: validation metrics of the final network are filled (classification and regression)
    {
        BatchJobConfig j; j.datasetPath = argv[1]; j.hiddenSizes = {8}; j.hiddenActivations = {Activation::Tanh};
        j.trainerCfg.epochs = 40; j.trainerCfg.validationSplit = 0.2; j.trainerCfg.earlyStoppingPatience = 5;
        for (const char* opt : {"adam", "de"}) {
            j.optimizerName = opt;
            BatchRunner br; br.setJobs({j});
            QObject::connect(&br, &BatchRunner::jobCompleted, [opt](int, BatchJobResult r) {
                const bool reg = r.task == TaskType::Regression;
                const double v = reg ? r.valRMSE : r.valErrorPct;
                std::printf("M2 %s %-5s | success %d  val %s %.4g  test %s %.4g\n", qPrintable(r.datasetName), opt, r.success,
                            reg ? "RMSE" : "err%", v, reg ? "RMSE" : "err%", reg ? r.testRMSE : r.testErrorPct);
                if (!r.success || v < 0) ++bad;
            });
            br.run();
        }
    }

    std::printf(bad ? "FAILED\n" : "ALL OK\n");
    return bad;
}
