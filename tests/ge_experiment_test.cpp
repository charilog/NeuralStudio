// Headless test of experiment files and overall tables.
//   ge_experiment_test <tenfolding dir> <out dir> [raw.csv]
#include "core/ge/GEBatch.h"
#include "core/ge/GEExperiment.h"
#include "core/ge/GEReport.h"
#include "core/ge/Baselines.h"
#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <cstdio>
using namespace NeuralStudio;

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { std::printf("FAIL: %s\n", msg); ++fails; } else std::printf("ok   %s\n", msg); } while (0)

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    if (argc < 3) return 2;
    const QString dataDir = argv[1], outDir = argv[2];
    QDir().mkpath(outDir);

    // 1) benchmark template -> file -> load (round trip)
    GEExperiment b = geBenchmarkExperiment();
    QString err; QStringList warns;
    const QString path = outDir + "/bench64.json";
    CHECK(geSaveExperiment(path, b, err), "save benchmark experiment");
    GEExperiment l;
    CHECK(geLoadExperiment(path, l, err, warns), "load benchmark experiment");
    CHECK(warns.isEmpty(), "no warnings on a clean file");
    CHECK(l.classification == b.classification && l.regression == b.regression, "datasets survive the round trip");
    CHECK(l.options == b.options, "options survive the round trip");
    CHECK(l.models.size() == 7 + geAdaptiveMethodIds().size() + geBaselineIds().size() && l.runModels().size() == 7 + geBaselineIds().size(), "GE models + reference models; the self-adaptive rows listed, off");
    std::printf("     %d classification + %d regression datasets\n", int(l.classification.size()), int(l.regression.size()));

    // 2) every dataset is read with the task the file says
    int mism = 0;
    for (const auto* list : {&l.classification, &l.regression})
        for (const QString& ds : *list) {
            QString d; const QString t = geDetectDatasetTask(dataDir, ds, &d);
            const QString want = list == &l.classification ? "classification" : "regression";
            if (t != want) { ++mism; std::printf("     mismatch %s: %s (%s)\n", qPrintable(ds), qPrintable(t), qPrintable(d)); }
        }
    CHECK(mism == 0, "all 64 datasets have the declared task");
    QString d;
    CHECK(geDetectDatasetTask(dataDir, "plastic", &d) == "classification", "plastic is read as classification (excluded on purpose)");

    // 3) the command built from the file parses, and --experiment gives the same options
    GEBatchOptions o1, o2;
    QString e1, e2;
    const QString cmd = geExperimentToCommand(l);
    CHECK(parseBatchArgs(splitBatchCommandLine(cmd), QDir::currentPath(), o1, e1), "command from the file parses");
    CHECK(parseBatchArgs({"--experiment", path}, QDir::currentPath(), o2, e2), "--experiment parses");
    CHECK(o1.datasets == o2.datasets && o1.args.value("--total-budget") == "600000" &&
          o2.args.value("--total-budget") == "600000", "both give the same datasets and budget");
    GEBatchOptions o3; QString e3;
    CHECK(parseBatchArgs({"--experiment", path, "--seeds", "3"}, QDir::currentPath(), o3, e3) && o3.seeds == 3,
          "command-line options override the file");

    // 4) wrong BNF flag, unknown model, skipped model
    {
        QFile f(outDir + "/bad.json");
        f.open(QIODevice::WriteOnly);
        f.write(R"({"data_dir":"x","output":"y","datasets":{"classification":["pima"]},
                   "options":{"total-budget":1000},
                   "models":[{"id":"grid_T","bnf":true},{"id":"ge1_T","bnf":true},{"id":"nonsense"},
                             {"id":"ge1_T_nobounds","run":false}]})");
        f.close();
        GEExperiment x; QStringList w;
        CHECK(geLoadExperiment(outDir + "/bad.json", x, err, w), "file with problems still loads");
        CHECK(w.size() >= 2, "warnings for the wrong BNF flag and the unknown model");
        for (const QString& s : w) std::printf("     warning: %s\n", qPrintable(s));
        bool gridBnf = true; for (const auto& m : x.models) if (m.id == "grid_T") gridBnf = m.bnf;
        CHECK(!gridBnf, "grid_T corrected to bnf = false");
        CHECK(geExperimentToCommand(x).contains("--methods grid_T,ge1_T"), "only the models that run are passed with --methods");
    }
    {
        QFile f(outDir + "/broken.json"); f.open(QIODevice::WriteOnly); f.write("{ \"data_dir\": "); f.close();
        GEExperiment x; QStringList w;
        CHECK(!geLoadExperiment(outDir + "/broken.json", x, err, w), "broken JSON is rejected");
        std::printf("     error: %s\n", qPrintable(err));
    }

    // 5) overall tables from an existing raw file
    if (argc > 3) {
        QVector<GERawRow> rows;
        CHECK(geLoadRawCsv(argv[3], rows, err), "load raw results");
        const GEOverall ov = geBuildOverall(rows);
        CHECK(!ov.isEmpty(), "overall tables built");
        std::printf("%s", qPrintable(geOverallText(ov)));
        try { geWriteOverallXlsx(ov, outDir + "/overall.xlsx"); CHECK(true, "overall xlsx written"); }
        catch (...) { CHECK(false, "overall xlsx written"); }
    }
    std::printf(fails ? "\n%d FAILED\n" : "\nALL OK\n", fails);
    return fails ? 1 : 0;
}
