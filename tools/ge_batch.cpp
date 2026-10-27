// ge_batch — run GE Studio over many datasets x folds x configurations x seeds,
// write raw CSV (resumable), a summary CSV/XLSX and paired statistical tests.
//
//   ge_batch --dir sample_data/tenfolding --datasets wine,airfoil --folds 1-10
//            --configs unified-ge,archbounds-ge,archonly-ge --seeds 3 --out results/run1
//
// The engine lives in src/core/ge/GEBatch.cpp and is shared with the GE Studio panel,
// where the same command can be pasted and run without a console window.
#include "core/ge/GEBatch.h"
#include <QCoreApplication>
#include <QDir>
#include <QTextStream>

using namespace NeuralStudio;

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    QTextStream out(stdout), err(stderr);

    QStringList args;
    for (int i = 1; i < argc; ++i) args << QString::fromLocal8Bit(argv[i]);

    GEBatchOptions opt;
    QString error;
    if (args.isEmpty()) { out << batchUsage(); return 2; }
    if (!parseBatchArgs(args, QDir::currentPath(), opt, error)) {
        err << "ge_batch: " << error << "\n\n" << batchUsage();
        return 2;
    }

    GEBatchRunner runner;
    runner.setOptions(opt);
    int rc = 0;
    QObject::connect(&runner, &GEBatchRunner::message, [&](QString m) { out << m << "\n"; out.flush(); });
    QObject::connect(&runner, &GEBatchRunner::errorOccurred, [&](QString m) { err << "ERROR: " << m << "\n"; rc = 1; });
    runner.run();   // synchronous: signals are delivered directly in this thread
    return rc;
}
