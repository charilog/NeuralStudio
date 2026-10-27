// Runs the reference models on one fold and prints test error + chosen hyper-parameters.
//   baselines_check <dataset.train> [ids comma-separated]
#include "core/ge/Baselines.h"
#include "core/dataset/DatasetLoader.h"
#include <QCoreApplication>
#include <cstdio>
using namespace NeuralStudio;

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    if (argc < 2) return 2;
    DatasetBundle b = DatasetLoader::loadBundle(argv[1]);
    if (!b.train || !b.test) { std::fprintf(stderr, "cannot load\n"); return 3; }
    b.train->computeStatistics(); b.test->computeStatistics();
    const QStringList ids = argc > 2 ? QString(argv[2]).split(',') : geBaselineIds();
    std::atomic<bool> stop{false};
    BaselineConfig c; c.seed = 7;
    const auto rows = geRunBaselines(*b.train, b.test.get(), ids, c, stop, nullptr);
    for (const auto& r : rows)
        std::printf("RESULT %s train %.6g test %.6g params %d evals %lld secs %.2f | %s\n", qPrintable(r.id), r.trainErr,
                    r.testErr, r.params, static_cast<long long>(r.evals), r.seconds, qPrintable(r.detail));
    return 0;
}
