// End-to-end smoke test of GEStudioWorker (synchronous, no GUI).
//   ./ge_worker_smoke <dataset.train> [ge|pge] [gens] [pop]
#include "core/ge/GEStudioWorker.h"
#include <QCoreApplication>
#include <cstdio>
using namespace NeuralStudio;

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    if (argc < 2) { std::printf("usage: %s dataset [ge|pge] [gens] [pop]\n", argv[0]); return 2; }
    GEStudioConfig c;
    c.datasetPaths << argv[1];
    c.genotype = argc > 2 ? argv[2] : "ge";
    c.gens = argc > 3 ? atoi(argv[3]) : 6;
    c.pop  = argc > 4 ? atoi(argv[4]) : 16;
    const bool defaults = argc > 5 && QString(argv[5]) == "defaults";
    if (!defaults) { c.innerPop = 20; c.innerIters = 25; c.finalPop = 40; c.finalIters = 60; c.topK = 2; c.maxFeatures = 2; }
    c.seed = 7;
    GEStudioWorker w; w.setConfig(c);
    QObject::connect(&w, &GEStudioWorker::datasetStarted, [](int, int, QString n, QString i) {
        std::printf("dataset %s: %s\n", qPrintable(n), qPrintable(i)); });
    QObject::connect(&w, &GEStudioWorker::generationDone,
        [](int g, int G, double b, QString d, double m, double dv, int u, int h, qint64 calls) {
        std::printf("gen %2d/%d best=%.3f mean=%.2f div=%.2f uniq=%d hits=%d calls=%lld  %s\n",
            g, G, b, m, dv, u, h, calls, qPrintable(d)); });
    QObject::connect(&w, &GEStudioWorker::finalRowDone, [](int r, int n, QString l) {
        std::printf("  final %d/%d %s\n", r, n, qPrintable(l)); });
    QObject::connect(&w, &GEStudioWorker::datasetFinished, [](GEDatasetResult r) {
        std::printf("\n== %s  cls=%d test=%d searchCalls=%lld unique=%d hits=%d  %.1fs\n",
            qPrintable(r.dsName), r.isCls, r.hasTest, (long long)r.searchCalls, r.uniqueEvals, r.cacheHits, r.searchSeconds);
        for (auto& p : r.top) std::printf("  top%d fit=%.3f train=%.3f val=%.3f params=%d in=%d  %s\n",
            p.rank, p.fitness, p.trainErr, p.valErr, p.params, p.inputs, qPrintable(p.desc));
        for (auto& x : r.rows) std::printf("  ROW %-48s train=%.3f test=%.3f val=%.3f params=%d evals=%lld %.1fs\n",
            qPrintable(x.label), x.trainErr, x.testErr, x.valErr, x.params, (long long)x.evals, x.seconds);
        std::printf("%s\nbest program: %s\n", qPrintable(r.probsReport), qPrintable(r.bestProgram)); });
    QObject::connect(&w, &GEStudioWorker::errorOccurred, [](QString m) { std::printf("ERROR: %s\n", qPrintable(m)); });
    w.run();
    return 0;
}
