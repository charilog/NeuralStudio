// Checks NSProblem::evalAndGradParallel against central finite differences and against evalAndGrad
// (binary, multi-class and regression datasets).   ./grad_check <file.train> ...
#include "core/optimizers/NSProblem.h"
#include "core/dataset/DatasetLoader.h"
#include "core/nn/NeuralNetwork.h"
#include "core/nn/Trainer.h"
#include "core/nn/RBFTrainer.h"
#include "core/nn/ModelSerializer.h"
#include "core/optimizers/lbfgs.h"
#include "core/optimizers/simann.h"
#include "core/optimizers/neldermead.h"
#include <memory>
#include <QDir>
#include <QFile>
#include <QTemporaryDir>
#include <QTextStream>
#include <map>
#include <QCoreApplication>
#include <cmath>
#include <cstdio>
#include <random>
using namespace NeuralStudio;
int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    int bad = 0;
    for (int f = 1; f < argc; ++f) {
        auto b = DatasetLoader::loadBundle(argv[f]);
        auto ds = b.train; ds->computeStatistics();
        NetworkConfig cfg; cfg.task = detectTask(ds.get()); cfg.hiddenSizes = {6, 4};
        cfg.hiddenActivations = {Activation::Tanh, Activation::Sigmoid};
        auto uv = ds->uniqueOutputValues(0);
        const int out = cfg.task == TaskType::MultiClassClassification ? static_cast<int>(uv.size()) : 1;
        NeuralNetwork net; net.build(ds->inputCount, out, cfg); net.setNormalization(ds->inputStats);
        if (cfg.task != TaskType::Regression) { net.setClassValues(uv); net.setOutputMapping(uv.front(), uv.back()); }
        net.fitOutputScaling(ds.get());   // regression: standardised output layer
        NSProblem p(&net, ds.get(), 2.0, 0.2);
        std::mt19937 rng(5); std::uniform_real_distribution<double> U(-1, 1);
        std::vector<double> x(p.dimension()); for (double& v : x) v = U(rng);
        auto [l1, g1] = p.evalAndGradParallel(x);
        auto [l2, g2] = p.evalAndGrad(x);       // serial backprop through Layer
        const double l0 = p.evaluate(x);
        double maxSer = 0;
        for (int i = 0; i < p.dimension(); ++i)
            maxSer = std::max(maxSer, std::fabs(g1[i] - g2[i]) / std::max(1e-12, std::fabs(g1[i]) + std::fabs(g2[i])));
        double maxRel = 0;
        for (int i = 0; i < p.dimension(); i += std::max(1, p.dimension() / 25)) {
            std::vector<double> a = x, c = x; const double h = 1e-5; a[i] += h; c[i] -= h;
            const double num = (p.evaluate(a) - p.evaluate(c)) / (2 * h);
            maxRel = std::max(maxRel, std::fabs(num - g1[i]) / std::max(1e-6, std::fabs(num) + std::fabs(g1[i])));
        }
        // predictions on the data scale must reproduce the loss
        p.evaluate(x);   // the finite differences left other weights in the network
        std::vector<std::vector<double>> pr, tg; p.predictions(false, pr, tg);
        double lp = 0; for (size_t r = 0; r < pr.size(); ++r) lp += p.predLoss(pr[r], tg[r]);
        lp /= std::max<size_t>(1, pr.size());
        std::printf("%s task %d: loss parallel %.8g serial %.8g evaluate %.8g predictions %.8g, max rel. gradient error %.2e (FD) %.2e (serial)\n",
                    argv[f], static_cast<int>(cfg.task), l1, l2, l0, lp, maxRel, maxSer);
        auto same = [&](double a) { return std::fabs(a - l0) <= 1e-9 * std::max(1.0, std::fabs(l0)); };
        if (!same(l1) || !same(l2) || !same(lp) || maxRel > 1e-4 || maxSer > 1e-9) ++bad;

        // Trainer (mini-batch SGD and Adam, early stopping on a 20 % split): the test error
        // must beat the trivial predictor (majority class / train mean)
        if (b.test) {
            auto te = b.test;
            double trivial = 0;
            if (cfg.task == TaskType::Regression) {
                double m = 0; for (int r = 0; r < ds->sampleCount; ++r) m += ds->outputAt(r, 0); m /= ds->sampleCount;
                for (int r = 0; r < te->sampleCount; ++r) { const double d = te->outputAt(r, 0) - m; trivial += d * d; }
                trivial /= te->sampleCount;
            } else {
                std::map<int, int> cnt; for (int r = 0; r < ds->sampleCount; ++r) ++cnt[net.classIndex(ds->outputAt(r, 0))];
                int maj = 0, best = -1; for (auto [c, n] : cnt) if (n > best) { best = n; maj = c; }
                for (int r = 0; r < te->sampleCount; ++r) trivial += net.classIndex(te->outputAt(r, 0)) != maj;
                trivial = 100.0 * trivial / te->sampleCount;
            }
            for (OptimizerType ot : {OptimizerType::SGD, OptimizerType::Adam}) {
                NeuralNetwork n2; n2.build(ds->inputCount, out, cfg); n2.setNormalization(ds->inputStats);
                if (cfg.task != TaskType::Regression) { n2.setClassValues(uv); n2.setOutputMapping(uv.front(), uv.back()); }
                n2.fitOutputScaling(ds.get());
                TrainerConfig tc; tc.epochs = 150; tc.batchSize = 16; tc.optimizer = ot;
                tc.learningRate = ot == OptimizerType::SGD ? 0.05 : 0.005; tc.earlyStoppingPatience = 20;
                Trainer tr; tr.setConfig(tc); tr.setNetwork(&n2); tr.setDataset(ds.get()); tr.run();
                double err = 0;
                for (int r = 0; r < te->sampleCount; ++r) {
                    const auto y = n2.predict(te->inputRow(r));
                    if (cfg.task == TaskType::Regression) { const double d = y[0] - te->outputAt(r, 0); err += d * d; }
                    else if (cfg.task == TaskType::BinaryClassification) err += (y[0] >= 0.5 ? 1 : 0) != (te->outputAt(r, 0) > 0.5 * (uv.front() + uv.back()) ? 1 : 0);
                    else err += static_cast<int>(std::max_element(y.begin(), y.end()) - y.begin()) != n2.classIndex(te->outputAt(r, 0));
                }
                err = cfg.task == TaskType::Regression ? err / te->sampleCount : 100.0 * err / te->sampleCount;
                std::printf("   Trainer %-4s: test %s %.4g   (trivial predictor %.4g)\n", ot == OptimizerType::SGD ? "SGD" : "Adam",
                            cfg.task == TaskType::Regression ? "MSE" : "error %", err, trivial);
                if (!(err < trivial)) ++bad;
            }
        }
    }
    // ── RBF network (Mode 1): must beat the trivial predictor on the test file ──
    for (int f = 1; f < argc; ++f) {
        auto b = DatasetLoader::loadBundle(argv[f]);
        if (!b.test) continue;
        auto ds = b.train; ds->computeStatistics();
        const TaskType task = detectTask(ds.get());
        auto uv = ds->uniqueOutputValues(0);
        const int nOut = task == TaskType::MultiClassClassification ? static_cast<int>(uv.size()) : 1;
        auto rbf = std::make_shared<RBFNetwork>();
        RBFConfig rc; rc.nCenters = 20; rbf->init(ds->inputCount, nOut, task, rc);
        RBFTrainer rt; rt.setNetwork(rbf); rt.setDataset(ds.get()); rt.run();
        double err = 0, trivial = 0;
        const auto& te = *b.test;
        if (task == TaskType::Regression) {
            double m = 0; for (int r = 0; r < ds->sampleCount; ++r) m += ds->outputAt(r, 0); m /= ds->sampleCount;
            for (int r = 0; r < te.sampleCount; ++r) {
                const double d = rbf->predict(te.inputRow(r))[0] - te.outputAt(r, 0); err += d * d;
                trivial += (te.outputAt(r, 0) - m) * (te.outputAt(r, 0) - m);
            }
        } else {
            std::map<int, int> cnt; for (int r = 0; r < ds->sampleCount; ++r) ++cnt[rbf->classIndex(ds->outputAt(r, 0))];
            int maj = 0, bestN = -1; for (auto [c, n] : cnt) if (n > bestN) { bestN = n; maj = c; }
            for (int r = 0; r < te.sampleCount; ++r) {
                const auto y = rbf->predict(te.inputRow(r));
                const int pc = nOut == 1 ? (y[0] >= 0.5 ? 1 : 0) : static_cast<int>(std::max_element(y.begin(), y.end()) - y.begin());
                const int tc = rbf->classIndex(te.outputAt(r, 0));
                err += pc != tc; trivial += tc != maj;
            }
        }
        std::printf("RBF %s task %d: test %s %.4g (trivial %.4g)\n", argv[f], static_cast<int>(task),
                    task == TaskType::Regression ? "SSE" : "errors", err, trivial);
        if (!(err < trivial)) ++bad;
    }

    // ── data layer: .trn/.tst roles, CSV with a missing field, model round trip ──
    {
        QTemporaryDir tmp;
        auto write = [&](const QString& name, const QString& text) {
            QFile fl(tmp.filePath(name)); fl.open(QIODevice::WriteOnly | QIODevice::Text); QTextStream(&fl) << text; };
        write("d.trn", "1\n3\n0.1 0\n0.2 0\n0.9 1\n");
        write("d.tst", "1\n2\n0.15 0\n0.8 1\n");
        auto bt = DatasetLoader::loadBundle(tmp.filePath("d.tst"));   // clicked the TEST file
        const bool roleOk = bt.train && bt.test && bt.train->sampleCount == 3 && bt.test->sampleCount == 2;
        std::printf("companion roles (.tst clicked): train %d rows, test %d rows -> %s\n",
                    bt.train ? bt.train->sampleCount : -1, bt.test ? bt.test->sampleCount : -1, roleOk ? "ok" : "WRONG");
        if (!roleOk) ++bad;
        write("m.csv", "a,b,y\n1,,0\n2,3,1\n4,5,0\n");
        auto c = DatasetLoader::load(tmp.filePath("m.csv"));
        const bool csvOk = c->sampleCount == 3 && std::isnan(c->inputs[0][1]);
        std::printf("CSV with an empty field: %d rows, missing stored as NaN -> %s\n", c->sampleCount, csvOk ? "ok" : "WRONG");
        if (!csvOk) ++bad;
        // missing input -> train mean; serializer round trip keeps predictions (incl. output scaling)
        auto hb = DatasetLoader::loadBundle(argv[argc - 1]);
        auto hd = hb.train; hd->computeStatistics();
        NetworkConfig nc; nc.task = detectTask(hd.get()); nc.hiddenSizes = {5};
        auto huv = hd->uniqueOutputValues(0);
        NeuralNetwork n1; n1.build(hd->inputCount, nc.task == TaskType::MultiClassClassification ? static_cast<int>(huv.size()) : 1, nc);
        n1.setNormalization(hd->inputStats); n1.fitOutputScaling(hd.get());
        if (nc.task != TaskType::Regression) { n1.setClassValues(huv); n1.setOutputMapping(huv.front(), huv.back()); }
        const QString mp = tmp.filePath("m.json");
        ModelSerializer::save(n1, mp);
        NeuralNetwork n2; ModelSerializer::load(n2, mp);
        auto row = hd->inputRow(0);
        double md = 0; for (size_t i = 0; i < n1.predict(row).size(); ++i) md = std::max(md, std::fabs(n1.predict(row)[i] - n2.predict(row)[i]));
        auto withNaN = row; withNaN[0] = std::nan("");
        auto meanRow = row; meanRow[0] = hd->inputStats[0].mean;
        const double dn = std::fabs(n2.predict(withNaN)[0] - n2.predict(meanRow)[0]);
        std::printf("model round trip: max |diff| %.2e; NaN input = train mean: |diff| %.2e\n", md, dn);
        if (md > 1e-9 || dn > 1e-9) ++bad;
    }

    // ── v17 checks ─────────────────────────────────────────────────────────────
    // (1) binary BCE from the logit: a saturated, confidently wrong output keeps its gradient.
    //     All weights 0 and output bias -40: z = -40 for every row, so the exact loss is
    //     40 * (fraction of class 1) and d loss / d bias = mean(p - t) = -(fraction of class 1).
    //     With the clipped BCE the loss was capped at 16.1 per row and the gradient was ~1e-11.
    for (int f = 1; f < argc; ++f) {
        auto b = DatasetLoader::loadBundle(argv[f]);
        auto ds = b.train; ds->computeStatistics();
        if (detectTask(ds.get()) != TaskType::BinaryClassification) continue;
        auto uv = ds->uniqueOutputValues(0);
        NetworkConfig cfg; cfg.task = TaskType::BinaryClassification; cfg.hiddenSizes = {4}; cfg.hiddenActivations = {Activation::Tanh};
        NeuralNetwork net; net.build(ds->inputCount, 1, cfg); net.setNormalization(ds->inputStats);
        net.setClassValues(uv); net.setOutputMapping(uv.front(), uv.back());
        NSProblem p(&net, ds.get(), 100.0, 0.0);
        std::vector<double> x(p.dimension(), 0.0); x.back() = -40.0;   // last parameter = output bias
        double pos = 0; for (int r = 0; r < ds->sampleCount; ++r) pos += ds->outputAt(r, 0) > 0.5 * (uv.front() + uv.back());
        pos /= ds->sampleCount;
        const double l0 = p.evaluate(x);
        auto [l1, g1] = p.evalAndGradParallel(x);
        auto [l2, g2] = p.evalAndGrad(x);
        const double expLoss = 40.0 * pos;   // + log1p(exp(-40)) ~ 4e-18
        const bool ok = std::fabs(l0 - expLoss) < 1e-9 * expLoss && std::fabs(l1 - l0) < 1e-9 * l0 && std::fabs(l2 - l0) < 1e-9 * l0 &&
                        std::fabs(g1.back() + pos) < 1e-9 && std::fabs(g2.back() + pos) < 1e-9;
        std::printf("saturated binary output (%s): loss %.6g (exact %.6g), d/d bias %.6g / %.6g (exact %.6g) -> %s\n",
                    argv[f], l0, expLoss, g1.back(), g2.back(), -pos, ok ? "ok" : "WRONG");
        if (!ok) ++bad;
        break;
    }
    // (2) evalAndGrad with dropout configured: no dropout in the gradient (same loss as evaluate)
    {
        auto b = DatasetLoader::loadBundle(argv[1]);
        auto ds = b.train; ds->computeStatistics();
        NetworkConfig cfg; cfg.task = detectTask(ds.get()); cfg.hiddenSizes = {8}; cfg.hiddenActivations = {Activation::Sigmoid};
        cfg.dropoutRate = 0.5;
        auto uv = ds->uniqueOutputValues(0);
        const int out = cfg.task == TaskType::MultiClassClassification ? static_cast<int>(uv.size()) : 1;
        NeuralNetwork net; net.build(ds->inputCount, out, cfg); net.setNormalization(ds->inputStats);
        if (cfg.task != TaskType::Regression) { net.setClassValues(uv); net.setOutputMapping(uv.front(), uv.back()); }
        net.fitOutputScaling(ds.get());
        NSProblem p(&net, ds.get(), 2.0, 0.2);
        std::mt19937 rng(3); std::uniform_real_distribution<double> U(-1, 1);
        std::vector<double> x(p.dimension()); for (double& v : x) v = U(rng);
        const double l0 = p.evaluate(x);
        auto [l1, g1] = p.evalAndGrad(x);
        auto [l2, g2] = p.evalAndGradParallel(x);
        double md = 0; for (size_t i = 0; i < g1.size(); ++i) md = std::max(md, std::fabs(g1[i] - g2[i]));
        const bool ok = std::fabs(l1 - l0) < 1e-12 * std::max(1.0, l0) && md < 1e-12;
        std::printf("dropout 0.5 configured: evaluate %.10g, evalAndGrad %.10g, max |grad diff| %.1e -> %s\n", l0, l1, md, ok ? "ok" : "WRONG");
        if (!ok) ++bad;
    }
    // (3) L-BFGS with a useless direction (gradient sign flipped): the Armijo search fails; the
    //     iterate must not move uphill and the method must stop evaluating (it took the last
    //     ~3e-8 step and repeated 25 evaluations in every iteration)
    {
        struct Up : optimsolution::Problem {
            std::vector<double> L = std::vector<double>(5, -10.0), U = std::vector<double>(5, 10.0); int c = 0;
            int dimension() const override { return 5; }
            const std::vector<double>& lb() const override { return L; }
            const std::vector<double>& ub() const override { return U; }
            double f(const std::vector<double>& x) const { double s = 0; for (double v : x) s += (v - 1) * (v - 1); return s; }
            double evaluate(const std::vector<double>& x) override { ++c; return f(x); }
            int calls() const override { return c; }
            std::pair<double, std::vector<double>> evalAndGrad(const std::vector<double>& x) override {
                ++c; std::vector<double> g(5); for (int i = 0; i < 5; ++i) g[i] = -2 * (x[i] - 1); return {f(x), g}; }
        } up;
        optimsolution::LBFGS o; o.setMemory(5); o.setup(&up, 1000000, 1, 1);
        o.setInitialPoint(std::vector<double>(5, 3.0)); o.init();
        const double f0 = o.bestF();
        for (int it = 0; it < 1000; ++it) { o.one_iteration(); o.advanceIter(); }
        const bool ok = up.calls() < 100 && o.bestF() <= f0;
        std::printf("L-BFGS with an ascent 'gradient': %d evaluations in 1000 iterations, best %.4g (start %.4g) -> %s\n",
                    up.calls(), o.bestF(), f0, ok ? "ok" : "WRONG");
        if (!ok) ++bad;
    }
    // (4) single-point methods spend the budget the drivers give them (iterations x population)
    {
        struct Sph : optimsolution::Problem {
            std::vector<double> L = std::vector<double>(10, -5.0), U = std::vector<double>(10, 5.0); int c = 0;
            int dimension() const override { return 10; }
            const std::vector<double>& lb() const override { return L; }
            const std::vector<double>& ub() const override { return U; }
            double evaluate(const std::vector<double>& x) override { ++c; double s = 0; for (double v : x) s += v * v; return s; }
            int calls() const override { return c; }
        };
        for (int m = 0; m < 2; ++m) {
            Sph sp;
            std::unique_ptr<optimsolution::Optimizer> o;
            if (m == 0) o = std::make_unique<optimsolution::SimulatedAnnealing>(); else o = std::make_unique<optimsolution::NelderMead>();
            const int iters = 100, pop = 30;
            o->setup(&sp, iters * pop, pop, 5); o->init();
            for (int it = 0; it < iters && !o->done(); ++it) { o->one_iteration(); o->advanceIter(); }
            const bool ok = sp.calls() >= iters * pop * 9 / 10 && sp.calls() <= iters * pop + 2 * 10 + 2;
            std::printf("%s: %d of %d evaluations used in %d iterations -> %s\n", m == 0 ? "SA" : "Nelder-Mead",
                        sp.calls(), iters * pop, iters, ok ? "ok" : "WRONG");
            if (!ok) ++bad;
        }
    }
    // (5) companion "name_train.csv" / "name_test.csv"; a row without a target is removed
    {
        QTemporaryDir tmp;
        auto write = [&](const QString& name, const QString& text) {
            QFile fl(tmp.filePath(name)); fl.open(QIODevice::WriteOnly | QIODevice::Text); QTextStream(&fl) << text; };
        write("iris_train.csv", "a,b,y\n1,2,0\n2,3,1\n4,5,\n5,6,1\n");
        write("iris_test.csv", "a,b,y\n1,2,0\n2,3,1\n");
        auto bt = DatasetLoader::loadBundle(tmp.filePath("iris_test.csv"));   // clicked the test file
        const bool ok = bt.train && bt.test && bt.train->sampleCount == 3 && bt.test->sampleCount == 2 &&
                        bt.train->uniqueOutputValues(0).size() == 2;
        std::printf("companion iris_train.csv / iris_test.csv: train %d rows (1 without target removed), test %d rows -> %s\n",
                    bt.train ? bt.train->sampleCount : -1, bt.test ? bt.test->sampleCount : -1, ok ? "ok" : "WRONG");
        if (!ok) ++bad;
    }

    std::printf(bad ? "FAILED\n" : "ALL OK\n");
    return bad;
}
