// GEReport.cpp — overall result tables of Mode 6 batch experiments (see GEReport.h).
#include "GEReport.h"
#include "core/export/XlsxWriter.h"
#include "core/stats/Statistics.h"
#include <QFile>
#include <QRegularExpression>
#include <QTextStream>
#include <algorithm>
#include <map>
#include <set>

namespace NeuralStudio {

namespace {
const double kNaN = std::numeric_limits<double>::quiet_NaN();

double meanV(const std::vector<double>& v) {
    if (v.empty()) return kNaN;
    double s = 0; for (double x : v) s += x; return s / static_cast<double>(v.size());
}
double sdV(const std::vector<double>& v) {
    if (v.size() < 2) return 0;
    const double m = meanV(v); double s = 0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / static_cast<double>(v.size() - 1));
}
bool sameValue(double a, double b) {
    return std::fabs(a - b) <= 1e-12 * std::max({1.0, std::fabs(a), std::fabs(b)});
}
QString fmt(double v, bool cls) {
    if (std::isnan(v)) return "-";
    return cls ? QString::number(v, 'f', 2) : QString::number(v, 'g', 5);
}
QString fmtP(double p) {
    if (std::isnan(p)) return "-";
    return p < 0.001 ? QString("<0.001") : QString::number(p, 'f', 3);
}
} // namespace

// ── Method catalog ────────────────────────────────────────────────────────────
const QVector<GEMethodInfo>& geMethodCatalog() {
    static const QVector<GEMethodInfo> cat = {
        {"standard",        "Standard",              false, "Fixed network (baseline hidden/activation), trained with the final budget only"},
        {"standard_T",      "Standard (T)",          false, "Fixed network trained with the whole total budget T"},
        {"standard_eq",     "Standard (eq)",         false, "Fixed network trained with the GE search budget + final budget"},
        {"grid_T",          "Grid search (T)",       false, "Best of 16 classic architectures (same search share), trained with the rest of T"},
        {"grid_eq",         "Grid search (eq)",      false, "Best of 16 classic architectures searched with the GE search budget"},
        {"adam",            "ADAM",                  false, "MLP with H=10 sigmoid nodes trained with mini-batch Adam"},
        {"bfgs",            "BFGS",                  false, "MLP with H=10 sigmoid nodes trained with limited-memory BFGS"},
        {"genetic",         "GENETIC",               false, "MLP with H=10 sigmoid nodes trained with a genetic algorithm (500 x 200)"},
        {"neat",            "NEAT",                  false, "NeuroEvolution of Augmenting Topologies"},
        {"rbf",             "RBF",                   false, "RBF network with H=10 Gaussian centres (k-means++, ridge output layer)"},
        {"knn",             "kNN",                   false, "k nearest neighbours (k chosen on a validation split)"},
        {"linear",          "Logistic / ridge",      false, "Logistic regression (classification) or ridge regression (regression)"},
        {"rf",              "Random forest",         false, "Random forest of CART trees"},
        {"svm",             "SVM",                   false, "Support vector machine with RBF kernel (C-SVC / epsilon-SVR)"},
        {"standard_T_ad",   "Standard (T) adaptive", false, "Fixed network, total budget T, self-adaptive training (global phase until it stalls, bound-respecting L-BFGS with validation early stopping and restarts)"},
        {"std_ens_T",       "Standard ensemble (T)", false, "Ensemble of fixed networks (as many as the GE ensemble), total budget T shared, self-adaptive training"},
        {"gnn",             "GNN",                   true,  "The BNF-guided training of the BNF paper (H=10, bound commands evolved by GE)"},
        {"ge1",             "GE #1",                 true,  "Best BNF phenotype, trained with the final budget only"},
        {"ge1_T",           "GE #1 (T)",             true,  "Best BNF phenotype (architecture + bounds + features), search + training = T"},
        {"ge1_T_nofeat",    "GE #1 (T) no features", true,  "Ablation: the same phenotype without its constructed features"},
        {"ge1_T_nobounds",  "GE #1 (T) no bounds",   true,  "Ablation: the same phenotype without its bound commands"},
        {"ge1_eq",          "GE #1 (eq)",            true,  "Best BNF phenotype retrained with the equal total budget"},
        {"ge1_nobounds",    "GE #1 no bounds",       true,  "Ablation: GE #1 without bound commands (final budget)"},
        {"ge1_nobounds_eq", "GE #1 (eq) no bounds",  true,  "Ablation: GE #1 (eq) without bound commands"},
        {"ge1_T_ad",        "GE #1 (T) adaptive",    true,  "Noise-aware selection of one phenotype (one-standard-error rule, simplest) + self-adaptive training, total T"},
        {"ens_T",           "GE ensemble (T)",       true,  "Greedy ensemble of archive phenotypes, grown while the validation gain exceeds one standard error; self-adaptive training of every member, total T"},
        {"ens_T_nofeat",    "GE ensemble (T) no features", true, "Ablation: the same ensemble members without their constructed features"},
        {"ens_T_nobounds",  "GE ensemble (T) no bounds",   true, "Ablation: the same ensemble members without their bound commands"},
        {"ens_T_div",       "GE ensemble (T) diverse",     true, "Development: diversity-aware noise-aware ensemble selection"},
        {"ens_T_cv",        "GE ensemble (T) CV",          true, "Development: diverse selection, every member a 5-fold cross-validation committee"},
        {"std_ens_T_cv",    "Standard ensemble (T) CV",    false, "Development control: standard networks as 5-fold CV committees, budget T"},
        {"sage",            "SAGE",                        true,  "Final method: grammar with periodic input embedding, diverse noise-aware ensemble selection, every member a 5-fold CV committee, total T"},
        {"sage_noemb",      "SAGE no embedding",           true,  "Ablation: the same members without the periodic input embedding (same data splits)"},
        {"sage_nofeat",     "SAGE no features",            true,  "Ablation: the same members without their constructed features (same data splits)"},
        {"sage_nobounds",   "SAGE no bounds",              true,  "Ablation: the same members without their bound commands (same data splits)"},
        {"sage_one",        "SAGE one member",             true,  "Ablation: the best member alone as a 5-fold CV committee (effect of the ensemble)"},
        {"sage_std",        "SAGE standard nets",          false, "Ablation: as many standard networks as SAGE members, same CV committees and data splits, budget T (effect of the grammar)"},
    };
    return cat;
}

GEMethodInfo geMethodInfo(const QString& id, bool* known) {
    for (const auto& m : geMethodCatalog())
        if (m.id == id) { if (known) *known = true; return m; }
    static const QRegularExpression re("^ge(\\d+)$");
    const auto mt = re.match(id);
    if (mt.hasMatch()) {
        if (known) *known = true;
        return {id, QString("GE #%1").arg(mt.captured(1)), true,
                QString("BNF phenotype ranked #%1, trained with the final budget").arg(mt.captured(1))};
    }
    if (known) *known = false;
    return {id, id, id.startsWith("ge"), QString()};
}

// ── Raw CSV ───────────────────────────────────────────────────────────────────
bool geLoadRawCsv(const QString& path, QVector<GERawRow>& rows, QString& error) {
    rows.clear();
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) { error = "Cannot open " + path; return false; }
    QTextStream t(&f);
    const QString header = t.readLine().trimmed();
    if (!header.startsWith("dataset,fold,config,seed,method,task,train_err,test_err")) {
        error = "Not a ge_batch raw results file (unexpected header): " + path;
        return false;
    }
    int lineNo = 1, bad = 0;
    while (!t.atEnd()) {
        const QString line = t.readLine().trimmed();
        ++lineNo;
        if (line.isEmpty()) continue;
        const QStringList c = line.split(',');
        if (c.size() < 17) { ++bad; continue; }
        GERawRow r;
        bool ok = true, okAll = true;
        r.ds = c[0]; r.fold = c[1].toInt(&ok); okAll &= ok; r.cfg = c[2]; r.seed = c[3].toInt(&ok); okAll &= ok;
        r.method = c[4]; r.cls = c[5] == "cls";
        r.trainErr = c[6].toDouble(&ok); okAll &= ok;
        r.testErr = c[7].toDouble(&ok); okAll &= ok;
        r.params = c[8].toInt(); r.evals = c[9].toLongLong(); r.secs = c[10].toDouble();
        r.searchEvals = c[11].toLongLong(); r.searchSecs = c[12].toDouble(); r.gensRun = c[13].toInt();
        r.uniqueNets = c[14].toInt(); r.cacheHits = c[15].toInt(); r.testN = c[16].toInt();
        if (!okAll || !std::isfinite(r.testErr)) { ++bad; continue; }
        rows.append(r);
    }
    if (rows.isEmpty()) { error = "No result rows in " + path; return false; }
    if (bad > 0) error = QString("%1 malformed line(s) skipped").arg(bad);
    return true;
}

// ── Scale-free paired comparison ──────────────────────────────────────────────
QVector<double> geHolm(const QVector<double>& p) {
    QVector<int> idx;
    for (int i = 0; i < p.size(); ++i) if (!std::isnan(p[i])) idx << i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return p[a] < p[b]; });
    QVector<double> out(p.size(), kNaN);
    const int m = idx.size();
    double run = 0;
    for (int k = 0; k < m; ++k) {
        run = std::max(run, std::min(1.0, (m - k) * p[idx[k]]));
        out[idx[k]] = run;
    }
    return out;
}

namespace {
// (method - reference) / mean(|method|, |reference|); 0 when both are 0. In [-2, 2].
double relDiff(double ref, double m) {
    const double den = 0.5 * (std::fabs(ref) + std::fabs(m));
    if (den < 1e-300 || sameValue(ref, m)) return 0.0;
    return (m - ref) / den;
}
struct PairStats { int n = 0, better = 0, ties = 0, worse = 0; double med = kNaN, geo = kNaN, p = kNaN; };
// rd values of one comparison over datasets -> counts (reference better = rd > 0), median %, geometric ratio, p
PairStats pairStats(const std::vector<double>& rd, const std::vector<double>& ratioLog) {
    PairStats s;
    s.n = static_cast<int>(rd.size());
    for (double v : rd) { if (v == 0.0) ++s.ties; else if (v > 0) ++s.better; else ++s.worse; }
    if (!rd.empty()) {
        std::vector<double> v = rd;
        std::sort(v.begin(), v.end());
        const size_t n = v.size();
        s.med = 100.0 * (n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]));
        const WilcoxonResult w = Statistics::wilcoxon(rd, std::vector<double>(rd.size(), 0.0));
        if (w.valid) s.p = w.pValue;
    }
    if (!ratioLog.empty()) s.geo = std::exp(meanV(ratioLog));
    return s;
}
void comparison(const GEOverallTable& tb, int ref, int j, std::vector<double>& rd, std::vector<double>& lg,
                const QStringList& exclude = {}, int minFolds = 0, QStringList* excluded = nullptr) {
    for (int i = 0; i < tb.datasets.size(); ++i) {
        bool full = true;
        for (int k = 0; k < tb.methods.size(); ++k) full = full && !std::isnan(tb.value[i][k]);
        if (!full) continue;
        if (exclude.contains(tb.datasets[i]) || tb.folds[i] < minFolds) {
            if (excluded && !excluded->contains(tb.datasets[i])) *excluded << tb.datasets[i];
            continue;
        }
        const double a = tb.value[i][ref], b = tb.value[i][j];
        rd.push_back(relDiff(a, b));
        if (a > 0 && b > 0) lg.push_back(std::log(b / a));
    }
}
struct HypDef { const char* id; const char* method; const char* meaning; };
QVector<HypDef> hypothesesFor(const QString& ref) {
    if (ref == "sage")   // primary claim P1-P10 (every reference model), ablations A1-A5; one Holm family
        return {{"P1", "rf", "random forest"},
                {"P2", "svm", "support vector machine (RBF kernel)"},
                {"P3", "knn", "k nearest neighbours"},
                {"P4", "linear", "logistic / ridge regression"},
                {"P5", "adam", "MLP trained with ADAM"},
                {"P6", "bfgs", "MLP trained with BFGS"},
                {"P7", "genetic", "MLP trained with a genetic algorithm"},
                {"P8", "neat", "NEAT"},
                {"P9", "rbf", "RBF network"},
                {"P10", "gnn", "BNF-guided training of the BNF paper (GNN)"},
                {"A1", "sage_std", "standard networks instead of the grammar, same ensemble, CV and budget"},
                {"A2", "sage_noemb", "the same members without the periodic input embedding"},
                {"A3", "sage_nofeat", "the same members without constructed features"},
                {"A4", "sage_nobounds", "the same members without bound commands"},
                {"A5", "sage_one", "the best member alone (effect of the ensemble)"}};
    if (ref == "ens_T")
        return {{"H1", "std_ens_T", "ensemble of standard networks, same size, same total budget T"},
                {"H2", "ge1_T", "single GE network of the first version (same search)"},
                {"H3", "ens_T_nobounds", "the same ensemble without bound commands"},
                {"H4", "ens_T_nofeat", "the same ensemble without constructed features"},
                {"H5", "gnn", "BNF-guided training of the BNF paper (GNN)"},
                {"H6", "ge1_T_ad", "one network with the same selection and training (effect of the ensemble)"}};
    if (ref == "ge1_T")
        return {{"H1", "standard_T", "fixed network, same total budget T"},
                {"H2", "grid_T", "grid search of classic architectures, same T"},
                {"H3", "ge1_T_nobounds", "same phenotype without its bound commands"},
                {"H4", "ge1_T_nofeat", "same phenotype without its constructed features"},
                {"H5", "gnn", "BNF-guided training of the BNF paper (GNN)"}};
    if (ref == "ge1_eq")
        return {{"H1", "standard_eq", "fixed network, equal total budget"},
                {"H2", "grid_eq", "grid search of classic architectures, equal budget"},
                {"H3", "ge1_nobounds_eq", "same phenotype without its bound commands"},
                {"H5", "gnn", "BNF-guided training of the BNF paper (GNN)"}};
    return {{"H1", "standard", "fixed network, final budget"},
            {"H3", "ge1_nobounds", "same phenotype without its bound commands"},
            {"H5", "gnn", "BNF-guided training of the BNF paper (GNN)"}};
}
} // namespace

// ── Overall tables ────────────────────────────────────────────────────────────
GEOverall geBuildOverall(const QVector<GERawRow>& allRows, const GEReportFilter& filter) {
    GEOverall out;
    QVector<GERawRow> rows;
    rows.reserve(allRows.size());
    for (const GERawRow& r : allRows) if (filter.keeps(r)) rows.append(r);
    // task -> cfg -> ds -> method -> fold -> seed values
    struct Acc { std::map<int, std::vector<double>> byFold; double secs = 0; double params = 0; int n = 0; };
    std::map<QString, std::map<QString, std::map<QString, std::map<QString, Acc>>>> data;
    std::map<QString, std::set<QString>> dsTasks;
    // Reference models are run as the configuration "baselines"; they are shown as extra
    // columns of every grammar configuration of the same task.
    std::map<QString, std::map<QString, std::map<QString, Acc>>> base;   // task -> ds -> method
    for (const GERawRow& r : rows) {
        const QString task = r.cls ? "classification" : "regression";
        Acc& a = r.cfg == "baselines" ? base[task][r.ds][r.method] : data[task][r.cfg][r.ds][r.method];
        a.byFold[r.fold].push_back(r.testErr);
        a.secs += r.secs; a.params += r.params; ++a.n;
        dsTasks[r.ds].insert(task);
    }
    for (auto& tk : base) {
        auto& cfgs = data[tk.first];
        if (cfgs.empty()) { cfgs["baselines"] = tk.second; continue; }
        for (auto& cfgKv : cfgs)
            for (const auto& dsKv : tk.second)
                for (const auto& mKv : dsKv.second) cfgKv.second[dsKv.first][mKv.first] = mKv.second;
    }
    for (const auto& kv : dsTasks)
        if (kv.second.size() > 1) out.warnings << QString("Dataset %1 appears both as classification and regression.").arg(kv.first);

    const QVector<GEMethodInfo>& cat = geMethodCatalog();
    for (const char* taskC : {"classification", "regression"}) {
        const QString task = taskC;
        auto itT = data.find(task);
        if (itT == data.end()) continue;
        for (const auto& cfgKv : itT->second) {
            GEOverallTable tb;
            tb.task = task; tb.config = cfgKv.first;
            // methods present, in catalog order, then unknown ones (ge2.. etc.) by name
            std::set<QString> present;
            for (const auto& dsKv : cfgKv.second) for (const auto& mKv : dsKv.second) present.insert(mKv.first);
            QStringList ordered;
            for (const auto& m : cat) if (present.count(m.id)) ordered << m.id;
            QStringList extra;
            for (const auto& m : present) if (!ordered.contains(m)) extra << m;
            std::sort(extra.begin(), extra.end());
            // non-BNF extras before BNF extras, BNF extras after GE #1
            QStringList nonBnf, bnf;
            for (const QString& m : ordered + extra) (geMethodInfo(m).bnf ? bnf : nonBnf) << m;
            tb.methods = nonBnf + bnf;
            for (const QString& m : tb.methods) tb.bnf << geMethodInfo(m).bnf;
            const int M = tb.methods.size();

            for (const auto& dsKv : cfgKv.second) tb.datasets << dsKv.first;
            tb.seconds.fill(0, M); tb.params.fill(0, M);
            QVector<int> nRuns(M, 0);
            for (const QString& ds : tb.datasets) {
                const auto& mm = cfgKv.second.at(ds);
                QVector<double> v(M, kNaN), s(M, kNaN);
                // folds present for every method of this dataset
                std::set<int> common;
                bool first = true;
                for (int j = 0; j < M; ++j) {
                    auto it = mm.find(tb.methods[j]);
                    if (it == mm.end()) continue;
                    std::set<int> fs;
                    for (const auto& f : it->second.byFold) fs.insert(f.first);
                    if (first) { common = fs; first = false; }
                    else { std::set<int> x; std::set_intersection(common.begin(), common.end(), fs.begin(), fs.end(), std::inserter(x, x.begin())); common = x; }
                }
                for (int j = 0; j < M; ++j) {
                    auto it = mm.find(tb.methods[j]);
                    if (it == mm.end()) continue;
                    std::vector<double> foldMeans;
                    for (const auto& f : it->second.byFold)
                        if (common.count(f.first)) foldMeans.push_back(meanV(f.second));
                    if (foldMeans.empty()) continue;
                    v[j] = meanV(foldMeans); s[j] = sdV(foldMeans);
                    tb.seconds[j] += it->second.secs; tb.params[j] += it->second.params; nRuns[j] += it->second.n;
                }
                tb.value << v; tb.sd << s; tb.folds << static_cast<int>(common.size());
            }
            for (int j = 0; j < M; ++j)
                if (nRuns[j] > 0) { tb.seconds[j] /= nRuns[j]; tb.params[j] /= nRuns[j]; }

            // datasets with every method
            std::vector<int> full;
            for (int i = 0; i < tb.datasets.size(); ++i) {
                bool ok = true;
                for (int j = 0; j < M; ++j) ok = ok && !std::isnan(tb.value[i][j]);
                if (ok) full.push_back(i);
            }
            tb.complete = static_cast<int>(full.size());
            tb.average.fill(kNaN, M); tb.avgRank.fill(kNaN, M);
            tb.wins.fill(0, M); tb.ties.fill(0, M); tb.losses.fill(0, M); tb.wilcoxonP.fill(kNaN, M);
            tb.relDiffMedian.fill(kNaN, M); tb.geoRatio.fill(kNaN, M); tb.wilcoxonRelP.fill(kNaN, M); tb.holmP.fill(kNaN, M);
            if (tb.complete < tb.datasets.size())
                out.warnings << QString("%1 [%2]: %3 of %4 datasets have results for every method; averages, ranks and tests use those.")
                                    .arg(task, tb.config).arg(tb.complete).arg(tb.datasets.size());
            if (!full.empty()) {
                std::vector<std::vector<double>> byMethod(M);
                for (int j = 0; j < M; ++j) {
                    for (int i : full) byMethod[j].push_back(tb.value[i][j]);
                    tb.average[j] = meanV(byMethod[j]);
                }
                if (M >= 2 && tb.complete >= 2) {
                    std::vector<std::string> names;
                    for (const QString& m : tb.methods) names.push_back(m.toStdString());
                    const FriedmanResult fr = Statistics::friedman(names, byMethod);
                    if (fr.valid) {
                        tb.friedmanP = fr.pValue;
                        for (int j = 0; j < M && j < static_cast<int>(fr.avgRanks.size()); ++j) tb.avgRank[j] = fr.avgRanks[j];
                    }
                    const NemenyiResult nr = Statistics::nemenyi(byMethod);
                    if (nr.valid) tb.cd05 = nr.cd05;
                }
                for (const char* r : {"sage", "ens_T", "ge1_T", "ge1_eq", "ge1"})
                    if (tb.methods.contains(r)) { tb.reference = r; break; }
                const int ref = tb.methods.indexOf(tb.reference);
                if (ref >= 0)
                    for (int j = 0; j < M; ++j) {
                        if (j == ref) continue;
                        for (size_t k = 0; k < full.size(); ++k) {
                            const double a = byMethod[ref][k], b = byMethod[j][k];
                            if (sameValue(a, b)) ++tb.ties[j];
                            else if (a < b) ++tb.wins[j];
                            else ++tb.losses[j];
                        }
                        const WilcoxonResult w = Statistics::wilcoxon(byMethod[ref], byMethod[j]);
                        if (w.valid) tb.wilcoxonP[j] = w.pValue;
                        std::vector<double> rd, lg;
                        comparison(tb, ref, j, rd, lg);
                        const PairStats ps = pairStats(rd, lg);
                        tb.relDiffMedian[j] = ps.med; tb.geoRatio[j] = ps.geo; tb.wilcoxonRelP[j] = ps.p;
                    }
                if (ref >= 0) tb.holmP = geHolm(tb.wilcoxonRelP);
            }
            out.tables << tb;
        }
    }

    // hypotheses: per task and over both tasks of the same configuration
    QStringList cfgs;
    for (const GEOverallTable& tb : out.tables) if (!tb.reference.isEmpty() && !cfgs.contains(tb.config)) cfgs << tb.config;
    for (const QString& cfg : cfgs) {
        QString reference;
        for (const GEOverallTable& tb : out.tables) if (tb.config == cfg && !tb.reference.isEmpty()) { reference = tb.reference; break; }
        for (const char* scopeC : {"classification", "regression", "all"}) {
            const QString scope = scopeC;
            QVector<GEHypothesisTest> hs;
            for (const HypDef& h : hypothesesFor(reference)) {
                std::vector<double> rd, lg;
                QStringList excluded;
                bool any = false;
                for (const GEOverallTable& tb : out.tables) {
                    if (tb.config != cfg || tb.reference != reference) continue;
                    if (scope != "all" && tb.task != scope) continue;
                    const int ref = tb.methods.indexOf(reference), j = tb.methods.indexOf(h.method);
                    if (ref < 0 || j < 0) continue;
                    // confirmation datasets only: no development datasets, every fold of the experiment present
                    const int maxFolds = tb.folds.isEmpty() ? 0 : *std::max_element(tb.folds.begin(), tb.folds.end());
                    comparison(tb, ref, j, rd, lg, filter.excludeFromTests, maxFolds, &excluded);
                    any = true;
                }
                if (!any || rd.empty()) continue;
                const PairStats ps = pairStats(rd, lg);
                GEHypothesisTest t;
                t.config = cfg; t.scope = scope; t.id = h.id; t.reference = reference; t.method = h.method; t.meaning = h.meaning;
                t.n = ps.n; t.better = ps.better; t.ties = ps.ties; t.worse = ps.worse;
                t.relDiffMedian = ps.med; t.geoRatio = ps.geo; t.p = ps.p;
                std::sort(excluded.begin(), excluded.end());
                t.excluded = excluded;
                hs << t;
            }
            QVector<double> p;
            for (const auto& t : hs) p << t.p;
            const QVector<double> adj = geHolm(p);
            for (int i = 0; i < hs.size(); ++i) { hs[i].holmP = adj[i]; out.hypotheses << hs[i]; }
        }
    }
    return out;
}

QString geHypothesesText(const GEOverall& o) {
    QString s;
    QTextStream t(&s);
    QString lastKey;
    for (const GEHypothesisTest& h : o.hypotheses) {
        const QString key = h.config + "|" + h.scope;
        if (key != lastKey) {
            t << "\n=== Hypotheses [" << h.config << "], " << h.scope << " (" << h.n << " datasets): "
              << geMethodInfo(h.reference).label << " < method; relative difference rd = (method - ref) / mean, "
              << "Wilcoxon on rd, Holm over the hypotheses ===\n";
            if (!h.excluded.isEmpty())
                t << "  left out (development datasets or incomplete folds): " << h.excluded.join(", ") << "\n";
            lastKey = key;
        }
        t << QString("  %1 vs %2").arg(h.id, 2).arg(geMethodInfo(h.method).label, -24)
          << QString("ref better/tie/worse %1/%2/%3").arg(h.better).arg(h.ties).arg(h.worse).leftJustified(30)
          << QString("median rd %1%  ").arg(std::isnan(h.relDiffMedian) ? QString("-") : QString::number(h.relDiffMedian, 'f', 1))
          << QString("ratio %1  ").arg(std::isnan(h.geoRatio) ? QString("-") : QString::number(h.geoRatio, 'f', 3))
          << "p " << fmtP(h.p) << "  Holm " << fmtP(h.holmP)
          << (h.holmP < 0.05 ? "  *" : "  ns") << "\n";
    }
    return s;
}

QString geOverallText(const GEOverall& o) {
    QString s;
    QTextStream t(&s);
    for (const GEOverallTable& tb : o.tables) {
        const bool cls = tb.task == "classification";
        t << "\n=== Overall " << tb.task << " [" << tb.config << "]  (" << (cls ? "test error %" : "test MSE")
          << ", mean over folds; " << tb.datasets.size() << " datasets) ===\n";
        t << QString("%1").arg("method", -24) << QString("%1").arg("BNF", 5) << QString("%1").arg("average", 12)
          << QString("%1").arg("avg rank", 10) << QString("%1").arg(tb.reference.isEmpty() ? "" : "W/T/L of " + tb.reference, 22)
          << QString("%1").arg("Wilcoxon p", 12) << QString("%1").arg("median rd %", 13)
          << QString("%1").arg("rd p", 10) << QString("%1").arg("Holm p", 10) << "\n";
        for (int j = 0; j < tb.methods.size(); ++j) {
            const bool isRef = tb.methods[j] == tb.reference;
            t << QString("%1").arg(geMethodInfo(tb.methods[j]).label, -24) << QString("%1").arg(tb.bnf[j] ? "yes" : "no", 5)
              << QString("%1").arg(fmt(tb.average[j], cls), 12)
              << QString("%1").arg(std::isnan(tb.avgRank[j]) ? QString("-") : QString::number(tb.avgRank[j], 'f', 2), 10)
              << QString("%1").arg(isRef ? QString("(reference)") : QString("%1/%2/%3").arg(tb.wins[j]).arg(tb.ties[j]).arg(tb.losses[j]), 22)
              << QString("%1").arg(isRef ? QString("") : fmtP(tb.wilcoxonP[j]), 12)
              << QString("%1").arg(isRef || std::isnan(tb.relDiffMedian[j]) ? QString("") : QString::number(tb.relDiffMedian[j], 'f', 1), 13)
              << QString("%1").arg(isRef ? QString("") : fmtP(tb.wilcoxonRelP[j]), 10)
              << QString("%1").arg(isRef ? QString("") : fmtP(tb.holmP[j]), 10) << "\n";
        }
        t << "Friedman p = " << fmtP(tb.friedmanP) << ";  Nemenyi CD(0.05) = "
          << (std::isnan(tb.cd05) ? QString("-") : QString::number(tb.cd05, 'f', 2))
          << ";  datasets with every method: " << tb.complete << "\n";
    }
    t << geHypothesesText(o);
    for (const QString& w : o.warnings) t << "note: " << w << "\n";
    return s;
}

// ── XLSX ──────────────────────────────────────────────────────────────────────
namespace {
void writeTaskSheet(XlsxSheet* sh, const QVector<const GEOverallTable*>& tabs, bool withStd) {
    bool firstBlock = true;
    for (const GEOverallTable* tb : tabs) {
        const bool cls = tb->task == "classification";
        const int M = tb->methods.size();
        // group row: Without BNF / With BNF
        std::vector<XlsxCell> group{XlsxCell(QString("Config: %1").arg(tb->config)), XlsxCell()};
        bool seenNo = false, seenYes = false;
        for (int j = 0; j < M; ++j) {
            if (!tb->bnf[j] && !seenNo) { group.push_back("Without BNF"); seenNo = true; }
            else if (tb->bnf[j] && !seenYes) { group.push_back("With BNF"); seenYes = true; }
            else group.push_back(XlsxCell());
        }
        QStringList header{"Dataset", "Folds"};
        for (const QString& m : tb->methods) header << geMethodInfo(m).label;
        if (firstBlock) {
            sh->setHeader(QStringList{withStd ? (cls ? "Std of test error % over folds" : "Std of test MSE over folds")
                                              : (cls ? "Test error % (mean over folds)" : "Test MSE (mean over folds)")});
            firstBlock = false;
        } else {
            sh->addRow({});
            sh->addRow({});
        }
        sh->addRow(group);
        std::vector<XlsxCell> hr;
        for (const QString& h : header) hr.push_back(h);
        sh->addRow(hr);
        for (int i = 0; i < tb->datasets.size(); ++i) {
            std::vector<XlsxCell> row{XlsxCell(tb->datasets[i]), XlsxCell(tb->folds[i])};
            for (int j = 0; j < M; ++j) {
                const double v = withStd ? tb->sd[i][j] : tb->value[i][j];
                row.push_back(std::isnan(v) ? XlsxCell() : XlsxCell(v));
            }
            sh->addRow(row);
        }
        if (withStd) continue;
        sh->addRow({});
        auto metricRow = [&](const QString& name, const QVector<double>& v) {
            std::vector<XlsxCell> row{XlsxCell(name), XlsxCell(tb->complete)};
            for (int j = 0; j < M; ++j) row.push_back(std::isnan(v[j]) ? XlsxCell() : XlsxCell(v[j]));
            sh->addRow(row);
        };
        metricRow("AVERAGE", tb->average);
        metricRow("Average rank (Friedman)", tb->avgRank);
        if (!tb->reference.isEmpty()) {
            const QString refL = geMethodInfo(tb->reference).label;
            std::vector<XlsxCell> wtl{XlsxCell(QString("%1: wins/ties/losses").arg(refL)), XlsxCell(tb->complete)};
            std::vector<XlsxCell> wp{XlsxCell(QString("Wilcoxon p (%1 vs column)").arg(refL)), XlsxCell(tb->complete)};
            for (int j = 0; j < M; ++j) {
                const bool isRef = tb->methods[j] == tb->reference;
                wtl.push_back(isRef ? XlsxCell("reference") : XlsxCell(QString("%1/%2/%3").arg(tb->wins[j]).arg(tb->ties[j]).arg(tb->losses[j])));
                wp.push_back(isRef || std::isnan(tb->wilcoxonP[j]) ? XlsxCell() : XlsxCell(tb->wilcoxonP[j]));
            }
            sh->addRow(wtl);
            sh->addRow(wp);
            auto refRow = [&](const QString& name, const QVector<double>& v) {
                std::vector<XlsxCell> row{XlsxCell(name), XlsxCell(tb->complete)};
                for (int j = 0; j < M; ++j)
                    row.push_back(tb->methods[j] == tb->reference ? XlsxCell("reference")
                                  : std::isnan(v[j]) ? XlsxCell() : XlsxCell(v[j]));
                sh->addRow(row);
            };
            refRow(QString("Median relative difference % (column vs %1; > 0: %1 better)").arg(refL), tb->relDiffMedian);
            refRow(QString("Geometric mean ratio column / %1").arg(refL), tb->geoRatio);
            refRow(QString("Wilcoxon p on relative differences (%1 vs column)").arg(refL), tb->wilcoxonRelP);
            refRow("Holm-adjusted p (all columns vs the reference)", tb->holmP);
        }
        sh->addRow({XlsxCell("Friedman p"), XlsxCell(), std::isnan(tb->friedmanP) ? XlsxCell("-") : XlsxCell(tb->friedmanP)});
        sh->addRow({XlsxCell("Nemenyi critical difference (alpha 0.05)"), XlsxCell(),
                    std::isnan(tb->cd05) ? XlsxCell("-") : XlsxCell(tb->cd05)});
        sh->addRow({XlsxCell("Datasets with every method"), XlsxCell(tb->complete)});
    }
    if (tabs.isEmpty()) sh->setHeader({"No datasets of this kind in the results."});
}
} // namespace

void geWriteOverallXlsx(const GEOverall& o, const QString& path) {
    XlsxWriter wb;
    QVector<const GEOverallTable*> cls, reg;
    for (const GEOverallTable& t : o.tables) (t.task == "classification" ? cls : reg) << &t;
    writeTaskSheet(wb.addSheet("Classification"), cls, false);
    writeTaskSheet(wb.addSheet("Regression"), reg, false);
    auto* hyp = wb.addSheet("Hypotheses");
    hyp->setHeader({"Config", "Scope", "Hypothesis", "Reference", "Compared with", "Meaning", "Datasets",
                    "Reference better", "Ties", "Reference worse", "Median relative difference %",
                    "Geometric mean ratio (method / reference)", "Wilcoxon p", "Holm-adjusted p", "Significant (Holm, 0.05)",
                    "Left out (development datasets, incomplete folds)"});
    for (const GEHypothesisTest& h : o.hypotheses)
        hyp->addRow({h.config, h.scope, h.id, geMethodInfo(h.reference).label, geMethodInfo(h.method).label, h.meaning, h.n,
                     h.better, h.ties, h.worse,
                     std::isnan(h.relDiffMedian) ? XlsxCell() : XlsxCell(h.relDiffMedian),
                     std::isnan(h.geoRatio) ? XlsxCell() : XlsxCell(h.geoRatio),
                     std::isnan(h.p) ? XlsxCell() : XlsxCell(h.p),
                     std::isnan(h.holmP) ? XlsxCell() : XlsxCell(h.holmP),
                     XlsxCell(h.holmP < 0.05 ? "yes" : "no"), XlsxCell(h.excluded.join(", "))});
    hyp->addRow({});
    hyp->addRow({XlsxCell("Relative difference rd = (method - reference) / mean(|method|, |reference|) per dataset; "
                          "rd > 0: the reference is better. Scale-free, so regression datasets with large MSE do not dominate. "
                          "Holm adjusts over the hypotheses of the same configuration and scope.")});
    writeTaskSheet(wb.addSheet("Classification (std)"), cls, true);
    writeTaskSheet(wb.addSheet("Regression (std)"), reg, true);

    auto* meth = wb.addSheet("Methods");
    meth->setHeader({"Method id", "Label", "Uses BNF", "Description"});
    std::set<QString> used;
    for (const GEOverallTable& t : o.tables) for (const QString& m : t.methods) used.insert(m);
    for (const auto& m : geMethodCatalog())
        if (used.count(m.id)) meth->addRow({m.id, m.label, m.bnf ? "Yes" : "No", m.description});
    for (const QString& m : used) {
        bool known = false; const GEMethodInfo mi = geMethodInfo(m, &known);
        bool inCat = false; for (const auto& c : geMethodCatalog()) inCat = inCat || c.id == m;
        if (!inCat) meth->addRow({mi.id, mi.label, mi.bnf ? "Yes" : "No", mi.description});
    }

    auto* cost = wb.addSheet("Cost");
    cost->setHeader({"Task", "Config", "Method", "Uses BNF", "Mean seconds per run (training of this row)", "Mean parameters"});
    for (const GEOverallTable& t : o.tables)
        for (int j = 0; j < t.methods.size(); ++j)
            cost->addRow({t.task, t.config, geMethodInfo(t.methods[j]).label, t.bnf[j] ? "Yes" : "No",
                          t.seconds[j], t.params[j]});
    if (!o.warnings.isEmpty()) {
        auto* notes = wb.addSheet("Notes");
        notes->setHeader({"Note"});
        for (const QString& w : o.warnings) notes->addRow({w});
    }
    wb.save(path);
}

} // namespace NeuralStudio
