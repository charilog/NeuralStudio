// GEBatch.cpp — batch experiment engine of Mode 6 (see GEBatch.h).
#include "GEBatch.h"
#include <QRegularExpression>
#include "core/ge/GEStudioWorker.h"
#include "core/ge/GEExperiment.h"
#include "core/ge/GEReport.h"
#include "core/ge/Baselines.h"
#include "core/ge/NetSpec.h"
#include "core/stats/Statistics.h"
#include "core/export/XlsxWriter.h"
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QMutexLocker>
#include <QSet>
#include <QTextStream>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <vector>

namespace NeuralStudio {

namespace {

struct Row {
    QString ds; int fold = 0; QString cfg; int seed = 0; QString method, task;
    double trainErr = 0, testErr = 0; int params = 0; qint64 evals = 0; double secs = 0;
    qint64 searchEvals = 0; double searchSecs = 0; int gensRun = 0, uniqueNets = 0, cacheHits = 0, testN = 0;
    bool cls = true;
};

QString num(double v) { return QString::number(v, 'g', 10); }

const char* kHeader = "dataset,fold,config,seed,method,task,train_err,test_err,params,evals,seconds,"
                      "search_evals,search_seconds,gens_run,unique_nets,cache_hits,test_n";

QString toCsv(const Row& r) {
    return QStringList{r.ds, QString::number(r.fold), r.cfg, QString::number(r.seed), r.method,
        r.cls ? "cls" : "reg", num(r.trainErr), num(r.testErr), QString::number(r.params),
        QString::number(r.evals), num(r.secs), QString::number(r.searchEvals), num(r.searchSecs),
        QString::number(r.gensRun), QString::number(r.uniqueNets), QString::number(r.cacheHits),
        QString::number(r.testN)}.join(',');
}

bool fromCsv(const QString& line, Row& r) {
    const QStringList f = line.split(',');
    if (f.size() < 17) return false;
    r.ds = f[0]; r.fold = f[1].toInt(); r.cfg = f[2]; r.seed = f[3].toInt(); r.method = f[4];
    r.cls = f[5] == "cls"; r.task = f[5];
    r.trainErr = f[6].toDouble(); r.testErr = f[7].toDouble(); r.params = f[8].toInt();
    r.evals = f[9].toLongLong(); r.secs = f[10].toDouble(); r.searchEvals = f[11].toLongLong();
    r.searchSecs = f[12].toDouble(); r.gensRun = f[13].toInt(); r.uniqueNets = f[14].toInt();
    r.cacheHits = f[15].toInt(); r.testN = f[16].toInt();
    return true;
}

QList<int> parseFolds(const QString& s) {
    QList<int> out;
    for (const QString& part : s.split(',', Qt::SkipEmptyParts)) {
        const QStringList ab = part.split('-');
        if (ab.size() == 2) for (int i = ab[0].toInt(); i <= ab[1].toInt(); ++i) out << i;
        else out << part.toInt();
    }
    return out;
}

double meanOf(const std::vector<double>& v) {
    if (v.empty()) return 0;
    double s = 0; for (double x : v) s += x; return s / v.size();
}
double stdOf(const std::vector<double>& v) {
    if (v.size() < 2) return 0;
    const double m = meanOf(v); double s = 0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / (v.size() - 1));
}

bool configFromName(const QString& name, GEStudioConfig& c) {
    if (name == "baselines") { c.mode = "baselines"; return true; }
    const QStringList p = name.split('-');
    if (p.size() != 2) return false;
    if (p[0] == "unified") c.grammarText.clear();
    else if (p[0] == "unifiedpe") c.grammarText = QString::fromStdString(ge::presetGrammar(ge::GrammarPreset::UnifiedEmbedding));
    else if (p[0] == "archbounds") c.grammarText = QString::fromStdString(ge::presetGrammar(ge::GrammarPreset::ArchAndBounds));
    else if (p[0] == "archonly")   c.grammarText = QString::fromStdString(ge::presetGrammar(ge::GrammarPreset::ArchOnly));
    else return false;
    if (p[1] == "ge")           { c.genotype = "ge";  c.crossover = "onepoint"; }
    else if (p[1] == "sage")    { c.genotype = "ge";  c.crossover = "onepoint"; c.adaptiveSearch = true; }
    else if (p[1] == "gelocal") { c.genotype = "ge";  c.crossover = "local"; }
    else if (p[1] == "pge")     { c.genotype = "pge"; }
    else return false;
    if (p[0] != "unified" && p[0] != "unifiedpe") c.maxFeatures = 0;
    return true;
}

// ── Summary ───────────────────────────────────────────────────────────────────
struct Unit { QString ds; int fold; int seed; };

QString stars(const WilcoxonResult& w) { return w.valid ? QString::fromStdString(w.stars()) : QString("-"); }

void summarize(const std::vector<Row>& rows, const QString& prefix, QTextStream& out) {
    // key: ds|cfg|method -> values by unit "fold:seed"
    std::map<std::string, std::map<std::string, Row>> cell;      // [ds|cfg|method][fold:seed]
    std::set<std::string> dsSet, cfgSet, methSet;
    for (const Row& r : rows) {
        const std::string unit = (QString::number(r.fold) + ":" + QString::number(r.seed)).toStdString();
        cell[(r.ds + "|" + r.cfg + "|" + r.method).toStdString()][unit] = r;
        dsSet.insert(r.ds.toStdString()); cfgSet.insert(r.cfg.toStdString()); methSet.insert(r.method.toStdString());
    }
    auto get = [&](const std::string& ds, const std::string& cfg, const std::string& m)
        -> const std::map<std::string, Row>* {
        auto it = cell.find(ds + "|" + cfg + "|" + m);
        return it == cell.end() ? nullptr : &it->second;
    };

    XlsxWriter wb;
    auto* sum = wb.addSheet("Summary");
    sum->setHeader({"Dataset", "Config", "Method", "Units", "Test err mean", "Test err std", "Pooled test err",
                    "Params (mean)", "Evals (mean)", "Seconds (mean)", "Test err median"});
    QString csv = "dataset,config,method,units,test_mean,test_std,pooled_test,params_mean,evals_mean,seconds_mean,test_median\n";
    std::vector<std::string> methodOrder;
    for (const char* m : {"standard", "standard_T", "standard_eq", "grid_T", "grid_eq"}) if (methSet.count(m)) methodOrder.push_back(m);
    // GE ranks ge1, ge2, ... (not "genetic", which also starts with "ge")
    auto isGeRank = [](const std::string& m) {
        return m.size() > 2 && m.compare(0, 2, "ge") == 0 &&
               std::all_of(m.begin() + 2, m.end(), [](char ch) { return ch >= '0' && ch <= '9'; });
    };
    for (const auto& m : methSet) if (isGeRank(m)) methodOrder.push_back(m);
    for (const char* m : {"ge1_nobounds", "ge1_eq", "ge1_nobounds_eq", "ge1_T", "ge1_T_nobounds", "ge1_T_nofeat"})
        if (methSet.count(m)) methodOrder.push_back(m);
    // self-adaptive rows (they were missing from the summary and the paired tests)
    for (const char* m : {"standard_T_ad", "std_ens_T", "ge1_T_ad", "ens_T", "ens_T_nofeat", "ens_T_nobounds",
                          "ens_T_div", "ens_T_cv", "std_ens_T_cv",
                          "sage", "sage_noemb", "sage_nofeat", "sage_nobounds", "sage_one", "sage_std"})
        if (methSet.count(m)) methodOrder.push_back(m);
    // reference models (configuration "baselines")
    for (const QString& m : geBaselineIds())
        if (methSet.count(m.toStdString())) methodOrder.push_back(m.toStdString());
    auto medianOf = [](std::vector<double> v) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        const size_t n = v.size();
        return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    };

    out << "\n=== Test error per dataset / config / method (mean +- std over folds x seeds) ===\n";
    for (const auto& ds : dsSet) for (const auto& cfg : cfgSet) {
        bool printed = false;
        for (const auto& m : methodOrder) {
            const auto* c = get(ds, cfg, m);
            if (!c) continue;
            std::vector<double> te, pa, ev, se; double wsum = 0, nsum = 0; bool cls = true;
            for (const auto& kv : *c) {
                te.push_back(kv.second.testErr); pa.push_back(kv.second.params);
                ev.push_back(kv.second.evals); se.push_back(kv.second.secs);
                wsum += kv.second.testErr * kv.second.testN; nsum += kv.second.testN; cls = kv.second.cls;
            }
            const double pooled = nsum > 0 ? wsum / nsum : meanOf(te);
            if (!printed) { out << QString("\n%1  [%2]\n").arg(QString::fromStdString(ds), QString::fromStdString(cfg)); printed = true; }
            out << QString("  %1  n=%2  test %3 +- %4  median %5  params %6  evals %7\n")
                    .arg(QString::fromStdString(m), -15).arg(te.size(), 3).arg(meanOf(te), 9, 'g', 5)
                    .arg(stdOf(te), 8, 'g', 4).arg(medianOf(te), 9, 'g', 5).arg(meanOf(pa), 7, 'f', 0).arg(meanOf(ev), 9, 'f', 0);
            (void)cls;
            sum->addRow({QString::fromStdString(ds), QString::fromStdString(cfg), QString::fromStdString(m),
                         static_cast<int>(te.size()), meanOf(te), stdOf(te), pooled, meanOf(pa), meanOf(ev), meanOf(se), medianOf(te)});
            csv += QStringList{QString::fromStdString(ds), QString::fromStdString(cfg), QString::fromStdString(m),
                QString::number(te.size()), num(meanOf(te)), num(stdOf(te)), num(pooled), num(meanOf(pa)),
                num(meanOf(ev)), num(meanOf(se)), num(medianOf(te))}.join(',') + "\n";
        }
    }

    // search cost per dataset / config (from the ge1 rows)
    auto* cost = wb.addSheet("Search cost");
    cost->setHeader({"Dataset", "Config", "Search evals (mean)", "Search seconds (mean)", "Generations run (mean)",
                     "Distinct nets (mean)", "Cache hits (mean)"});
    out << "\n=== Search cost (GE) ===\n";
    for (const auto& ds : dsSet) for (const auto& cfg : cfgSet) {
        const auto* c = get(ds, cfg, "ge1");
        if (!c) continue;
        std::vector<double> se, ss, gr, un, ch;
        for (const auto& kv : *c) {
            se.push_back(kv.second.searchEvals); ss.push_back(kv.second.searchSecs);
            gr.push_back(kv.second.gensRun); un.push_back(kv.second.uniqueNets); ch.push_back(kv.second.cacheHits);
        }
        out << QString("  %1 [%2]  evals %3  seconds %4  generations %5  distinct nets %6  cache hits %7\n")
                .arg(QString::fromStdString(ds), QString::fromStdString(cfg)).arg(meanOf(se), 0, 'f', 0)
                .arg(meanOf(ss), 0, 'f', 1).arg(meanOf(gr), 0, 'f', 1).arg(meanOf(un), 0, 'f', 0).arg(meanOf(ch), 0, 'f', 0);
        cost->addRow({QString::fromStdString(ds), QString::fromStdString(cfg), meanOf(se), meanOf(ss), meanOf(gr), meanOf(un), meanOf(ch)});
    }

    // ── paired tests ─────────────────────────────────────────────────────────
    auto* tests = wb.addSheet("Paired tests");
    tests->setHeader({"Scope", "A", "B", "Units", "Mean A", "Mean B", "A better", "Tie", "B better", "p (Wilcoxon)", "Sig."});
    QString tcsv = "scope,A,B,units,mean_A,mean_B,A_better,tie,B_better,p,sig\n";

    struct Series { std::string label; std::string ds, cfg, m; };
    auto compare = [&](const QString& scope, const std::string& ds, const Series& A, const Series& B) {
        const auto* ca = get(A.ds, A.cfg, A.m); const auto* cb = get(B.ds, B.cfg, B.m);
        if (!ca || !cb) return;
        // units = folds: the seeds of one fold share the same train/test split, so they are
        // averaged first (treating them as independent would overstate significance).
        std::map<int, std::pair<double, double>> sumByFold; std::map<int, int> cnt;
        for (const auto& kv : *ca) {
            auto it = cb->find(kv.first);
            if (it == cb->end()) continue;
            auto& pr = sumByFold[kv.second.fold];
            pr.first += kv.second.testErr; pr.second += it->second.testErr; ++cnt[kv.second.fold];
        }
        std::vector<double> a, b;
        for (const auto& kv : sumByFold) { a.push_back(kv.second.first / cnt[kv.first]); b.push_back(kv.second.second / cnt[kv.first]); }
        if (a.empty()) return;
        int aw = 0, tie = 0, bw = 0;
        for (size_t i = 0; i < a.size(); ++i) { if (a[i] < b[i] - 1e-12) ++aw; else if (a[i] > b[i] + 1e-12) ++bw; else ++tie; }
        const WilcoxonResult w = Statistics::wilcoxon(a, b);
        const QString pv = w.valid ? QString::number(w.pValue, 'g', 4) : "-";
        out << QString("  %1  %2 vs %3  n=%4  mean %5 vs %6  better %7/tie %8/worse %9  p=%10 %11\n")
                .arg(scope, -22).arg(QString::fromStdString(A.label)).arg(QString::fromStdString(B.label)).arg(a.size(), 3)
                .arg(meanOf(a), 0, 'g', 5).arg(meanOf(b), 0, 'g', 5).arg(aw).arg(tie).arg(bw).arg(pv).arg(stars(w));
        tests->addRow({scope, QString::fromStdString(A.label), QString::fromStdString(B.label), static_cast<int>(a.size()),
                       meanOf(a), meanOf(b), aw, tie, bw, pv, stars(w)});
        tcsv += QStringList{scope, QString::fromStdString(A.label), QString::fromStdString(B.label), QString::number(a.size()),
            num(meanOf(a)), num(meanOf(b)), QString::number(aw), QString::number(tie), QString::number(bw), pv, stars(w)}.join(',') + "\n";
        (void)ds;
    };

    out << "\n=== Paired tests per dataset (units = folds, seeds averaged; lower test error is better) ===\n";
    for (const auto& ds : dsSet) for (const auto& cfg : cfgSet) {
        const QString scope = QString::fromStdString(ds + " [" + cfg + "]");
        if (get(ds, cfg, "sage")) {   // SAGE vs its ablations, per dataset
            for (const char* b : {"sage_std", "sage_noemb", "sage_nofeat", "sage_nobounds", "sage_one"})
                compare(scope, ds, {"sage", ds, cfg, "sage"}, {b, ds, cfg, b});
        }
        if (get(ds, cfg, "ens_T")) {   // hypotheses H1-H4, H6 of the self-adaptive method, per dataset
            for (const char* b : {"std_ens_T", "ge1_T", "ens_T_nobounds", "ens_T_nofeat", "ge1_T_ad"})
                compare(scope, ds, {"ens_T", ds, cfg, "ens_T"}, {b, ds, cfg, b});
        }
        if (get(ds, cfg, "ge1_T")) {
            compare(scope, ds, {"ge1_T", ds, cfg, "ge1_T"}, {"standard_T", ds, cfg, "standard_T"});
            compare(scope, ds, {"ge1_T", ds, cfg, "ge1_T"}, {"grid_T", ds, cfg, "grid_T"});
            compare(scope, ds, {"ge1_T", ds, cfg, "ge1_T"}, {"ge1_T_nobounds", ds, cfg, "ge1_T_nobounds"});
            compare(scope, ds, {"ge1_T", ds, cfg, "ge1_T"}, {"ge1_T_nofeat", ds, cfg, "ge1_T_nofeat"});
            compare(scope, ds, {"ge1", ds, cfg, "ge1"}, {"standard", ds, cfg, "standard"});
            continue;
        }
        compare(scope, ds, {"ge1", ds, cfg, "ge1"}, {"standard_eq", ds, cfg, "standard_eq"});
        compare(scope, ds, {"ge1", ds, cfg, "ge1"}, {"grid_eq", ds, cfg, "grid_eq"});
        compare(scope, ds, {"ge1", ds, cfg, "ge1"}, {"standard", ds, cfg, "standard"});
        compare(scope, ds, {"ge1", ds, cfg, "ge1"}, {"ge1_nobounds", ds, cfg, "ge1_nobounds"});
        compare(scope, ds, {"ge1_eq", ds, cfg, "ge1_eq"}, {"standard_eq", ds, cfg, "standard_eq"});
        compare(scope, ds, {"ge1_eq", ds, cfg, "ge1_eq"}, {"ge1_nobounds_eq", ds, cfg, "ge1_nobounds_eq"});
    }
    if (cfgSet.size() > 1) {
        out << "\n=== GE #1 across configurations (per dataset) ===\n";
        for (const auto& ds : dsSet)
            for (auto i = cfgSet.begin(); i != cfgSet.end(); ++i)
                for (auto j = std::next(i); j != cfgSet.end(); ++j)
                    compare(QString::fromStdString(ds), ds, {"ge1@" + *i, ds, *i, get(ds, *i, "ge1_T") ? "ge1_T" : "ge1"},
                            {"ge1@" + *j, ds, *j, get(ds, *j, "ge1_T") ? "ge1_T" : "ge1"});
    }

    // across datasets: relative error = dataset mean / dataset mean of standard_eq
    auto datasetMean = [&](const std::string& ds, const std::string& cfg, const std::string& m, double& v) {
        const auto* c = get(ds, cfg, m); if (!c || c->empty()) return false;
        std::vector<double> x; for (const auto& kv : *c) x.push_back(kv.second.testErr);
        v = meanOf(x); return true;
    };
    if (dsSet.size() >= 2) {
        out << "\n=== Across datasets (relative test error = mean / mean of Standard equal budget; < 1 is better) ===\n";
        for (const auto& cfg : cfgSet) {
            bool tMode = true;
            for (const auto& ds : dsSet) if (!get(ds, cfg, "ge1_T")) tMode = false;
            const int NM = 5; std::vector<double> rel[NM];
            const char* msOld[NM] = {"standard", "standard_eq", "grid_eq", "ge1", "ge1_nobounds"};
            const char* msT[NM]   = {"ge1_T_nofeat", "standard_T", "grid_T", "ge1_T", "ge1_T_nobounds"};
            const char* const* ms = tMode ? msT : msOld;   // index 1 = reference, 2 = grid, 3 = GE
            std::vector<std::vector<double>> frd(NM); int used = 0;
            for (const auto& ds : dsSet) {
                double v[NM]; bool ok = true;
                for (int k = 0; k < NM; ++k) ok = ok && datasetMean(ds, cfg, ms[k], v[k]);
                if (!ok || v[1] <= 0) continue;
                for (int k = 0; k < NM; ++k) { rel[k].push_back(v[k] / v[1]); frd[k].push_back(v[k]); }
                ++used;
            }
            {   // equal-budget GE rows (present only in newer raw files)
                std::vector<double> a, b, c; int u2 = 0;
                for (const auto& ds : dsSet) {
                    double ve, vs, vn;
                    if (datasetMean(ds, cfg, "ge1_eq", ve) && datasetMean(ds, cfg, "standard_eq", vs)
                        && datasetMean(ds, cfg, "ge1_nobounds_eq", vn) && vs > 0) {
                        a.push_back(ve / vs); b.push_back(1.0); c.push_back(vn / vs); ++u2;
                    }
                }
                if (u2 >= 2) {
                    const WilcoxonResult w1 = Statistics::wilcoxon(a, b);
                    const WilcoxonResult w2 = Statistics::wilcoxon(a, c);
                    out << QString("  [%1] equal-budget GE #1 (relative to Standard equal budget, %2 datasets): mean %3\n")
                               .arg(QString::fromStdString(cfg)).arg(u2).arg(meanOf(a), 0, 'f', 3);
                    out << QString("    GE #1 eq vs Standard eq: p=%1 %2;  GE #1 eq vs GE #1 eq without bounds: p=%3 %4\n")
                               .arg(w1.valid ? QString::number(w1.pValue, 'g', 4) : "-").arg(stars(w1))
                               .arg(w2.valid ? QString::number(w2.pValue, 'g', 4) : "-").arg(stars(w2));
                }
            }
            if (used < 2) continue;
            out << QString("  [%1] datasets used: %2\n").arg(QString::fromStdString(cfg)).arg(used);
            for (int k = 0; k < NM; ++k) out << QString("    %1  mean relative error %2\n").arg(ms[k], -14).arg(meanOf(rel[k]), 0, 'f', 3);
            const WilcoxonResult w = Statistics::wilcoxon(rel[3], rel[1]);
            const WilcoxonResult wg = Statistics::wilcoxon(rel[3], rel[2]);
            const WilcoxonResult wb2 = Statistics::wilcoxon(rel[3], rel[4]);
            const WilcoxonResult wf2 = Statistics::wilcoxon(rel[3], rel[0]);
            if (tMode) out << QString("    GE #1 vs without bounds: p=%1 %2;  GE #1 vs without features: p=%3 %4\n")
                                  .arg(wb2.valid ? QString::number(wb2.pValue, 'g', 4) : "-").arg(stars(wb2))
                                  .arg(wf2.valid ? QString::number(wf2.pValue, 'g', 4) : "-").arg(stars(wf2));
            out << QString("    GE #1 vs Grid equal budget: p=%1 %2\n").arg(wg.valid ? QString::number(wg.pValue, 'g', 4) : "-").arg(stars(wg));
            out << QString("    GE #1 vs Standard equal budget: p=%1 %2\n").arg(w.valid ? QString::number(w.pValue, 'g', 4) : "-").arg(stars(w));
            tests->addRow({QString("ALL datasets [%1]").arg(QString::fromStdString(cfg)), "ge1", "standard_eq", used,
                           meanOf(rel[3]), meanOf(rel[1]), "", "", "", w.valid ? QString::number(w.pValue, 'g', 4) : "-", stars(w)});
            if (used >= 3) {
                std::vector<std::string> names(ms, ms + NM);
                const FriedmanResult fr = Statistics::friedman(names, frd);
                if (fr.valid) {
                    out << QString("    Friedman p=%1; average ranks (lower better):").arg(fr.pValue, 0, 'g', 4);
                    for (int k = 0; k < NM; ++k) out << QString("  %1=%2").arg(ms[k]).arg(fr.avgRanks[k], 0, 'f', 2);
                    out << "\n";
                }
            }
        }
    }

    auto* raw = wb.addSheet("Raw");
    raw->setHeader(QString(kHeader).split(','));
    for (const Row& r : rows)
        raw->addRow({r.ds, r.fold, r.cfg, r.seed, r.method, r.cls ? "cls" : "reg", r.trainErr, r.testErr, r.params,
                     r.evals, r.secs, r.searchEvals, r.searchSecs, r.gensRun, r.uniqueNets, r.cacheHits, r.testN});

    QFile f1(prefix + "_summary.csv");
    if (f1.open(QIODevice::WriteOnly | QIODevice::Text)) { QTextStream t(&f1); t << csv; }
    QFile f2(prefix + "_paired_tests.csv");
    if (f2.open(QIODevice::WriteOnly | QIODevice::Text)) { QTextStream t(&f2); t << tcsv; }
    try { wb.save(prefix + "_summary.xlsx"); }
    catch (const std::exception& ex) { out << "xlsx export failed: " << ex.what() << "\n"; }
    out << "\nWritten: " << prefix << "_raw.csv, _summary.csv, _paired_tests.csv, _summary.xlsx\n";
}

} // namespace

// ── Command line ──────────────────────────────────────────────────────────────
QStringList splitBatchCommandLine(const QString& cmd) {
    QStringList out;
    QString cur;
    bool inTok = false;
    QChar quote;
    for (int i = 0; i < cmd.size(); ++i) {
        const QChar ch = cmd[i];
        if (!quote.isNull()) {
            if (ch == quote) quote = QChar();
            else cur += ch;
            continue;
        }
        if (ch == '"' || ch == '\'') { quote = ch; inTok = true; continue; }
        if (ch.isSpace()) {
            if (inTok) { out << cur; cur.clear(); inTok = false; }
            continue;
        }
        // PowerShell / cmd / bash line continuations pasted from a multi-line command
        if ((ch == '`' || ch == '^' || ch == '\\') && i + 1 < cmd.size() && (cmd[i + 1] == '\n' || cmd[i + 1] == '\r')) {
            continue;
        }
        cur += ch; inTok = true;
    }
    if (inTok) out << cur;
    return out;
}

namespace {
QString resolvePath(const QString& p, const QString& baseDir) {
    QString s = p;
    s.replace('\\', '/');                       // Windows-style paths work on Linux too
    if (s.startsWith("~/")) s = QDir::homePath() + s.mid(1);
    if (QDir::isAbsolutePath(s)) return QDir::cleanPath(s);
    return QDir::cleanPath(QDir(baseDir.isEmpty() ? QDir::currentPath() : baseDir).absoluteFilePath(s));
}
const QStringList kFlags = {"--summarize-only", "--no-resume", "--help", "-h"};
const QStringList kKeys = {"--dir", "--datasets", "--out", "--folds", "--configs", "--seeds", "--gens", "--pop",
    "--inner-pop", "--inner-iters", "--final-pop", "--final-iters", "--topk", "--patience", "--parsimony",
    "--metric", "--optimizer", "--val-split", "--bound", "--baseline-hidden", "--baseline-act",
    "--total-budget", "--search-frac", "--polish-frac", "--bound-mode", "--fan-k", "--lamarck", "--finalists",
    "--inner-polish", "--inner-polish-evals", "--methods", "--experiment", "--dev-datasets",
    "--se-rule", "--ens-pool", "--ens-max"};

bool applyConfig(const GEBatchOptions& o, const QString& cfgName, GEStudioConfig& c) {
    if (!configFromName(cfgName, c)) return false;
    const auto& a = o.args;
    if (a.contains("--gens")) c.gens = a["--gens"].toInt();
    if (a.contains("--pop")) c.pop = a["--pop"].toInt();
    if (a.contains("--inner-pop")) c.innerPop = a["--inner-pop"].toInt();
    if (a.contains("--inner-iters")) c.innerIters = a["--inner-iters"].toInt();
    if (a.contains("--final-pop")) c.finalPop = a["--final-pop"].toInt();
    if (a.contains("--final-iters")) c.finalIters = a["--final-iters"].toInt();
    c.topK = a.contains("--topk") ? a["--topk"].toInt() : 1;
    if (a.contains("--total-budget")) c.totalBudget = a["--total-budget"].toLongLong();
    if (a.contains("--search-frac")) c.searchFrac = a["--search-frac"].toDouble();
    if (a.contains("--polish-frac")) c.polishFrac = a["--polish-frac"].toDouble();
    if (a.contains("--bound-mode")) c.boundMode = a["--bound-mode"];
    if (a.contains("--fan-k")) c.fanK = a["--fan-k"].toDouble();
    if (a.contains("--lamarck")) c.lamarck = a["--lamarck"].toInt();
    if (a.contains("--finalists")) c.finalists = a["--finalists"].toInt();
    if (a.contains("--inner-polish")) c.innerPolish = a["--inner-polish"].toInt() != 0;
    if (a.contains("--inner-polish-evals")) c.innerPolishEvals = a["--inner-polish-evals"].toInt();
    if (a.contains("--se-rule")) c.seRule = a["--se-rule"].toDouble();
    if (a.contains("--ens-pool")) c.ensPool = a["--ens-pool"].toInt();
    if (a.contains("--ens-max")) c.ensMax = a["--ens-max"].toInt();
    if (a.contains("--methods")) {
        c.methods = a["--methods"].split(',', Qt::SkipEmptyParts);
        for (QString& m : c.methods) m = m.trimmed();
    }
    if (a.contains("--patience")) c.patience = a["--patience"].toInt();
    if (a.contains("--parsimony")) c.parsimony = a["--parsimony"].toDouble();
    if (a.contains("--metric")) c.fitnessMetric = a["--metric"];
    if (a.contains("--optimizer")) c.innerOpt = a["--optimizer"];
    if (a.contains("--val-split")) c.valSplit = a["--val-split"].toDouble();
    if (a.contains("--bound")) c.baseBound = a["--bound"].toDouble();
    if (a.contains("--baseline-hidden")) c.baselineHidden = a["--baseline-hidden"];
    if (a.contains("--baseline-act")) c.baselineAct = a["--baseline-act"];
    return true;
}
} // namespace

QString batchUsage() {
    return
        "usage: ge_batch --dir <folder> --datasets a,b,c --out <prefix> [--folds 1-10]\n"
        "                [--configs unified-ge,archbounds-ge,archonly-ge,unified-pge,unified-gelocal,unified-sage,unifiedpe-sage]\n"
        "                (<preset>-sage: GE with the noise-aware search stop; with --methods the self-adaptive rows\n"
        "                 standard_T_ad, std_ens_T, ge1_T_ad, ens_T, ens_T_nofeat, ens_T_nobounds)\n"
        "                [--seeds 1] [--gens 40] [--pop 40] [--inner-pop 30] [--inner-iters 60]\n"
        "                [--final-pop 80] [--final-iters 200] [--topk 1] [--patience 15] [--parsimony 0]\n"
        "                [--metric loss|error] [--optimizer de] [--val-split 0.2] [--bound 10]\n"
        "                [--baseline-hidden 16] [--baseline-act relu] [--no-resume] [--summarize-only]\n"
        "                [--total-budget T] [--search-frac 0.5] [--polish-frac 0.5] [--bound-mode fan|abs] [--fan-k 3]\n"
        "                [--lamarck 3] [--finalists 5] [--inner-polish 1] [--inner-polish-evals 300]\n"
        "                [--methods standard_T,grid_T,ge1_T,...]   (final rows to compute; default all)\n"
        "                [--dev-datasets airfoil,concrete]   (in the tables, left out of the hypothesis tests)\n"
        "                [--se-rule 1] [--ens-pool 12] [--ens-max 8]   (self-adaptive rows, configurations <preset>-sage)\n"
        "       ge_batch --experiment <file.json> [--summarize-only] [--no-resume] [other options override the file]\n"
        "Files are read as <dir>/<dataset>_<fold>.train with the .test companion.\n"
        "Paths may use \\ or /; relative paths are resolved against the working folder.\n"
        "Re-running the same command resumes: finished runs in <prefix>_raw.csv are skipped.\n";
}

bool parseBatchArgs(const QStringList& argsIn, const QString& baseDir, GEBatchOptions& o, QString& error) {
    o = GEBatchOptions();
    QStringList args = argsIn;
    // drop the program name (and anything else) before the first option
    while (!args.isEmpty() && !args.first().startsWith("-")) args.removeFirst();
    for (int i = 0; i < args.size(); ++i) {
        const QString k = args[i];
        if (kFlags.contains(k)) { o.args[k == "-h" ? "--help" : k] = "1"; continue; }
        if (!k.startsWith("--")) { error = QString("Unexpected argument: %1").arg(k); return false; }
        if (!kKeys.contains(k)) { error = QString("Unknown option: %1").arg(k); return false; }
        if (i + 1 >= args.size() || (args[i + 1].startsWith("--"))) {
            error = QString("Option %1 needs a value.").arg(k); return false;
        }
        o.args[k] = args[++i];
    }
    o.help = o.args.contains("--help");
    if (o.help) return true;
    if (o.args.contains("--experiment")) {
        const QString ep = resolvePath(o.args["--experiment"], baseDir);
        GEExperiment ex;
        QString err2; QStringList warns;
        if (!geLoadExperiment(ep, ex, err2, warns)) { error = "Experiment file: " + err2; return false; }
        QString err3;
        // Relative paths inside the file are resolved against the working folder, like the command.
        // The file gives the defaults, explicit options on the command line override them
        QStringList fileArgs = splitBatchCommandLine(geExperimentToCommand(ex));
        QMap<QString, QString> merged;
        for (int i = 0; i + 1 < fileArgs.size(); i += 2) merged[fileArgs[i]] = fileArgs[i + 1];
        for (auto it = o.args.begin(); it != o.args.end(); ++it) if (it.key() != "--experiment") merged[it.key()] = it.value();
        QStringList all;
        for (auto it = merged.begin(); it != merged.end(); ++it) {
            all << it.key();
            if (!kFlags.contains(it.key())) all << it.value();
        }
        if (!parseBatchArgs(all, baseDir, o, err3)) { error = err3; return false; }
        o.experimentPath = ep;
        o.warnings = warns;
        return true;
    }
    for (const char* req : {"--dir", "--datasets", "--out"})
        if (!o.args.contains(req)) { error = QString("Missing required option %1.").arg(req); return false; }

    o.dir = resolvePath(o.args["--dir"], baseDir);
    o.prefix = resolvePath(o.args["--out"], baseDir);
    o.datasets = o.args["--datasets"].split(',', Qt::SkipEmptyParts);
    o.devDatasets = o.args.value("--dev-datasets").split(',', Qt::SkipEmptyParts);
    for (QString& d : o.devDatasets) d = d.trimmed();
    for (QString& d : o.datasets) d = d.trimmed();
    o.folds = parseFolds(o.args.value("--folds", "1-10"));
    o.configs = o.args.value("--configs", "unified-ge").split(',', Qt::SkipEmptyParts);
    for (QString& c : o.configs) c = c.trimmed();
    bool ok = true;
    o.seeds = o.args.value("--seeds", "1").toInt(&ok);
    if (!ok || o.seeds < 1) { error = "--seeds must be a positive integer."; return false; }
    if (o.datasets.isEmpty()) { error = "--datasets is empty."; return false; }
    if (o.folds.isEmpty()) { error = "--folds is empty or invalid (e.g. 1-10 or 1,3,5)."; return false; }
    for (int f : o.folds) if (f <= 0) { error = "--folds must contain positive numbers (e.g. 1-10)."; return false; }
    for (const QString& c : o.configs) {
        GEStudioConfig tmp;
        if (!configFromName(c, tmp)) {
            error = QString("Unknown configuration '%1' (use <unified|archbounds|archonly>-<ge|gelocal|pge> or baselines).").arg(c);
            return false;
        }
    }
    static const QStringList intKeys = {"--gens", "--pop", "--inner-pop", "--inner-iters", "--final-pop",
        "--final-iters", "--topk", "--patience", "--lamarck", "--finalists", "--inner-polish", "--inner-polish-evals",
        "--total-budget"};
    for (const QString& k : intKeys)
        if (o.args.contains(k)) {
            bool good = false; o.args[k].toLongLong(&good);
            if (!good) { error = QString("%1 expects an integer, got '%2'.").arg(k, o.args[k]); return false; }
        }
    static const QStringList realKeys = {"--parsimony", "--val-split", "--bound", "--search-frac", "--polish-frac", "--fan-k"};
    for (const QString& k : realKeys)
        if (o.args.contains(k)) {
            bool good = false; o.args[k].toDouble(&good);
            if (!good) { error = QString("%1 expects a number, got '%2'.").arg(k, o.args[k]); return false; }
        }
    if (o.args.contains("--methods")) {
        for (const QString& m0 : o.args["--methods"].split(',', Qt::SkipEmptyParts)) {
            bool known = false; geMethodInfo(m0.trimmed(), &known);
            if (!known) { error = QString("Unknown method '%1' in --methods.").arg(m0.trimmed()); return false; }
        }
    }
    if (o.args.contains("--metric") && o.args["--metric"] != "loss" && o.args["--metric"] != "error") {
        error = "--metric must be loss or error."; return false;
    }
    if (o.args.contains("--bound-mode") && o.args["--bound-mode"] != "fan" && o.args["--bound-mode"] != "abs") {
        error = "--bound-mode must be fan or abs."; return false;
    }
    o.summarizeOnly = o.args.contains("--summarize-only");
    o.noResume = o.args.contains("--no-resume");
    if (!o.summarizeOnly && !QFileInfo(o.dir).isDir()) {
        error = QString("Dataset folder not found: %1").arg(QDir::toNativeSeparators(o.dir)); return false;
    }
    return true;
}

// ── Experiment <-> options ────────────────────────────────────────────────────
QStringList geRequestedMethods(const GEBatchOptions& o, const QString& cfg) {
    const bool fixed = o.args.value("--total-budget").toLongLong() > 0;
    // default of a grammar configuration: the classic rows; "-sage" configurations add the self-adaptive rows
    QStringList all = cfg == "baselines" ? geBaselineIds()
                    : cfg.endsWith("-sage") ? geGeMethodIds(fixed) : geClassicGeMethodIds(fixed);
    if (o.args.contains("--methods")) all = cfg == "baselines" ? geBaselineIds() : geGeMethodIds(fixed);
    if (o.args.contains("--methods")) {
        const QStringList want = o.args["--methods"].split(',', Qt::SkipEmptyParts);
        QStringList keep;
        for (const QString& m : all) if (want.contains(m)) keep << m;
        // explicitly requested GE ranks ge2..geK
        // (only ids ge<digits>: "genetic" and "gnn" are reference models of configuration "baselines")
        static const QRegularExpression rank("^ge\\d+$");
        if (cfg != "baselines") for (const QString& m : want) if (rank.match(m).hasMatch() && !keep.contains(m)) keep << m;
        return keep;
    }
    return all;
}

bool geExperimentFromBatchOptions(const GEBatchOptions& o, GEExperiment& e, QStringList& unreadable) {
    e = GEExperiment();
    auto raw = [&](const char* k) { QString v = o.args.value(k); v.replace('\\', '/'); return v; };
    e.name = QFileInfo(o.prefix).fileName();
    e.dataDir = raw("--dir");
    e.output = raw("--out");
    e.folds = geFoldsToText(o.folds);
    e.seeds = o.seeds;
    e.configs.clear();
    for (const QString& c : o.configs) if (c != "baselines") e.configs << c;
    if (e.configs.isEmpty()) e.configs << "unified-ge";
    static const QSet<QString> structural = {"--dir", "--datasets", "--out", "--folds", "--seeds", "--configs",
                                             "--methods", "--experiment", "--summarize-only", "--no-resume", "--help",
                                             "--dev-datasets"};
    for (auto it = o.args.begin(); it != o.args.end(); ++it)
        if (!structural.contains(it.key())) e.options[it.key().mid(2)] = it.value();
    e.development = o.devDatasets;
    e.models = geDefaultModels(e.fixedBudget());
    const bool hasBase = o.configs.contains("baselines");
    const bool hasGe = !e.configs.isEmpty() && o.configs.size() > (hasBase ? 1 : 0);
    QSet<QString> run;
    for (const QString& c : o.configs) for (const QString& m : geRequestedMethods(o, c)) run.insert(m);
    for (auto& m : e.models) {
        const bool isBase = geBaselineIds().contains(m.id);
        m.run = run.contains(m.id) && (isBase ? hasBase : hasGe);
    }
    unreadable.clear();
    for (const QString& ds : o.datasets) {
        const QString t = geDetectDatasetTask(o.dir, ds);
        if (t == "classification") e.classification << ds;
        else if (t == "regression") e.regression << ds;
        else unreadable << ds;
    }
    return !e.datasets().isEmpty();
}

// ── Runner ────────────────────────────────────────────────────────────────────
GEBatchRunner::GEBatchRunner(QObject* parent) : QObject(parent) {}

void GEBatchRunner::requestStop() {
    m_stop.store(true);
    QMutexLocker lock(&m_mx);
    if (m_current) m_current->requestStop();
}

void GEBatchRunner::run() {
    try {
        m_stop.store(false);
        if (m_opt.help) { emit message(batchUsage()); emit finished(false, QString()); return; }
        if (!m_opt.experimentPath.isEmpty()) emit message("Experiment file: " + QDir::toNativeSeparators(m_opt.experimentPath));
        for (const QString& w : m_opt.warnings) emit message("warning: " + w);
        const QString prefix = m_opt.prefix;
        if (!QDir().mkpath(QFileInfo(prefix).absolutePath())) {
            emit errorOccurred("Cannot create the output folder " + QDir::toNativeSeparators(QFileInfo(prefix).absolutePath()));
            return;
        }

        // The settings of every experiment are kept next to its results.
        if (!m_opt.summarizeOnly) {
            const QString expOut = prefix + "_experiment.json";
            if (!m_opt.experimentPath.isEmpty()) {
                if (QFileInfo(m_opt.experimentPath).absoluteFilePath() != QFileInfo(expOut).absoluteFilePath()) {
                    QFile::remove(expOut);
                    QFile::copy(m_opt.experimentPath, expOut);
                }
            } else {
                GEExperiment ex; QStringList bad;
                if (geExperimentFromBatchOptions(m_opt, ex, bad)) { QString err; geSaveExperiment(expOut, ex, err); }
            }
        }

        // existing results (resume, per model: a job runs again only for the models it lacks)
        std::vector<Row> rows;
        QMap<QString, QSet<QString>> have;   // ds|fold|cfg|seed -> methods present
        auto keyOf = [](const QString& ds, int fold, const QString& cfg, int seed) {
            return QString("%1|%2|%3|%4").arg(ds).arg(fold).arg(cfg).arg(seed);
        };
        const QString rawPath = prefix + "_raw.csv";
        if (!m_opt.noResume && QFileInfo::exists(rawPath)) {
            QFile f(rawPath);
            if (f.open(QIODevice::ReadOnly | QIODevice::Text)) {
                QTextStream t(&f); t.readLine();
                while (!t.atEnd()) {
                    Row r;
                    if (fromCsv(t.readLine().trimmed(), r)) {
                        rows.push_back(r);
                        have[keyOf(r.ds, r.fold, r.cfg, r.seed)].insert(r.method);
                    }
                }
            }
            emit message(QString("Resuming: %1 runs already in %2").arg(have.size()).arg(QDir::toNativeSeparators(rawPath)));
        } else if (!m_opt.summarizeOnly) {
            QFile f(rawPath);
            if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
                emit errorOccurred("Cannot write " + QDir::toNativeSeparators(rawPath)); return;
            }
            QTextStream t(&f); t << kHeader << "\n";
        }

        if (!m_opt.summarizeOnly) {
            struct Job { QString ds; int fold; QString cfg; int seed; QString path; QStringList methods; bool replace = false; };
            int complete = 0;
            std::vector<Job> jobs;
            QSet<QString> warned;
            for (const QString& ds : m_opt.datasets) for (int fold : m_opt.folds)
                for (const QString& cfg : m_opt.configs) for (int s = 1; s <= m_opt.seeds; ++s) {
                    const QString path = QString("%1/%2_%3.train").arg(m_opt.dir, ds).arg(fold);
                    if (!QFileInfo::exists(path) || QFileInfo(path).size() == 0) {
                        if (!warned.contains(path)) {
                            emit message("skip (missing or empty): " + QDir::toNativeSeparators(path));
                            warned.insert(path);
                        }
                        continue;
                    }
                    const QStringList req = geRequestedMethods(m_opt, cfg);
                    if (req.isEmpty()) continue;
                    const QString key = keyOf(ds, fold, cfg, s);
                    QStringList missing;
                    for (const QString& m : req) if (!have.value(key).contains(m)) missing << m;
                    if (missing.isEmpty()) { ++complete; continue; }
                    Job j{ds, fold, cfg, s, path, missing, false};
                    if (cfg != "baselines" && have.contains(key)) {
                        // GE rows of one run belong together (the ablations use the same phenotype):
                        // a GE run that lacks a model is repeated for all of its requested models.
                        j.methods = req;
                        j.replace = true;
                    }
                    jobs.push_back(j);
                }
            // Smallest datasets first (by train-file size): most datasets finish early, and the
            // long ones (tens of thousands of rows) run last. Order within a dataset is unchanged.
            {
                QMap<QString, qint64> dsSize;
                for (const Job& j : jobs) if (!dsSize.contains(j.ds)) dsSize[j.ds] = QFileInfo(j.path).size();
                std::stable_sort(jobs.begin(), jobs.end(), [&](const Job& a, const Job& b) {
                    return dsSize.value(a.ds) < dsSize.value(b.ds);
                });
            }
            const int nJobs = static_cast<int>(jobs.size());
            emit message(QString("%1 runs to do.").arg(nJobs));
            emit batchStarted(nJobs, complete);

            QElapsedTimer total; total.start();
            int idx = 0;
            for (const Job& j : jobs) {
                if (m_stop.load()) break;
                GEStudioConfig c;
                applyConfig(m_opt, j.cfg, c);
                c.methods = j.methods;
                c.datasetPaths << j.path;
                c.seed = static_cast<quint32>(j.seed * 1000 + j.fold);
                const QString label = QString("%1 fold %2 %3 seed %4").arg(j.ds).arg(j.fold).arg(j.cfg).arg(j.seed);

                GEStudioWorker w;
                w.setConfig(c);
                QString failure;
                std::vector<Row> got;
                int finalRows = 1;
                QObject::connect(&w, &GEStudioWorker::errorOccurred, [&](QString m) { failure = m; });
                QObject::connect(&w, &GEStudioWorker::generationDone,
                    [&](int gen, int totalGens, double, QString, double, double, int, int, qint64) {
                        emit progress(idx, nJobs, 0.7 * gen / std::max(1, totalGens), label);
                    });
                QObject::connect(&w, &GEStudioWorker::finalStageStarted, [&](int n) {
                    finalRows = std::max(1, n);
                    emit progress(idx, nJobs, 0.7, label + " — final training");
                });
                QObject::connect(&w, &GEStudioWorker::finalRowDone, [&](int row, int, QString) {
                    emit progress(idx, nJobs, 0.7 + 0.3 * row / finalRows, label + " — final training");
                });
                QObject::connect(&w, &GEStudioWorker::datasetFinished, [&](GEDatasetResult r) {
                    for (const GEFinalRow& x : r.rows) {
                        Row row;
                        row.ds = j.ds; row.fold = j.fold; row.cfg = j.cfg; row.seed = j.seed; row.method = x.id;
                        row.cls = r.isCls; row.trainErr = x.trainErr;
                        row.testErr = x.testErr >= 0 ? x.testErr : x.valErr;   // no test file: held-out part
                        row.params = x.params; row.evals = x.evals; row.secs = x.seconds;
                        row.searchEvals = r.searchCalls; row.searchSecs = r.searchSeconds;
                        row.gensRun = r.gensRun; row.uniqueNets = r.uniqueEvals; row.cacheHits = r.cacheHits;
                        row.testN = r.hasTest ? r.testSamples : 0;
                        got.push_back(row);
                    }
                });
                emit progress(idx, nJobs, 0.0, label);
                {
                    QMutexLocker lock(&m_mx);
                    m_current = &w;
                }
                if (m_stop.load()) w.requestStop();   // stop pressed between jobs
                QElapsedTimer t; t.start();
                w.run();
                {
                    QMutexLocker lock(&m_mx);
                    m_current = nullptr;
                }
                // A stopped run is incomplete: never write it, so resuming re-runs it.
                if (m_stop.load()) break;
                ++idx;
                const double per = total.elapsed() / 1000.0 / idx;
                const QString line = QString("[%1/%2] %3  %4 s  (ETA %5 min)").arg(idx).arg(nJobs).arg(label)
                                         .arg(t.elapsed() / 1000.0, 0, 'f', 1).arg(per * (nJobs - idx) / 60.0, 0, 'f', 1);
                if (!failure.isEmpty() || got.empty()) {
                    emit message(QString("FAILED %1: %2").arg(label, failure.isEmpty() ? QString("no result") : failure));
                    emit jobFinished(idx, nJobs, line + "  FAILED", false);
                    continue;
                }
                if (j.replace) {
                    // drop the old rows of this run and rewrite the file (temporary file + rename)
                    rows.erase(std::remove_if(rows.begin(), rows.end(), [&](const Row& r) {
                        return r.ds == j.ds && r.fold == j.fold && r.cfg == j.cfg && r.seed == j.seed; }), rows.end());
                    for (const Row& r : got) rows.push_back(r);
                    const QString tmp = rawPath + ".tmp";
                    QFile f(tmp);
                    bool ok = f.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Truncate);
                    if (ok) { QTextStream ts(&f); ts << kHeader << "\n"; for (const Row& r : rows) ts << toCsv(r) << "\n"; f.close(); }
                    ok = ok && (!QFileInfo::exists(rawPath) || QFile::remove(rawPath)) && QFile::rename(tmp, rawPath);
                    if (!ok) emit message("WARNING: cannot rewrite " + QDir::toNativeSeparators(rawPath));
                } else {
                    QFile f(rawPath);
                    if (f.open(QIODevice::Append | QIODevice::Text)) {
                        QTextStream ts(&f);
                        for (const Row& r : got) ts << toCsv(r) << "\n";
                    } else {
                        emit message("WARNING: cannot append to " + QDir::toNativeSeparators(rawPath));
                    }
                    for (const Row& r : got) rows.push_back(r);
                }
                emit message(line);
                emit jobFinished(idx, nJobs, line, true);
            }
        }

        if (m_stop.load()) {
            emit message("Stopped. The unfinished run was discarded; run the same command again to resume.");
        }
        // Reports cover what this experiment asks for; other rows stay in the file (they can be re-added).
        {
            std::vector<Row> sel;
            for (const Row& r : rows)
                if (m_opt.datasets.contains(r.ds) && m_opt.configs.contains(r.cfg) &&
                    geRequestedMethods(m_opt, r.cfg).contains(r.method)) sel.push_back(r);
            if (sel.size() != rows.size())
                emit message(QString("Reports use %1 of the %2 result rows (the others belong to datasets, configurations or "
                                     "models that are not part of this experiment).").arg(sel.size()).arg(rows.size()));
            rows.swap(sel);
        }
        if (!rows.empty()) {
            QString text;
            QTextStream out(&text);
            summarize(rows, prefix, out);
            // overall tables: classification / regression, with and without BNF
            QVector<GERawRow> rr;
            rr.reserve(static_cast<int>(rows.size()));
            for (const Row& r : rows) {
                GERawRow x;
                x.ds = r.ds; x.fold = r.fold; x.cfg = r.cfg; x.seed = r.seed; x.method = r.method; x.cls = r.cls;
                x.trainErr = r.trainErr; x.testErr = r.testErr; x.params = r.params; x.evals = r.evals; x.secs = r.secs;
                x.searchEvals = r.searchEvals; x.searchSecs = r.searchSecs; x.gensRun = r.gensRun;
                x.uniqueNets = r.uniqueNets; x.cacheHits = r.cacheHits; x.testN = r.testN;
                rr.append(x);
            }
            GEReportFilter tests;
            tests.excludeFromTests = m_opt.devDatasets;
            const GEOverall ov = geBuildOverall(rr, tests);
            out << geOverallText(ov);
            try {
                geWriteOverallXlsx(ov, prefix + "_overall.xlsx");
                out << "Written: " << QDir::toNativeSeparators(prefix + "_overall.xlsx") << "  (Classification / Regression tables, with and without BNF)\n";
            } catch (const std::exception& ex) {
                out << "overall xlsx export failed: " << ex.what() << "\n";
            }
            out.flush();
            emit message(text);
        } else {
            emit message("No results to summarize yet.");
        }
        emit finished(m_stop.load(), prefix);
    } catch (const std::exception& ex) {
        emit errorOccurred(QString::fromStdString(ex.what()));
    }
}

} // namespace NeuralStudio
