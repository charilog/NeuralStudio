#pragma once
// GEReport.h — overall result tables of Mode 6 batch experiments (Qt Core only).
//
// From the raw results (<prefix>_raw.csv) it builds, separately for classification
// and regression and for every grammar configuration, the table that papers report:
//
//     dataset | methods WITHOUT BNF ... | methods WITH BNF ...
//
// One value per dataset = mean test error over the folds (the seeds of a fold are
// averaged first). Below the datasets: average, average Friedman rank, wins/ties/losses
// and Wilcoxon p of the main GE method against every other method (datasets as units),
// Friedman p-value and the Nemenyi critical difference (Demšar 2006).
#include <QString>
#include <QStringList>
#include <QVector>
#include <cmath>
#include <limits>

namespace NeuralStudio {

struct GEMethodInfo {
    QString id;
    QString label;
    bool    bnf = false;      // does the model come from the BNF grammar?
    QString description;
};

// Every method id the worker can produce, in report order (non-BNF first).
const QVector<GEMethodInfo>& geMethodCatalog();
// Info for an id; ids "ge2".."geK" are generated on the fly. nullptr if unknown.
GEMethodInfo geMethodInfo(const QString& id, bool* known = nullptr);

struct GERawRow {
    QString ds; int fold = 0; QString cfg; int seed = 0; QString method;
    bool    cls = true;
    double  trainErr = 0, testErr = 0;
    int     params = 0; qint64 evals = 0; double secs = 0;
    qint64  searchEvals = 0; double searchSecs = 0;
    int     gensRun = 0, uniqueNets = 0, cacheHits = 0, testN = 0;
};

bool geLoadRawCsv(const QString& path, QVector<GERawRow>& rows, QString& error);

struct GEOverallTable {
    QString         task;          // "classification" | "regression"
    QString         config;        // e.g. unified-ge
    QStringList     methods;       // method ids, report order
    QVector<bool>   bnf;           // per method
    QStringList     datasets;
    QVector<QVector<double>> value;   // [dataset][method] mean test error (NaN = missing)
    QVector<QVector<double>> sd;      // [dataset][method] std over folds
    QVector<int>    folds;            // folds per dataset (complete for all methods)
    QVector<double> seconds;          // [method] mean wall time per run (row training only)
    QVector<double> params;           // [method] mean parameters
    // over the datasets that have every method:
    int             complete = 0;
    QVector<double> average;          // [method]
    QVector<double> avgRank;          // [method] Friedman average rank (1 = best)
    double          friedmanP = std::numeric_limits<double>::quiet_NaN();
    double          cd05 = std::numeric_limits<double>::quiet_NaN();
    QString         reference;        // main GE method compared against the others
    QVector<int>    wins, ties, losses;   // reference vs method, datasets as units
    QVector<double> wilcoxonP;            // reference vs method, on the raw values
    // Scale-free comparison (needed for regression, where MSE differs by orders of magnitude
    // between datasets): relative difference rd = (method - reference) / mean(|method|, |reference|)
    // per dataset (0 when both are 0); rd > 0 means the reference is better.
    QVector<double> relDiffMedian;        // [method] median rd over the datasets, in %
    QVector<double> geoRatio;             // [method] geometric mean of method / reference (datasets where both > 0)
    QVector<double> wilcoxonRelP;         // [method] Wilcoxon signed-rank test of rd against 0
    QVector<double> holmP;                // [method] wilcoxonRelP, Holm-adjusted over all methods compared with the reference
};

// Pre-declared hypotheses "reference < method" (H1 fixed network, H2 grid search, H3 without bound commands,
// H4 without features, H5 GNN), per task and over both tasks together, Holm-adjusted over the hypotheses
// of the same configuration and scope.
struct GEHypothesisTest {
    QString config, scope;                // scope: classification | regression | all
    QStringList excluded;                 // datasets left out (development datasets, incomplete folds)
    QString id, reference, method, meaning;
    int     n = 0, better = 0, ties = 0, worse = 0;   // datasets where the reference is better / tie / worse
    double  relDiffMedian = std::numeric_limits<double>::quiet_NaN();   // %
    double  geoRatio = std::numeric_limits<double>::quiet_NaN();
    double  p = std::numeric_limits<double>::quiet_NaN();
    double  holmP = std::numeric_limits<double>::quiet_NaN();
};

struct GEOverall {
    QVector<GEOverallTable> tables;
    QVector<GEHypothesisTest> hypotheses;
    QStringList             warnings;
    bool isEmpty() const { return tables.isEmpty(); }
};

// Which rows enter the tables (empty list = no restriction). "baselines" in `configs` keeps the
// reference models.
struct GEReportFilter {
    QStringList datasets, configs, methods;
    QStringList excludeFromTests;   // development datasets: shown in the tables, left out of the hypothesis tests
    bool keeps(const GERawRow& r) const {
        return (datasets.isEmpty() || datasets.contains(r.ds)) &&
               (configs.isEmpty() || configs.contains(r.cfg)) &&
               (methods.isEmpty() || methods.contains(r.method));
    }
};

GEOverall geBuildOverall(const QVector<GERawRow>& rows, const GEReportFilter& filter = GEReportFilter());
QString   geOverallText(const GEOverall& o);
QString   geHypothesesText(const GEOverall& o);
// Holm step-down adjustment of p-values (NaN entries are ignored and stay NaN).
QVector<double> geHolm(const QVector<double>& p);
// Writes sheets "Classification", "Regression", "Hypotheses", "Classification (std)", "Regression (std)",
// "Methods" and "Cost". Throws std::runtime_error on I/O failure.
void      geWriteOverallXlsx(const GEOverall& o, const QString& path);

} // namespace NeuralStudio
