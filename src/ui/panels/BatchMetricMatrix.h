#pragma once
// BatchMetricMatrix.h — the optimizer × dataset matrix of one metric, shared by the
// Wilcoxon and Friedman tabs of the batch mode.
//
//  * A value is valid by its task, not by its sign: R² may be negative (a model worse
//    than the mean), which the old "v < 0 means missing" rule silently dropped, removing
//    exactly the datasets on which some optimizer failed.
//  * Several results of the same optimizer on the same dataset (repeated jobs, and the
//    folds name_1 ... name_K of one dataset) are AVERAGED into one observation. Before, the
//    last result overwrote the others, and every fold counted as a separate dataset, which
//    treats the dependent folds of one dataset as independent observations.
#include "core/batch/BatchJob.h"
#include <QMap>
#include <QRegularExpression>
#include <QSet>
#include <QString>
#include <QVector>
#include <algorithm>
#include <cmath>
#include <vector>

namespace NeuralStudio {

inline double batchMetricValue(const BatchJobResult& r, const QString& m) {
    if (m == "valErrorPct")   return r.valErrorPct;
    if (m == "testErrorPct")  return r.testErrorPct;
    if (m == "trainErrorPct") return r.trainErrorPct;
    if (m == "valMAE")        return r.valMAE;
    if (m == "testMAE")       return r.testMAE;
    if (m == "trainMAE")      return r.trainMAE;
    if (m == "valRMSE")       return r.valRMSE;
    if (m == "testRMSE")      return r.testRMSE;
    if (m == "valR2")         return r.valR2;
    if (m == "testR2")        return r.testR2;
    return std::nan("");
}

// true when the metric exists for this result
inline bool batchMetricValid(const BatchJobResult& r, const QString& m, double v) {
    if (!r.success || !std::isfinite(v)) return false;
    const bool reg = r.task == TaskType::Regression;
    const bool regMetric = m.contains("MAE") || m.contains("RMSE") || m.contains("R2");
    if (reg != regMetric) return false;
    if (m.startsWith("test") && !r.hasTestSet) return false;
    if (m.contains("R2")) return m != "valR2" || v != -1.0;   // R² may be negative; val R² = -1: no validation split
    return v >= 0.0;                                         // error %, MAE, RMSE: -1 = not computed
}

// "wine_3" -> "wine" (fold suffix), anything else unchanged
inline QString batchDatasetKey(const QString& name) {
    static const QRegularExpression fold("^(.+)_(\\d+)$");
    const auto m = fold.match(name);
    return m.hasMatch() ? m.captured(1) : name;
}

inline bool taskPasses(const BatchJobResult& r, const QString& taskFilter) {
    if (taskFilter == "Regression" && r.task != TaskType::Regression) return false;
    if (taskFilter == "Binary Classification" && r.task != TaskType::BinaryClassification) return false;
    if (taskFilter == "Multi-class" && r.task != TaskType::MultiClassClassification) return false;
    return true;
}

// Fills optNames (sorted), dsNames (datasets on which EVERY optimizer has a value) and
// matrix[optimizer][dataset] (mean over repeats / folds). Returns false when there are
// fewer than minOpt optimizers or 2 datasets.
inline bool buildBatchMetricMatrix(const std::vector<BatchJobResult>& results, const QString& metricKey,
                                   const QString& taskFilter, int minOpt,
                                   QVector<QString>& optNames, QVector<QVector<double>>& matrix,
                                   QVector<QString>& dsNames, int* averagedRuns = nullptr) {
    QMap<QString, QPair<double, int>> acc;   // "opt|ds" -> (sum, n)
    QSet<QString> optSet, dsSet;
    int extra = 0;
    for (const auto& r : results) {
        if (!taskPasses(r, taskFilter)) continue;
        const double v = batchMetricValue(r, metricKey);
        if (!batchMetricValid(r, metricKey, v)) continue;
        const QString o = r.optimizerName.toLower(), d = batchDatasetKey(r.datasetName);
        auto& a = acc[o + "|" + d];
        if (a.second > 0) ++extra;
        a.first += v; ++a.second;
        optSet.insert(o); dsSet.insert(d);
    }
    if (averagedRuns) *averagedRuns = extra;
    if (optSet.size() < minOpt || dsSet.size() < 2) return false;
    optNames = QVector<QString>(optSet.begin(), optSet.end());
    std::sort(optNames.begin(), optNames.end());
    QVector<QString> all(dsSet.begin(), dsSet.end());
    std::sort(all.begin(), all.end());
    dsNames.clear();
    for (const auto& d : all) {
        bool ok = true;
        for (const auto& o : optNames) if (!acc.contains(o + "|" + d)) { ok = false; break; }
        if (ok) dsNames.push_back(d);
    }
    if (dsNames.size() < 2) return false;
    matrix = QVector<QVector<double>>(optNames.size(), QVector<double>(dsNames.size(), 0.0));
    for (int j = 0; j < optNames.size(); ++j)
        for (int i = 0; i < dsNames.size(); ++i) {
            const auto a = acc.value(optNames[j] + "|" + dsNames[i]);
            matrix[j][i] = a.first / a.second;
        }
    return true;
}

} // namespace NeuralStudio
