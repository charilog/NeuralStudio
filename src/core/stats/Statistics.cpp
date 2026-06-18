#include "Statistics.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cassert>

// M_SQRT1_2 = 1/√2 — not in C++ standard; define if missing (MSVC without _USE_MATH_DEFINES)
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

namespace NeuralStudio {

// ─── stars ───────────────────────────────────────────────────────────────────
std::string WilcoxonResult::stars() const {
    if (!valid) return "—";
    if (pValue < 0.001) return "***";
    if (pValue < 0.01)  return "**";
    if (pValue < 0.05)  return "*";
    return "ns";
}
std::string FriedmanResult::stars() const {
    if (!valid) return "—";
    if (pValue < 0.001) return "***";
    if (pValue < 0.01)  return "**";
    if (pValue < 0.05)  return "*";
    return "ns";
}

// ─── normalCDF ───────────────────────────────────────────────────────────────
double Statistics::normalCDF(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

// ─── chi2pvalue (Wilson-Hilferty approximation) ───────────────────────────
double Statistics::chi2pvalue(double chi2, int df) {
    if (df <= 0 || chi2 <= 0) return 1.0;
    double nu = static_cast<double>(df);
    double h  = 1.0 - 2.0 / (9.0 * nu);
    double s  = std::sqrt(2.0 / (9.0 * nu));
    double z  = (std::pow(chi2 / nu, 1.0 / 3.0) - h) / s;
    // Right-tail: P(X > chi2) = P(Z > z) = Φ(-z)
    return normalCDF(-z);
}

// ─── boxStats ────────────────────────────────────────────────────────────────
Statistics::BoxStats Statistics::boxStats(std::vector<double> v) {
    BoxStats bs;
    if (v.size() < 2) return bs;
    std::sort(v.begin(), v.end());
    const int n = static_cast<int>(v.size());

    auto percentile = [&](double pct) {
        double idx = pct * (n - 1);
        int lo = static_cast<int>(idx);
        int hi = lo + 1;
        double frac = idx - lo;
        if (hi >= n) return v[n-1];
        return v[lo] * (1.0 - frac) + v[hi] * frac;
    };

    bs.q1 = percentile(0.25);
    bs.q2 = percentile(0.50);
    bs.q3 = percentile(0.75);
    const double iqr = bs.q3 - bs.q1;
    const double lo  = bs.q1 - 1.5 * iqr;
    const double hi  = bs.q3 + 1.5 * iqr;

    // Whiskers = most extreme non-outlier
    bs.q0 = bs.q1;  for (double x : v) { if (x >= lo) { bs.q0 = x; break; } }
    bs.q4 = bs.q3;  for (auto it = v.rbegin(); it != v.rend(); ++it) { if (*it <= hi) { bs.q4 = *it; break; } }

    for (double x : v)
        if (x < lo || x > hi) bs.outliers.push_back(x);

    bs.mean  = std::accumulate(v.begin(), v.end(), 0.0) / n;
    bs.valid = true;
    return bs;
}

// ─── wilcoxon ────────────────────────────────────────────────────────────────
WilcoxonResult Statistics::wilcoxon(const std::vector<double>& a,
                                     const std::vector<double>& b) {
    WilcoxonResult res;
    if (a.size() != b.size() || a.empty()) return res;

    // Differences (exclude zeros — tied pairs)
    std::vector<double> diffs;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        if (std::abs(d) > 1e-12) diffs.push_back(d);
    }
    res.n = static_cast<int>(diffs.size());
    if (res.n < 3) return res;   // too few for a meaningful test

    // Rank |diffs| with average-rank tie-breaking
    const int n = res.n;
    std::vector<double> absDiffs(n);
    for (int i = 0; i < n; ++i) absDiffs[i] = std::abs(diffs[i]);
    std::vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return absDiffs[a] < absDiffs[b]; });

    std::vector<double> ranks(n);
    int i = 0;
    while (i < n) {
        int j = i;
        while (j < n && std::abs(absDiffs[idx[j]] - absDiffs[idx[i]]) < 1e-12) ++j;
        double avgRank = (i + j + 2) / 2.0;   // 1-indexed average
        for (int k = i; k < j; ++k) ranks[idx[k]] = avgRank;
        i = j;
    }

    double wPlus = 0, wMinus = 0;
    for (int i = 0; i < n; ++i)
        (diffs[i] > 0 ? wPlus : wMinus) += ranks[i];

    // Normal approximation with continuity correction
    double W    = std::min(wPlus, wMinus);
    double mean = n * (n + 1) / 4.0;
    double var  = n * (n + 1) * (2 * n + 1) / 24.0;
    if (var < 1e-14) return res;
    double z = (W - mean + 0.5) / std::sqrt(var);  // continuity correction

    res.wPlus  = wPlus;
    res.wMinus = wMinus;
    res.z      = z;
    res.pValue = 2.0 * normalCDF(-std::abs(z));    // two-tailed
    res.valid  = true;
    return res;
}

// ─── friedman ────────────────────────────────────────────────────────────────
FriedmanResult Statistics::friedman(
    const std::vector<std::string>& names,
    const std::vector<std::vector<double>>& data)
{
    FriedmanResult res;
    const int k = static_cast<int>(names.size());
    if (k < 2 || data.empty() || (int)data.size() != k) return res;
    const int N = static_cast<int>(data[0].size());
    if (N < 2) return res;
    for (int j = 1; j < k; ++j) if ((int)data[j].size() != N) return res;

    // For each dataset (row), rank the k optimizers
    // Lower metric = better rank (rank 1 = best)
    std::vector<double> R(k, 0.0);
    for (int row = 0; row < N; ++row) {
        std::vector<int> order(k); std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b){ return data[a][row] < data[b][row]; });
        // Assign ranks with tie averaging
        int i = 0;
        while (i < k) {
            int j = i;
            while (j < k && std::abs(data[order[j]][row] - data[order[i]][row]) < 1e-12) ++j;
            double avgR = (i + j + 2) / 2.0;
            for (int m = i; m < j; ++m) R[order[m]] += avgR;
            i = j;
        }
    }

    // Average ranks
    std::vector<double> avgR(k);
    for (int j = 0; j < k; ++j) avgR[j] = R[j] / N;

    // Friedman χ² statistic
    double sumSq = 0;
    for (int j = 0; j < k; ++j) sumSq += R[j] * R[j];
    double chi2 = (12.0 / (N * k * (k + 1))) * sumSq - 3.0 * N * (k + 1);

    // Sort optimizers by average rank (best first)
    std::vector<int> rankOrder(k); std::iota(rankOrder.begin(), rankOrder.end(), 0);
    std::sort(rankOrder.begin(), rankOrder.end(),
              [&](int a, int b){ return avgR[a] < avgR[b]; });

    res.chi2      = chi2;
    res.df        = k - 1;
    res.pValue    = chi2pvalue(chi2, k - 1);
    res.avgRanks  = avgR;
    res.rankOrder = rankOrder;
    res.N         = N;
    res.k         = k;
    res.valid     = true;
    return res;
}

} // namespace NeuralStudio
