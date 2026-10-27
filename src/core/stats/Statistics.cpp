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

// ─── chi2pvalue: exact upper tail Q(df/2, chi2/2) ─────────────────────────
// Regularized upper incomplete gamma (series for x < a+1, Lentz continued
// fraction otherwise; Numerical Recipes 6.2). Replaces the Wilson-Hilferty
// approximation, which was off by up to ~2x for small p-values.
namespace {
double upperGammaQ(double a, double x) {
    if (x <= 0) return 1.0;
    const double lnPre = -x + a * std::log(x) - std::lgamma(a);
    if (x < a + 1.0) {
        double ap = a, sum = 1.0 / a, del = sum;
        for (int n = 0; n < 1000; ++n) {
            ap += 1.0; del *= x / ap; sum += del;
            if (std::fabs(del) < std::fabs(sum) * 1e-15) break;
        }
        return std::max(0.0, 1.0 - sum * std::exp(lnPre));
    }
    const double tiny = 1e-300;
    double b = x + 1.0 - a, c = 1.0 / tiny, d = 1.0 / b, h = d;
    for (int i = 1; i < 1000; ++i) {
        const double an = -i * (i - a);
        b += 2.0;
        d = an * d + b; if (std::fabs(d) < tiny) d = tiny;
        c = b + an / c; if (std::fabs(c) < tiny) c = tiny;
        d = 1.0 / d;
        const double del = d * c; h *= del;
        if (std::fabs(del - 1.0) < 1e-15) break;
    }
    return std::min(1.0, std::exp(lnPre) * h);
}

// P(range of k iid N(0,1) < w)  =  k * Int phi(z) [Phi(z) - Phi(z-w)]^(k-1) dz
// (studentized range with infinite degrees of freedom, as used by Nemenyi).
double rangeCdfInf(double w, int k) {
    if (w <= 0) return 0.0;
    const int n = 4000;                    // Simpson on [-9, 9]
    const double a = -9.0, b = 9.0, h = (b - a) / n;
    double s = 0;
    for (int i = 0; i <= n; ++i) {
        const double z = a + i * h;
        const double phi = std::exp(-0.5 * z * z) / 2.5066282746310002;
        const double f = phi * std::pow(Statistics::normalCDF(z) - Statistics::normalCDF(z - w), k - 1);
        s += f * ((i == 0 || i == n) ? 1.0 : (i % 2 ? 4.0 : 2.0));
    }
    return std::min(1.0, std::max(0.0, k * s * h / 3.0));
}
// Critical q_alpha / sqrt(2) for k groups (Demsar 2006, Table 5, any k).
double nemenyiQ(int k, double alpha) {
    double lo = 0.0, hi = 10.0;
    for (int it = 0; it < 80; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (1.0 - rangeCdfInf(mid * std::sqrt(2.0), k) > alpha) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}
} // namespace

double Statistics::chi2pvalue(double chi2, int df) {
    if (df <= 0 || !(chi2 > 0)) return 1.0;
    return upperGammaQ(0.5 * df, 0.5 * chi2);
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

// ─── wilcoxonExactPValue ───────────────────────────────────────────────────────
//  Exact two-tailed p-value for the Wilcoxon signed-rank test on n UNTIED
//  ranks {1, ..., n}. Under H0, each rank is independently assigned to the
//  positive or negative group with probability 1/2, so the distribution of
//  W+ (sum of positive-group ranks) is the count of subsets of {1..n} that
//  sum to each possible total — computed via the classic 0/1-knapsack style
//  DP (equivalent to the coefficients of the generating function
//  prod_{i=1}^{n} (1 + x^i). This is the standard exact method used by
//  R's wilcox.test() and SciPy's wilcoxon(..., mode='exact').
static double wilcoxonExactPValue(int n, double wObserved) {
    const int maxSum = n * (n + 1) / 2;
    std::vector<double> dp(maxSum + 1, 0.0);
    dp[0] = 1.0;
    for (int r = 1; r <= n; ++r)
        for (int s = maxSum; s >= r; --s)
            dp[s] += dp[s - r];

    double total = std::pow(2.0, n);
    const int wFloor = static_cast<int>(std::floor(wObserved + 1e-9));
    double cum = 0.0;
    for (int s = 0; s <= wFloor && s <= maxSum; ++s) cum += dp[s];

    const double pOneTail = cum / total;
    return std::min(1.0, 2.0 * pOneTail);   // two-tailed
}

// ─── Tie/zero-difference tolerance ─────────────────────────────────────────────
//  NeuralStudio's metrics (Val Err%, Test Err%, etc.) are percentages in the
//  0–100 range. Two results that are mathematically identical (e.g. both
//  represent "1 misclassified out of 14 samples") can still differ at the
//  ~1e-9 to 1e-11 level purely from floating-point computation-order noise
//  (different code paths arriving at the same rate via different arithmetic,
//  e.g. wrong/total*100 vs (1-correct/total)*100). A tolerance of 1e-12 is
//  far too tight to absorb this and incorrectly treats such pairs as "really
//  different", which corrupts the rank structure and can flip the test
//  between its exact and normal-approximation branches. The tolerance is
//  scale-aware: two values tie when they differ by at most 1e-9 of their
//  magnitude (and of 1), which absorbs such noise. The former absolute 1e-6
//  also merged real differences of small regression errors (MSE ~1e-4).
static inline bool nearlyEqual(double x, double y) {
    return std::abs(x - y) <= 1e-9 * std::max({1.0, std::abs(x), std::abs(y)});
}

WilcoxonResult Statistics::wilcoxon(const std::vector<double>& a,
                                     const std::vector<double>& b) {
    WilcoxonResult res;
    if (a.size() != b.size() || a.empty()) return res;

    // Differences (exclude zeros — tied pairs)
    std::vector<double> diffs;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        if (!nearlyEqual(a[i], b[i])) diffs.push_back(d);
    }
    res.n = static_cast<int>(diffs.size());
    // The exact signed-rank distribution is well-defined for ANY n >= 1
    // (this matches R's wilcox.test and SciPy's wilcoxon, both of which
    // compute a p-value even for n=1 or n=2 — with very limited power, but
    // a real number rather than "not applicable"). The only case that
    // truly cannot be tested is n=0, i.e. every paired difference was
    // exactly zero, which carries no directional information at all.
    if (res.n < 1) return res;

    // Rank |diffs| with average-rank tie-breaking; track whether any ties
    // actually occurred (needed to decide exact vs. normal-approximation).
    const int n = res.n;
    std::vector<double> absDiffs(n);
    for (int i = 0; i < n; ++i) absDiffs[i] = std::abs(diffs[i]);
    std::vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return absDiffs[a] < absDiffs[b]; });

    std::vector<double> ranks(n);
    bool hasTies = false;
    double tieSum = 0.0;   // Σ(t³ - t) over tie groups, for the variance correction below
    int i = 0;
    while (i < n) {
        int j = i;
        while (j < n && nearlyEqual(absDiffs[idx[j]], absDiffs[idx[i]])) ++j;
        const int t = j - i;
        if (t > 1) { hasTies = true; tieSum += (static_cast<double>(t) * t * t - t); }
        // Average 1-indexed rank of a tie group occupying 0-indexed slots
        // [i, j): the 1-indexed ranks involved are i+1 .. j, whose mean is
        // (i+j+1)/2. (NOTE: an earlier version of this formula used
        // (i+j+2)/2, which is off by +0.5 on every single rank — verified by
        // checking that ranks of n untied values must sum to n(n+1)/2.)
        double avgRank = (i + j + 1) / 2.0;   // 1-indexed average
        for (int k = i; k < j; ++k) ranks[idx[k]] = avgRank;
        i = j;
    }

    double wPlus = 0, wMinus = 0;
    for (int i = 0; i < n; ++i)
        (diffs[i] > 0 ? wPlus : wMinus) += ranks[i];

    res.wPlus  = wPlus;
    res.wMinus = wMinus;
    const double W = std::min(wPlus, wMinus);

    // ── Exact method ─────────────────────────────────────────────────────────
    // Valid only when ranks are untied (1..n distinct integers); this matches
    // the convention used by R/SciPy, which both fall back to the normal
    // approximation as soon as ties are present. The DP cost is negligible
    // (O(n²)) for the dataset counts this app realistically sees.
    if (!hasTies) {
        res.exact  = true;
        res.pValue = wilcoxonExactPValue(n, W);
        res.z      = 0.0;   // not applicable for the exact method
        res.valid  = true;
        return res;
    }

    // ── Normal approximation with continuity correction (ties present) ───────
    // Variance uses the standard tie-correction term (Lehmann 1975), the same
    // convention used by SAS/SPSS: Var(W) = [n(n+1)(2n+1) - 0.5·Σ(t³−t)] / 24.
    // Without it the variance is biased upward whenever ties are present,
    // which understates significance.
    double mean = n * (n + 1) / 4.0;
    double var  = (n * (n + 1) * (2 * n + 1) - 0.5 * tieSum) / 24.0;
    if (var < 1e-14) return res;
    double z = (W - mean + 0.5) / std::sqrt(var);  // continuity correction

    res.exact  = false;
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
    double tieSum = 0.0;   // Σ(t³ - t) over all tie groups, for the tie correction below
    for (int row = 0; row < N; ++row) {
        std::vector<int> order(k); std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b){ return data[a][row] < data[b][row]; });
        // Assign ranks with tie averaging
        int i = 0;
        while (i < k) {
            int j = i;
            while (j < k && nearlyEqual(data[order[j]][row], data[order[i]][row])) ++j;
            const int t = j - i;
            if (t > 1) tieSum += (static_cast<double>(t) * t * t - t);
            // Same correct formula as in wilcoxon(): mean of 1-indexed
            // ranks i+1..j is (i+j+1)/2, not (i+j+2)/2.
            double avgR = (i + j + 1) / 2.0;
            for (int m = i; m < j; ++m) R[order[m]] += avgR;
            i = j;
        }
    }

    // Average ranks
    std::vector<double> avgR(k);
    for (int j = 0; j < k; ++j) avgR[j] = R[j] / N;

    // Friedman χ² statistic, with the standard tie-correction factor
    // (Conover; matches R's friedman.test and SciPy's friedmanchisquare).
    // Without this correction, the statistic is biased downward whenever
    // ties are present — which is the common case here since optimizer
    // results often land on identical or near-identical values.
    double sumSq = 0;
    for (int j = 0; j < k; ++j) sumSq += R[j] * R[j];
    double chi2 = (12.0 / (N * k * (k + 1))) * sumSq - 3.0 * N * (k + 1);
    const double tieDenom = static_cast<double>(N) * k * (k * k - 1);
    if (tieDenom > 1e-12) {
        const double C = 1.0 - tieSum / tieDenom;
        if (C > 1e-9) chi2 /= C;
    }

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

// ─── nemenyi ───────────────────────────────────────────────────────────────────
//  All-pairs post-hoc on Friedman average ranks. See NemenyiResult in the header
//  for the rationale. The Studentized-range critical values q_alpha (already
//  divided by sqrt(2), as tabulated in Demšar 2006, Table 5) are looked up for
//  k = 2..10; for larger k we fall back to the k=10 value (mildly conservative).
NemenyiResult Statistics::nemenyi(const std::vector<std::vector<double>>& data) {
    NemenyiResult res;
    const int k = static_cast<int>(data.size());
    if (k < 2) return res;
    const int N = static_cast<int>(data[0].size());
    if (N < 1) return res;
    for (int j = 1; j < k; ++j) if ((int)data[j].size() != N) return res;

    // Average ranks (lower metric = better = rank 1), with tie averaging.
    std::vector<double> R(k, 0.0);
    for (int row = 0; row < N; ++row) {
        std::vector<int> order(k); std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b){ return data[a][row] < data[b][row]; });
        int i = 0;
        while (i < k) {
            int j = i;
            while (j < k && nearlyEqual(data[order[j]][row], data[order[i]][row])) ++j;
            double avgR = (i + j + 1) / 2.0;
            for (int m = i; m < j; ++m) R[order[m]] += avgR;
            i = j;
        }
    }
    std::vector<double> avg(k);
    for (int j = 0; j < k; ++j) avg[j] = R[j] / N;

    // Studentized-range critical values q_alpha / sqrt(2) for this k, computed
    // from the infinite-df range distribution (equals Demsar 2006, Table 5, and
    // is valid for every k; the old table stopped at k = 10 and reused that
    // value for larger k, which made the critical difference too small).
    const double qA05 = nemenyiQ(k, 0.05);
    const double qA10 = nemenyiQ(k, 0.10);

    const double cdFactor = std::sqrt(static_cast<double>(k) * (k + 1) / (6.0 * N));
    res.cd05 = qA05 * cdFactor;
    res.cd10 = qA10 * cdFactor;

    // Per-pair Nemenyi p-values (family-wise adjusted): the rank gap divided by
    // its standard error is q/sqrt(2) with q the studentized range of k groups,
    // so p = P(range_k > |Ri-Rj| / se * sqrt(2)). Consistent with cd05 / cd10:
    // a gap equal to cd05 has p = 0.05. (The old code returned the unadjusted
    // two-sided normal p-value of one pair.)
    const double seGap = cdFactor;   // standard error of an average-rank gap
    res.pPair.assign(k, std::vector<double>(k, 1.0));
    for (int a = 0; a < k; ++a) {
        for (int b = a + 1; b < k; ++b) {
            const double z = std::abs(avg[a] - avg[b]) / (seGap > 1e-12 ? seGap : 1.0);
            const double p = std::min(1.0, std::max(0.0, 1.0 - rangeCdfInf(z * std::sqrt(2.0), k)));
            res.pPair[a][b] = res.pPair[b][a] = p;
        }
    }

    res.avgRanks = avg;
    res.N = N; res.k = k;
    res.valid = true;
    return res;
}

} // namespace NeuralStudio
