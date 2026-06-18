#pragma once
#include <vector>
#include <string>
#include <map>

namespace NeuralStudio {

// ─── WilcoxonResult ──────────────────────────────────────────────────────────
struct WilcoxonResult {
    double wPlus  = 0;   // sum of positive ranks
    double wMinus = 0;   // sum of negative ranks
    double z      = 0;   // z-statistic (normal approx)
    double pValue = 1;   // two-tailed p-value
    int    n      = 0;   // non-zero differences used
    bool   valid  = false;

    // Significance stars: "***"/<0.001  "**"/<0.01  "*"/<0.05  "ns"
    std::string stars() const;
};

// ─── FriedmanResult ──────────────────────────────────────────────────────────
struct FriedmanResult {
    double             chi2     = 0;
    int                df       = 0;   // k-1
    double             pValue   = 1;
    std::vector<double>avgRanks;        // per optimizer (same order as names)
    std::vector<int>   rankOrder;       // indices sorted best→worst
    int                N        = 0;   // datasets used
    int                k        = 0;   // optimizers
    bool               valid    = false;

    std::string stars() const;
};

// ─── Statistics ──────────────────────────────────────────────────────────────
class Statistics {
public:
    // Wilcoxon signed-rank test: pairwise comparison of optimizer A vs B
    // across the same set of datasets.  Lower metric = better.
    static WilcoxonResult wilcoxon(const std::vector<double>& a,
                                    const std::vector<double>& b);

    // Friedman test: k optimizers × N datasets complete block design.
    // data[optimizer][dataset] = metric value; lower is better.
    static FriedmanResult friedman(
        const std::vector<std::string>& names,
        const std::vector<std::vector<double>>& data);  // [optimizer][dataset]

    // ── Distribution helpers ─────────────────────────────────────────────────
    // Standard normal CDF Φ(x)
    static double normalCDF(double x);

    // Chi-squared right-tail probability P(X > chi2 | df)
    // Uses Wilson-Hilferty normal approximation (accurate for df ≥ 2)
    static double chi2pvalue(double chi2, int df);

    // Compute boxplot statistics from a sample
    struct BoxStats {
        double q0, q1, q2, q3, q4;   // min-whisker, Q1, median, Q3, max-whisker
        std::vector<double> outliers;
        double mean;
        bool   valid = false;
    };
    static BoxStats boxStats(std::vector<double> v);   // copy intentional
};

} // namespace NeuralStudio
