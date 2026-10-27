#include "Dataset.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

namespace NeuralStudio {

// ─── isValid ─────────────────────────────────────────────────────────────────
bool Dataset::isValid() const {
    return inputCount  > 0
        && outputCount > 0
        && sampleCount > 0
        && static_cast<int>(inputs.size())  == sampleCount
        && static_cast<int>(outputs.size()) == sampleCount;
}

// ─── computeStatistics ───────────────────────────────────────────────────────
void Dataset::computeStatistics() {
    // Generic lambda: compute stats for one matrix
    auto compute = [](const std::vector<std::vector<double>>& data,
                      int cols,
                      std::vector<ColumnStats>& stats) {
        stats.assign(cols, ColumnStats{});
        const int n = static_cast<int>(data.size());
        if (n == 0 || cols == 0) return;

        for (int col = 0; col < cols; ++col) {
            double sum = 0.0, sumSq = 0.0;
            double minV = std::numeric_limits<double>::max();
            double maxV = std::numeric_limits<double>::lowest();
            int valid = 0;

            std::vector<double> vals;
            vals.reserve(n);

            for (int row = 0; row < n; ++row) {
                const double v = data[row][col];
                if (std::isnan(v) || std::isinf(v)) {
                    ++stats[col].missing;
                    continue;
                }
                sum   += v;
                sumSq += v * v;
                minV = std::min(minV, v);
                maxV = std::max(maxV, v);
                vals.push_back(v);
                ++valid;
            }

            if (valid == 0) continue;

            ColumnStats& s = stats[col];
            s.min  = minV;
            s.max  = maxV;
            s.mean = sum / valid;

            // two-pass variance (E[x^2] - E[x]^2 cancels catastrophically for large values
            // with a small spread, e.g. years or prices, and could give 0 or garbage)
            double ss = 0.0;
            for (double v : vals) { const double d = v - s.mean; ss += d * d; }
            const double var = ss / valid;
            (void)sumSq;
            s.stddev = (var > 0.0) ? std::sqrt(var) : 0.0;

            // median
            std::sort(vals.begin(), vals.end());
            if (valid % 2 == 0)
                s.median = 0.5 * (vals[valid / 2 - 1] + vals[valid / 2]);
            else
                s.median = vals[valid / 2];
        }
    };

    compute(inputs,  inputCount,  inputStats);
    compute(outputs, outputCount, outputStats);
}

// ─── dropRowsWithMissingTargets ───────────────────────────────────────────────
int Dataset::dropRowsWithMissingTargets() {
    int dropped = 0;
    std::vector<std::vector<double>> in, out;
    in.reserve(inputs.size()); out.reserve(outputs.size());
    for (size_t r = 0; r < outputs.size() && r < inputs.size(); ++r) {
        bool ok = true;
        for (double v : outputs[r]) if (!std::isfinite(v)) { ok = false; break; }
        if (ok) { in.push_back(std::move(inputs[r])); out.push_back(std::move(outputs[r])); }
        else ++dropped;
    }
    inputs.swap(in); outputs.swap(out);
    sampleCount = static_cast<int>(inputs.size());
    return dropped;
}

// ─── summary ─────────────────────────────────────────────────────────────────
QString Dataset::summary() const {
    return QString("[%1] %2 samples | %3 inputs | %4 outputs | format: %5")
        .arg(name)
        .arg(sampleCount)
        .arg(inputCount)
        .arg(outputCount)
        .arg(sourceFormat);
}

// ─── Row helpers ─────────────────────────────────────────────────────────────
std::vector<double> Dataset::inputRow(int i) const {
    if (i < 0 || i >= sampleCount) return {};
    return inputs[i];
}

std::vector<double> Dataset::outputRow(int i) const {
    if (i < 0 || i >= sampleCount) return {};
    return outputs[i];
}

// ─── uniqueOutputValues ──────────────────────────────────────────────────────
std::vector<double> Dataset::uniqueOutputValues(int outputIdx) const {
    if (outputIdx < 0 || outputIdx >= outputCount) return {};
    std::set<double> seen;
    for (int r = 0; r < sampleCount; ++r)
        seen.insert(outputs[r][outputIdx]);
    return std::vector<double>(seen.begin(), seen.end());
}

} // namespace NeuralStudio
