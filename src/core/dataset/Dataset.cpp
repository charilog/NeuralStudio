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

            const double var = (sumSq / valid) - (s.mean * s.mean);
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
