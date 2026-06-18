#pragma once

#include <QString>
#include <QStringList>
#include <vector>
#include <limits>

namespace NeuralStudio {

// ─── Variable role for column selection (Phase 4) ────────────────────────────
enum class VariableRole { Input, Target, Ignored };

// ─── Per-column descriptive statistics ────────────────────────────────────────
struct ColumnStats {
    double min     = 0.0;
    double max     = 0.0;
    double mean    = 0.0;
    double stddev  = 0.0;
    double median  = 0.0;
    int    missing = 0;   // NaN or Inf count
};

// ─── Dataset ──────────────────────────────────────────────────────────────────
//  Canonical in-memory representation of a loaded dataset.
//  Owns both the input matrix and the output matrix.
//  Row-major storage: data[sample][feature].
// ─────────────────────────────────────────────────────────────────────────────
class Dataset {
public:
    Dataset() = default;

    // ── Provenance ──────────────────────────────────────────────────────────
    QString name;          // display name (basename without extension)
    QString sourcePath;    // absolute path of the original file
    QString sourceFormat;  // "neuraldesigner" | "csv" | "json" | "excel"

    // ── Structure ────────────────────────────────────────────────────────────
    int inputCount  = 0;
    int outputCount = 0;
    int sampleCount = 0;

    // ── Column names ─────────────────────────────────────────────────────────
    QStringList inputNames;   // length == inputCount
    QStringList outputNames;  // length == outputCount

    // ── Data matrices ────────────────────────────────────────────────────────
    std::vector<std::vector<double>> inputs;   // [sampleCount][inputCount]
    std::vector<std::vector<double>> outputs;  // [sampleCount][outputCount]

    // ── Statistics (populated by computeStatistics()) ────────────────────────
    std::vector<ColumnStats> inputStats;
    std::vector<ColumnStats> outputStats;

    // ── API ──────────────────────────────────────────────────────────────────
    void    computeStatistics();
    bool    isValid()   const;
    QString summary()   const;

    // Element access (bounds-checked in debug, raw in release)
    double inputAt (int sample, int feature) const { return inputs [sample][feature]; }
    double outputAt(int sample, int out)     const { return outputs[sample][out];     }

    // Row access (returns empty vector on bad index)
    std::vector<double> inputRow (int i) const;
    std::vector<double> outputRow(int i) const;

    // Unique output values — useful for classification tasks
    std::vector<double> uniqueOutputValues(int outputIdx = 0) const;
};

} // namespace NeuralStudio
