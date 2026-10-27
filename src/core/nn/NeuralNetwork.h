#pragma once
#include "Layer.h"
#include "LossFunction.h"
#include "core/dataset/Dataset.h"
#include <vector>
#include <QString>

namespace NeuralStudio {

enum class TaskType { Regression, BinaryClassification, MultiClassClassification };

struct NetworkConfig {
    std::vector<int>        hiddenSizes;
    std::vector<Activation> hiddenActivations;
    TaskType task = TaskType::BinaryClassification;
    double   dropoutRate = 0.0;  // applied to all hidden layers (0 = disabled)
};

class NeuralNetwork {
public:
    NeuralNetwork() = default;

    void build(int inputSize, int outputSize, const NetworkConfig& cfg);

    // Toggle training mode on every layer (controls dropout)
    void setTrainingMode(bool train);

    // ── Inference ─────────────────────────────────────────────────────────────
    std::vector<double> predict(const std::vector<double>& rawInput);
    std::vector<double> predictNorm(const std::vector<double>& normInput);

    // ── Normalization ─────────────────────────────────────────────────────────
    void setNormalization(const std::vector<ColumnStats>& stats);
    std::vector<double> normalize(const std::vector<double>& raw) const;
    const std::vector<double>& normMin()   const { return m_normMin;   }
    const std::vector<double>& normRange() const { return m_normRange; }
    const std::vector<double>& normFill()  const { return m_normFill;  }

    // ── Regression output scaling ─────────────────────────────────────────────
    // A regression network computes  y = shift + scale * (linear output layer),
    // with shift / scale = mean / standard deviation of the TRAIN targets. The
    // hidden layers and the weight bounds therefore work on a standardised
    // target (as every reference MLP does) while every loss, prediction and
    // reported error stays on the original scale of the data. Classification
    // networks and models without scaling use the identity (shift 0, scale 1).
    void fitOutputScaling(const Dataset* trainDs);       // no-op unless Regression
    void setOutputScaling(const std::vector<double>& shift, const std::vector<double>& scale);
    bool hasOutputScaling() const { return !m_outScale.empty(); }
    double outShift(int o) const { return o < static_cast<int>(m_outShift.size()) ? m_outShift[o] : 0.0; }
    double outScale(int o) const { return o < static_cast<int>(m_outScale.size()) ? m_outScale[o] : 1.0; }
    const std::vector<double>& outShifts() const { return m_outShift; }
    const std::vector<double>& outScales() const { return m_outScale; }
    // in place: raw output-layer values -> predictions on the data scale
    void applyOutputScaling(std::vector<double>& x) const {
        if (m_task != TaskType::Regression || m_outScale.empty()) return;
        for (size_t o = 0; o < x.size() && o < m_outScale.size(); ++o) x[o] = m_outShift[o] + m_outScale[o] * x[o];
    }

    // ── Classification helpers ────────────────────────────────────────────────
    void   setOutputMapping(double minVal, double maxVal);
    void   setClassValues(const std::vector<double>& vals);
    int    classIndex(double rawVal) const;          // raw output → class index
    std::vector<double> oneHot(int classIdx) const;  // class index → one-hot
    const std::vector<double>& classValues() const { return m_classValues; }

    double outputMin() const { return m_outMin; }
    double outputMax() const { return m_outMax; }

    // ── Accessors ─────────────────────────────────────────────────────────────
    std::vector<Layer>& layers()     { return m_layers; }
    int       inputSize()  const     { return m_inputSize;  }
    int       outputSize() const     { return m_outputSize; }
    int       numClasses() const     { return static_cast<int>(m_classValues.size()); }
    TaskType  taskType()   const     { return m_task; }
    LossType  lossType()   const     { return m_loss; }
    int       totalParams() const;
    QString   summary()    const;
    bool      isBuilt()    const     { return !m_layers.empty(); }

private:
    std::vector<Layer>  m_layers;
    int      m_inputSize  = 0;
    int      m_outputSize = 0;
    TaskType m_task = TaskType::BinaryClassification;
    LossType m_loss = LossType::BinaryCrossEntropy;

    std::vector<double> m_normMin;
    std::vector<double> m_normRange;
    std::vector<double> m_normFill;    // normalised train mean: replaces missing (NaN/inf) inputs
    bool m_hasNorm = false;

    std::vector<double> m_outShift, m_outScale;   // regression output scaling (empty = identity)

    std::vector<double> m_classValues; // sorted unique raw output values
    double m_outMin = 0.0;
    double m_outMax = 1.0;
};

// ── Free helpers ──────────────────────────────────────────────────────────────
TaskType detectTask(const Dataset* ds);

} // namespace NeuralStudio
