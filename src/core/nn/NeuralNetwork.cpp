#include "NeuralNetwork.h"
#include <algorithm>
#include <stdexcept>
#include <set>

namespace NeuralStudio {

void NeuralNetwork::build(int inputSize, int outputSize, const NetworkConfig& cfg) {
    m_inputSize  = inputSize;
    m_outputSize = outputSize;
    m_task       = cfg.task;
    switch (cfg.task) {
        case TaskType::BinaryClassification:      m_loss = LossType::BinaryCrossEntropy;      break;
        case TaskType::MultiClassClassification:  m_loss = LossType::CategoricalCrossEntropy;  break;
        default:                                  m_loss = LossType::MSE;                      break;
    }
    m_layers.clear();

    int prev = inputSize;
    for (int i = 0; i < static_cast<int>(cfg.hiddenSizes.size()); ++i) {
        Activation act = (i < static_cast<int>(cfg.hiddenActivations.size()))
                         ? cfg.hiddenActivations[i] : Activation::ReLU;
        m_layers.emplace_back(prev, cfg.hiddenSizes[i], act);
        prev = cfg.hiddenSizes[i];
    }

    // Output activation
    Activation outAct = (cfg.task == TaskType::BinaryClassification)
                        ? Activation::Sigmoid : Activation::Linear;
    m_layers.emplace_back(prev, outputSize, outAct);

    // Apply dropout to hidden layers only (never the output layer)
    if (cfg.dropoutRate > 0.0 && cfg.dropoutRate < 1.0) {
        for (int i = 0; i < static_cast<int>(m_layers.size()) - 1; ++i)
            m_layers[i].setDropoutRate(cfg.dropoutRate);
    }
}

void NeuralNetwork::setTrainingMode(bool train) {
    for (auto& l : m_layers) l.setTrainingMode(train);
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& rawInput) {
    return predictNorm(m_hasNorm ? normalize(rawInput) : rawInput);
}

std::vector<double> NeuralNetwork::predictNorm(const std::vector<double>& normInput) {
    std::vector<double> x = normInput;
    for (auto& layer : m_layers) x = layer.forward(x);
    // For multi-class: apply softmax to convert logits → probabilities
    if (m_task == TaskType::MultiClassClassification)
        x = softmax(x);
    return x;
}

void NeuralNetwork::setNormalization(const std::vector<ColumnStats>& stats) {
    m_normMin  .resize(stats.size());
    m_normRange.resize(stats.size());
    for (int i = 0; i < static_cast<int>(stats.size()); ++i) {
        m_normMin[i]   = stats[i].min;
        m_normRange[i] = stats[i].max - stats[i].min;
        if (m_normRange[i] < 1e-10) m_normRange[i] = 1.0;
    }
    m_hasNorm = true;
}

std::vector<double> NeuralNetwork::normalize(const std::vector<double>& raw) const {
    std::vector<double> out(raw.size());
    for (int i = 0; i < static_cast<int>(raw.size()); ++i)
        out[i] = (raw[i] - m_normMin[i]) / m_normRange[i];
    return out;
}

void NeuralNetwork::setOutputMapping(double minVal, double maxVal) {
    m_outMin = minVal;
    m_outMax = maxVal;
}

void NeuralNetwork::setClassValues(const std::vector<double>& vals) {
    m_classValues = vals;
    std::sort(m_classValues.begin(), m_classValues.end());
}

int NeuralNetwork::classIndex(double rawVal) const {
    if (m_classValues.empty()) return 0;
    // Find nearest class value
    int best = 0;
    double bestDist = std::abs(rawVal - m_classValues[0]);
    for (int i = 1; i < static_cast<int>(m_classValues.size()); ++i) {
        double d = std::abs(rawVal - m_classValues[i]);
        if (d < bestDist) { bestDist = d; best = i; }
    }
    return best;
}

std::vector<double> NeuralNetwork::oneHot(int classIdx) const {
    std::vector<double> v(m_classValues.size(), 0.0);
    if (classIdx >= 0 && classIdx < static_cast<int>(v.size()))
        v[classIdx] = 1.0;
    return v;
}

int NeuralNetwork::totalParams() const {
    int total = 0;
    for (const auto& l : m_layers) total += l.paramCount();
    return total;
}

QString NeuralNetwork::summary() const {
    if (m_layers.empty()) return "(not built)";
    QString s = QString("Input(%1)").arg(m_inputSize);
    for (const auto& l : m_layers)
        s += QString(" \u2192 Dense(%1, %2)").arg(l.outputSize()).arg(activationName(l.activation()));
    s += QString("  |  %1 params").arg(totalParams());
    return s;
}

TaskType detectTask(const Dataset* ds) {
    if (!ds) return TaskType::Regression;
    auto unique = ds->uniqueOutputValues(0);
    if (unique.size() == 2)  return TaskType::BinaryClassification;
    if (unique.size() <= 20) {
        // Check if values look like discrete classes (multiples of 0.5 or integers)
        bool allDiscrete = true;
        for (double v : unique) {
            double frac = v - std::floor(v);
            if (frac > 0.01 && std::abs(frac - 0.5) > 0.01) { allDiscrete = false; break; }
        }
        if (allDiscrete) return TaskType::MultiClassClassification;
    }
    return TaskType::Regression;
}

} // namespace NeuralStudio
