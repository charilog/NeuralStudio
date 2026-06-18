#include "CppExporter.h"
#include "utils/Logger.h"

#include <QFile>
#include <QFileInfo>
#include <QTextStream>
#include <QDateTime>
#include <stdexcept>
#include <cmath>

namespace NeuralStudio {

void CppExporter::exportHeader(const NeuralNetwork& net,
                               const QString& path,
                               const QString& nsName) {
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Text))
        throw std::runtime_error("Cannot write export file: " + path.toStdString());

    QTextStream out(&f);
    out.setEncoding(QStringConverter::Utf8);

    const auto& layers = const_cast<NeuralNetwork&>(net).layers();
    const int   nIn    = net.inputSize();
    const int   nOut   = net.outputSize();
    const TaskType task = net.taskType();

    auto taskStr = [&]() -> QString {
        switch(task) {
            case TaskType::BinaryClassification:     return "BinaryClassification";
            case TaskType::MultiClassClassification: return "MultiClassClassification";
            default:                                 return "Regression";
        }
    };

    // ── File header ──────────────────────────────────────────────────────────
    out << "// ============================================================\n";
    out << "// NeuralStudio — Exported Model\n";
    out << "// Generated: " << QDateTime::currentDateTime().toString(Qt::ISODate) << "\n";
    out << "// Task:       " << taskStr() << "\n";
    out << "// Inputs:     " << nIn  << "\n";
    out << "// Outputs:    " << nOut << "\n";
    out << "// Params:     " << net.totalParams() << "\n";
    out << "// Architecture: " << net.summary() << "\n";
    out << "//\n";
    out << "// Usage:\n";
    out << "//   #include \"" << QFileInfo(path).fileName() << "\"\n";
    out << "//   double inputs[" << nIn << "] = { ... };\n";
    if (task == TaskType::BinaryClassification)
        out << "//   double prob = " << nsName << "::predict(inputs); // 0..1\n";
    else
        out << "//   auto probs = " << nsName << "::predict(inputs); // vector of size " << nOut << "\n";
    out << "// ============================================================\n\n";

    out << "#pragma once\n";
    out << "#include <cmath>\n";
    out << "#include <array>\n";
    out << "#include <algorithm>\n\n";

    out << "namespace " << nsName << " {\n\n";

    // ── Normalization parameters ──────────────────────────────────────────────
    out << "// Min-max normalization parameters\n";
    out << "static constexpr double kNormMin[" << nIn << "] = {";
    for (int i = 0; i < nIn; ++i)
        out << (i ? ", " : "") << net.normMin()[i];
    out << "};\n";

    out << "static constexpr double kNormRange[" << nIn << "] = {";
    for (int i = 0; i < nIn; ++i)
        out << (i ? ", " : "") << net.normRange()[i];
    out << "};\n\n";

    // ── Layer weights ─────────────────────────────────────────────────────────
    for (int li = 0; li < static_cast<int>(layers.size()); ++li) {
        const auto& layer = layers[li];
        const int   in    = layer.inputSize();
        const int   outN  = layer.outputSize();

        out << "// Layer " << li << ": " << in << " → " << outN
            << "  (" << activationName(layer.activation()) << ")\n";

        // Biases
        out << "static constexpr double kB" << li << "[" << outN << "] = {";
        for (int o = 0; o < outN; ++o)
            out << (o ? ", " : "") << layer.biases[o];
        out << "};\n";

        // Weights (row-major: [output][input])
        out << "static constexpr double kW" << li << "[" << outN << "][" << in << "] = {\n";
        for (int o = 0; o < outN; ++o) {
            out << "  {";
            for (int i = 0; i < in; ++i)
                out << (i ? ", " : "") << layer.weights[o][i];
            out << (o < outN-1 ? "},\n" : "}\n");
        }
        out << "};\n\n";
    }

    // ── normalize() helper ────────────────────────────────────────────────────
    out << "// Normalize inputs to [0, 1] using training dataset statistics\n";
    out << "inline void normalize(const double* raw, double* norm) {\n";
    out << "  for (int i = 0; i < " << nIn << "; ++i)\n";
    out << "    norm[i] = (raw[i] - kNormMin[i]) / kNormRange[i];\n";
    out << "}\n\n";

    // ── Activation lambdas ────────────────────────────────────────────────────
    out << "inline double applyAct(int act, double z) {\n";
    out << "  switch(act) {\n";
    out << "    case 1: return z > 0.0 ? z : 0.0;                  // ReLU\n";
    out << "    case 2: return 1.0 / (1.0 + std::exp(-z));          // Sigmoid\n";
    out << "    case 3: return std::tanh(z);                         // Tanh\n";
    out << "    default: return z;                                   // Linear\n";
    out << "  }\n}\n\n";

    // ── predict() ────────────────────────────────────────────────────────────
    if (task == TaskType::BinaryClassification) {
        out << "// Returns probability of class 1 (0.0 to 1.0)\n";
        out << "inline double predict(const double* rawInputs) {\n";
    } else {
        out << "// Returns output probabilities (size " << nOut << ")\n";
        out << "inline std::array<double," << nOut << "> predict(const double* rawInputs) {\n";
    }

    out << "  double x[" << nIn << "];\n";
    out << "  normalize(rawInputs, x);\n\n";

    // Generate layer-by-layer computation
    int prevSize = nIn;
    for (int li = 0; li < static_cast<int>(layers.size()); ++li) {
        const auto& layer = layers[li];
        const int   outN  = layer.outputSize();
        const int   act   = static_cast<int>(layer.activation());

        out << "  // Layer " << li << "\n";
        out << "  double y" << li << "[" << outN << "];\n";
        out << "  for (int o = 0; o < " << outN << "; ++o) {\n";
        out << "    double z = kB" << li << "[o];\n";
        out << "    for (int i = 0; i < " << prevSize << "; ++i) z += kW" << li << "[o][i] * x[i];\n";
        out << "    y" << li << "[o] = applyAct(" << act << ", z);\n";
        out << "  }\n";

        if (li < static_cast<int>(layers.size()) - 1) {
            out << "  for (int i = 0; i < " << outN << "; ++i) x[i] = y" << li << "[i];\n\n";
        }
        prevSize = outN;
    }

    const int lastIdx = static_cast<int>(layers.size()) - 1;

    if (task == TaskType::MultiClassClassification) {
        // Apply softmax
        out << "\n  // Softmax\n";
        out << "  double maxV = *std::max_element(y" << lastIdx << ", y" << lastIdx << "+" << nOut << ");\n";
        out << "  double sum = 0.0;\n";
        out << "  std::array<double," << nOut << "> probs;\n";
        out << "  for (int i = 0; i < " << nOut << "; ++i) { probs[i] = std::exp(y"
            << lastIdx << "[i] - maxV); sum += probs[i]; }\n";
        out << "  for (auto& p : probs) p /= sum;\n";
        out << "  return probs;\n";
    } else if (task == TaskType::BinaryClassification) {
        out << "  return y" << lastIdx << "[0];\n";
    } else {
        out << "  std::array<double," << nOut << "> result;\n";
        out << "  for (int i = 0; i < " << nOut << "; ++i) result[i] = y" << lastIdx << "[i];\n";
        out << "  return result;\n";
    }

    out << "}\n\n";
    out << "} // namespace " << nsName << "\n";

    NS_INFO << "C++ model exported to" << path;
}

} // namespace NeuralStudio
