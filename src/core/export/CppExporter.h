#pragma once
#include "core/nn/NeuralNetwork.h"
#include <QString>

namespace NeuralStudio {

// ─── CppExporter ──────────────────────────────────────────────────────────────
//  Generates a self-contained C++ header file that embeds the trained weights.
//  The generated file has zero dependencies — just #include it and call predict().
//
//  Usage of the generated file:
//    #include "neuralstudio_model.h"
//    double inputs[30] = { ... };
//    auto result = NeuralStudioModel::predict(inputs);
// ─────────────────────────────────────────────────────────────────────────────
class CppExporter {
public:
    static void exportHeader(const NeuralNetwork& net,
                             const QString& path,
                             const QString& namespaceName = "NeuralStudioModel");

    static QString fileFilter() {
        return "C++ Header (*.h);;All files (*)";
    }
};

} // namespace NeuralStudio
