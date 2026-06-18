#pragma once
#include "NeuralNetwork.h"
#include <QString>

namespace NeuralStudio {

// ─── ModelSerializer ──────────────────────────────────────────────────────────
//  Saves/loads a trained NeuralNetwork to/from a JSON file (.nsmodel).
//
//  Format: plain QJsonDocument — no external dependencies.
// ─────────────────────────────────────────────────────────────────────────────
class ModelSerializer {
public:
    // Throws std::runtime_error on failure
    static void save(const NeuralNetwork& net, const QString& path);
    static void load(NeuralNetwork& net,       const QString& path);

    // File dialog filter
    static QString fileFilter() {
        return "NeuralStudio Model (*.nsmodel);;All files (*)";
    }
};

} // namespace NeuralStudio
