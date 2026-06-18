#pragma once

#include "IDataLoader.h"

namespace NeuralStudio {

// ─── NeuralDesignerLoader ────────────────────────────────────────────────────
//  Parses the Neural Designer binary-free text format:
//
//    Line 1 : <int>   — number of input features
//    Line 2 : <int>   — declared number of samples (used as hint only)
//    Lines 3+: space-separated doubles
//              first <inputCount> values  → inputs
//              remaining values           → outputs
//
//  Handles extensions: .train  .test  .data
// ─────────────────────────────────────────────────────────────────────────────
class NeuralDesignerLoader : public IDataLoader {
public:
    std::unique_ptr<Dataset> load(const QString& path) override;
    QStringList              extensions()  const override;
    QString                  formatName()  const override;
    bool                     canLoad(const QString& path) const override;
};

} // namespace NeuralStudio
