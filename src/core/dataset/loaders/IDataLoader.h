#pragma once

#include "core/dataset/Dataset.h"
#include <QString>
#include <QStringList>
#include <memory>
#include <stdexcept>

namespace NeuralStudio {

// ─── IDataLoader ──────────────────────────────────────────────────────────────
//  Abstract base for all file format loaders.
//  Each loader handles one or more file extensions and knows how to parse
//  its format into a canonical Dataset.
//
//  To add a new format:
//    1. Create a class that inherits IDataLoader
//    2. Implement load(), extensions(), formatName()
//    3. Register it in DatasetLoader::createLoader()
// ─────────────────────────────────────────────────────────────────────────────
class IDataLoader {
public:
    virtual ~IDataLoader() = default;

    // Parse the file at `path` and return a fully populated Dataset.
    // Throws std::runtime_error with a descriptive message on failure.
    virtual std::unique_ptr<Dataset> load(const QString& path) = 0;

    // Lowercase file extensions handled by this loader (without leading dot).
    // e.g. {"train", "test", "data"}
    virtual QStringList extensions() const = 0;

    // Human-readable format name shown in the UI file filter.
    // e.g. "Neural Designer", "CSV", "JSON"
    virtual QString formatName() const = 0;

    // Optional: peek at the file and confirm this loader can handle it.
    // Default implementation returns true (rely on extension matching).
    virtual bool canLoad(const QString& /*path*/) const { return true; }
};

} // namespace NeuralStudio
