#pragma once

#include "Dataset.h"
#include "DatasetBundle.h"
#include <QString>
#include <QStringList>
#include <memory>

namespace NeuralStudio {

// ─── DatasetLoader ────────────────────────────────────────────────────────────
//  Central factory.  Call DatasetLoader::load(path) and it will:
//    1. Match the file extension to a registered loader.
//    2. Ask each candidate loader if it canLoad() the file (optional probe).
//    3. Parse and return a populated Dataset.
//
//  Or call loadBundle(path) to also auto-load the companion train/test file.
//
//  Adding a new format:
//    Add its extension + loader instantiation in createLoader().
// ─────────────────────────────────────────────────────────────────────────────
class DatasetLoader {
public:
    // Main entry point — throws std::runtime_error on failure.
    static std::unique_ptr<Dataset> load(const QString& path);

    // Loads `path` as the primary (train) dataset and also probes for a
    // companion .test/.train file in the same directory.
    // If the user clicked a .test file, the train file becomes primary
    // (and the clicked file goes into bundle.test).
    static DatasetBundle loadBundle(const QString& path);

    // Human-readable format filter string for QFileDialog::getOpenFileName.
    static QString fileDialogFilter();

    // All extensions we support (lowercase, no dot).
    static QStringList supportedExtensions();

    // Detect format string from path (returns "" if unknown).
    static QString detectFormat(const QString& path);

private:
    static class IDataLoader* loaderForExtension(const QString& ext,
                                                  QList<class IDataLoader*>& pool);
};

} // namespace NeuralStudio
