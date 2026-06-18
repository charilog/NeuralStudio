#include "DatasetLoader.h"

#include "CompanionFinder.h"
#include "loaders/IDataLoader.h"
#include "loaders/NeuralDesignerLoader.h"
#include "loaders/CsvLoader.h"
#include "utils/Logger.h"

#include <QFileInfo>
#include <stdexcept>

namespace NeuralStudio {

// ─── Registry ─────────────────────────────────────────────────────────────────
//  Returns all registered loaders (owned here as static locals).
static QList<IDataLoader*> allLoaders() {
    // Static storage — loaders live for the application lifetime.
    static NeuralDesignerLoader ndLoader;
    static CsvLoader             csvLoader;

    return { &ndLoader, &csvLoader };
}

// ─── detectFormat ────────────────────────────────────────────────────────────
QString DatasetLoader::detectFormat(const QString& path) {
    const QString ext = QFileInfo(path).suffix().toLower();
    for (IDataLoader* loader : allLoaders())
        if (loader->extensions().contains(ext))
            return loader->formatName();
    return {};
}

// ─── supportedExtensions ─────────────────────────────────────────────────────
QStringList DatasetLoader::supportedExtensions() {
    QStringList all;
    for (IDataLoader* loader : allLoaders())
        all << loader->extensions();
    return all;
}

// ─── fileDialogFilter ────────────────────────────────────────────────────────
QString DatasetLoader::fileDialogFilter() {
    // Build "All supported (*.train *.csv ...)" followed by per-format filters.
    QStringList allExts;
    QStringList perFormat;

    for (IDataLoader* loader : allLoaders()) {
        QStringList exts = loader->extensions();
        QStringList wildcards;
        for (const QString& e : exts) wildcards << ("*." + e);
        allExts << wildcards;
        perFormat << QString("%1 (%2)").arg(loader->formatName(), wildcards.join(' '));
    }

    QStringList parts;
    parts << QString("All supported files (%1)").arg(allExts.join(' '));
    parts << perFormat;
    parts << "All files (*)";
    return parts.join(";;");
}

// ─── load ────────────────────────────────────────────────────────────────────
std::unique_ptr<Dataset> DatasetLoader::load(const QString& path) {
    const QFileInfo fi(path);
    if (!fi.exists())
        throw std::runtime_error("File not found: " + path.toStdString());

    const QString ext = fi.suffix().toLower();
    IDataLoader* chosen = nullptr;

    // 1. Match by extension
    for (IDataLoader* loader : allLoaders()) {
        if (loader->extensions().contains(ext)) {
            chosen = loader;
            break;
        }
    }

    // 2. No extension match — probe every loader
    if (!chosen) {
        for (IDataLoader* loader : allLoaders()) {
            if (loader->canLoad(path)) {
                NS_WARN << "Extension" << ext << "unknown; probed loader:"
                        << loader->formatName();
                chosen = loader;
                break;
            }
        }
    }

    if (!chosen)
        throw std::runtime_error(
            "Unsupported file format: '" + ext.toStdString() + "'. "
            "Supported extensions: " +
            supportedExtensions().join(", ").toStdString());

    // 3. Delegate
    NS_INFO << "Loading" << fi.fileName() << "with" << chosen->formatName() << "loader";
    return chosen->load(path);
}

// ─── loadBundle ──────────────────────────────────────────────────────────────
//  Loads `path` plus its companion file if found.  If `path` is a *.test file,
//  swap them so the train dataset is always the bundle's `train` field.
DatasetBundle DatasetLoader::loadBundle(const QString& path) {
    DatasetBundle bundle;

    // Always load the requested file first — that's what the user clicked.
    auto primary = std::shared_ptr<Dataset>(load(path).release());

    // Probe for companion
    const QString companionPath = CompanionFinder::findCompanion(path);
    std::shared_ptr<Dataset> companion;

    if (!companionPath.isEmpty()) {
        try {
            companion = std::shared_ptr<Dataset>(load(companionPath).release());
            NS_INFO << "Companion file auto-loaded:" << companionPath;
        } catch (const std::exception& ex) {
            // Companion load failure is non-fatal — just log and continue
            NS_WARN << "Companion file found but failed to load:"
                    << companionPath << "—" << ex.what();
        }
    }

    // Decide which one is "train": prefer the one whose filename role is "train".
    const QString primaryRole   = CompanionFinder::roleOf(path);
    const bool    primaryIsTest = (primaryRole.compare("test", Qt::CaseInsensitive) == 0);

    if (primaryIsTest && companion) {
        // User clicked the .test file — flip them so train stays the primary
        bundle.train = companion;
        bundle.test  = primary;
    } else {
        bundle.train = primary;
        bundle.test  = companion;  // may be null
    }
    return bundle;
}

} // namespace NeuralStudio
