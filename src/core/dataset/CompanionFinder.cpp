#include "CompanionFinder.h"
#include <QFileInfo>
#include <QFile>

namespace NeuralStudio {

namespace {

// Pairs of (suffix, companion) — checked in order, both directions.
struct Pair { QString a; QString b; };

static const std::vector<Pair> kPairs = {
    { ".train",  ".test"  },
    { ".trn",    ".tst"   },
    { "_train",  "_test"  },
    { "-train",  "-test"  },
};

// Try to swap suffix `from` → `to` in the filename (case-insensitive).
// Returns the swapped path if `from` was found at the right position.
static QString trySwap(const QString& filename, const QString& from, const QString& to) {
    // Look for `from` either at the end of the basename (".train"/"_train")
    // or as an extension. We do a simple endsWith check on the stem.
    if (filename.endsWith(from, Qt::CaseInsensitive)) {
        const int len = filename.length() - from.length();
        return filename.left(len) + to;
    }
    return {};
}

} // anonymous namespace

// ─── findCompanion ───────────────────────────────────────────────────────────
QString CompanionFinder::findCompanion(const QString& path) {
    QFileInfo fi(path);
    const QString dir      = fi.absolutePath();
    const QString filename = fi.fileName();

    for (const auto& p : kPairs) {
        // Try a → b
        QString swapped = trySwap(filename, p.a, p.b);
        if (!swapped.isEmpty()) {
            const QString candidate = dir + "/" + swapped;
            if (QFile::exists(candidate)) return candidate;
        }
        // Try b → a
        swapped = trySwap(filename, p.b, p.a);
        if (!swapped.isEmpty()) {
            const QString candidate = dir + "/" + swapped;
            if (QFile::exists(candidate)) return candidate;
        }
    }
    return {};
}

// ─── roleOf ──────────────────────────────────────────────────────────────────
QString CompanionFinder::roleOf(const QString& path) {
    const QString name = QFileInfo(path).fileName().toLower();
    for (const auto& p : kPairs) {
        if (name.endsWith(p.a)) return p.a.mid(1).replace('_', "").replace('-', ""); // "train"
        if (name.endsWith(p.b)) return p.b.mid(1).replace('_', "").replace('-', ""); // "test"
    }
    return {};
}

} // namespace NeuralStudio
