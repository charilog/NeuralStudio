#include "CompanionFinder.h"
#include <QFileInfo>
#include <QFile>
#include <utility>

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
    // "_train"/"-train" sit before the extension ("iris_train.csv"): they are matched on the
    // name without its last suffix and the suffix is put back. (Only the whole file name was
    // tried, so these pairs could never match a file that has an extension.)
    const QString suffix = fi.suffix();
    const QString stem   = suffix.isEmpty() ? QString() : filename.left(filename.length() - suffix.length() - 1);

    for (const auto& p : kPairs) {
        for (const auto& [from, to] : {std::pair{p.a, p.b}, std::pair{p.b, p.a}}) {
            QString swapped = trySwap(filename, from, to);
            if (!swapped.isEmpty() && QFile::exists(dir + "/" + swapped)) return dir + "/" + swapped;
            if (stem.isEmpty()) continue;
            swapped = trySwap(stem, from, to);
            if (!swapped.isEmpty() && QFile::exists(dir + "/" + swapped + "." + suffix))
                return dir + "/" + swapped + "." + suffix;
        }
    }
    return {};
}

// ─── roleOf ──────────────────────────────────────────────────────────────────
QString CompanionFinder::roleOf(const QString& path) {
    const QFileInfo fi(path);
    const QString name = fi.fileName().toLower();
    const QString stem = fi.suffix().isEmpty() ? name : name.left(name.length() - fi.suffix().length() - 1);
    // the first suffix of every pair is the training file, the second the test file
    // (".trn"/".tst" used to be returned as "trn"/"tst", so a clicked .tst file was
    // loaded as the TRAINING set and the .trn file as its test set); "_test" before an
    // extension ("iris_test.csv") counts as well
    for (const auto& p : kPairs) {
        if (name.endsWith(p.a) || stem.endsWith(p.a)) return "train";
        if (name.endsWith(p.b) || stem.endsWith(p.b)) return "test";
    }
    return {};
}

} // namespace NeuralStudio
