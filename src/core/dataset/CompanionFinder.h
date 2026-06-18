#pragma once
#include <QString>
#include <QStringList>

namespace NeuralStudio {

// ─── CompanionFinder ──────────────────────────────────────────────────────────
//  Given a path like "wine.train", returns the matching companion "wine.test"
//  if it exists in the same directory (or vice-versa).
//
//  Recognised pairs:
//    .train  ↔  .test
//    .trn    ↔  .tst
//    _train  ↔  _test    (e.g. "data_train.csv" ↔ "data_test.csv")
//    -train  ↔  -test
// ─────────────────────────────────────────────────────────────────────────────
class CompanionFinder {
public:
    // Returns absolute path of the companion file, or empty string if none.
    static QString findCompanion(const QString& path);

    // Returns a human-readable label like "train" / "test" / "" if unknown.
    static QString roleOf(const QString& path);
};

} // namespace NeuralStudio
