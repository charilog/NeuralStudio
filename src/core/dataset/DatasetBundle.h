#pragma once
#include "Dataset.h"
#include <memory>

namespace NeuralStudio {

// ─── DatasetBundle ────────────────────────────────────────────────────────────
//  Container that pairs a training dataset with an optional companion test
//  dataset. When the user loads a *.train file, the bundle is created with
//  test = auto-loaded *.test file, and vice versa.
//
//  Throughout the app `train` is the "primary" dataset (the one shown in the
//  Data preview and used to fit normalisation stats); `test` is held-out.
// ─────────────────────────────────────────────────────────────────────────────
struct DatasetBundle {
    std::shared_ptr<Dataset> train;   // never null after a successful load
    std::shared_ptr<Dataset> test;    // may be null if no companion was found

    bool hasTest() const { return test != nullptr; }
};

} // namespace NeuralStudio
