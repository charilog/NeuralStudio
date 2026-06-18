# NeuralStudio

**NeuralStudio** is an open-source, cross-platform no-code workbench for designing, training, evaluating, and deploying neural network models — built with **C++20** and **Qt6**.

> No Python. No browser. No setup scripts. Just load a dataset, configure a network, and train.

---

## Platforms

| Linux | Windows | macOS |
|-------|---------|-------|
| ✓ | ✓ | ✓ |

**Build:** CMake ≥ 3.20 · Qt6 (Widgets, Charts, Gui, Core, Svg) · C++20 compiler · Eigen3 (optional)

---

## Network Architectures

### Multilayer Perceptron (MLP)
- Arbitrary depth and width (hidden layers configured as comma-separated list, e.g. `64,32,16`)
- Per-layer activation: **Linear, ReLU, Sigmoid, Tanh**
- Global dropout regularisation (applied to all hidden layers; disabled at inference)
- Output layer and loss function derived automatically from task type:
  - Regression → Linear + MSE
  - Binary Classification → Sigmoid + Binary Cross-Entropy
  - Multi-class Classification → Softmax + Categorical Cross-Entropy

### Radial Basis Function Network (RBF)
- Kernels: **Gaussian, Multiquadric, Inverse Multiquadric**
- Centre placement: **k-means++** with configurable Lloyd iterations
- Width strategies: **MaxDist, Nearest Neighbour, Mean Nearest Neighbour** (per-centre or global)
- Output weights: ridge regression solved analytically via **Cholesky decomposition**
- Training is fully analytic — no gradient computation, completes in seconds

---

## Training Methods (23 total)

### Gradient-Based (6)
| Optimiser | Notes |
|-----------|-------|
| **Adam** | Adaptive moment estimation |
| **SGD** | Stochastic Gradient Descent |
| **AdamW** | Adam with decoupled weight decay |
| **RMSProp** | Root mean square propagation |
| **Nadam** | Nesterov + Adam |
| **AdaGrad** | Adaptive subgradient |

**Learning-rate schedules:** Constant · Step Decay · Exponential Decay

### Quasi-Newton & Classical Local Search (4)
`L-BFGS` · `BFGS` · `Conjugate Gradient (Polak-Ribière)` · `Levenberg-Marquardt`

### Single-Point Derivative-Free (2)
`Simulated Annealing` · `Nelder-Mead Simplex`

### Population-Based Evolutionary & Swarm (11)
`GA` · `DE` · `PSO` · `CLPSO` · `ACOR` · `CMA-ES` · `LM-CMA-ES` · `jSO` · `mLSHADE-RL` · `ARQ3` · `PBIL` · `UMDA`

All metaheuristic methods use a configurable **weight bound** as the symmetric search-space bound for the network weight vector.

---

## Data Ingestion

| Format | Detection |
|--------|-----------|
| `.csv` `.tsv` `.txt` | Auto-detects delimiter (`,` `;` `\t`) and header presence |
| `.train` `.test` `.data` | Neural Designer binary format (header: num\_inputs / num\_samples) |

- **Drag-and-drop** supported
- **Companion file auto-load**: loading `dataset.train` automatically locates and loads `dataset.test` (and vice versa)
- **Automatic task detection**: Regression / Binary Classification / Multi-class Classification inferred from output column structure
- **Class-imbalance diagnostic** reported in-panel for classification tasks
- **Data preview**: first 100 rows in a scrollable table
- **Per-column statistics**: min, max, mean, standard deviation, median

---

## Workflow Panels (Single-Run Mode)

```
① Data  →  ② Network  →  ③ Training  →  ④ Evaluation  →  ⑤ Predict
```

Each stage is unlocked progressively as prerequisites are satisfied.

| Panel | Key features |
|-------|-------------|
| **Data** | Load, preview, statistics, companion file auto-detection |
| **Network** | MLP / RBF architecture builder, automatic input/output sizing |
| **Training** | All 23 optimisers, LR scheduling, early stopping, K-fold CV, live loss/accuracy charts |
| **Evaluation** | Accuracy, Precision, Recall, F1, Confusion Matrix (classification) · MAE, RMSE, R² (regression) · per-sample prediction table |
| **Predict** | Manual inference on individual samples, Load 1st Sample / Reset to Means shortcuts |

---

## Batch Run Mode

Automates training of **all selected optimiser × dataset combinations** in a single run.

- **Shared config**: one architecture and training configuration applies to all jobs
- **Optimiser matrix**: select any subset of the 23 methods with All / None / Gradient only / Evolutionary only shortcuts
- **Queue**: unlimited datasets via Add Files or drag-and-drop
- **Live progress**: job counter, elapsed time, ETA, incremental results table
- **Results table**: one row per (dataset × optimiser) — task, N, inputs, classes, epochs, time(s), Train Err%, Val Err%, Test Err%, MAE, RMSE, R²
- **Export XLSX**: full results table exported to a formatted spreadsheet

---

## Statistical Analysis (Batch Mode)

### Wilcoxon Signed-Rank Tab
- Pairwise two-tailed Wilcoxon signed-rank test for all optimiser pairs
- Metric selector: Val Err%, Train Err%, Test Err%, MAE, RMSE, R²
- Task filter: All / Regression / Binary / Multi-class
- Box plots with significance brackets and p-value annotations
- Full pairwise p-value matrix

### Friedman Test Tab
- Global Friedman χ² test across all optimisers and datasets
- Post-hoc average-rank table (best → worst, best highlighted green, worst in red)
- Bonferroni-corrected pairwise Wilcoxon p-value matrix
- Export PNG at **300 DPI** (publication-ready)

---

## Validation

- **K-fold Cross-Validation** (configurable K) integrated directly in the Training panel
- Runs K independent models on disjoint folds; reports mean ± std of loss and accuracy

---

## Model Export

| Format | Description |
|--------|-------------|
| **C++ header** (`.h`) | Self-contained, zero-dependency header embedding all weights and normalisation parameters. `#include` and call `predict()` — no external libraries required at inference time. |
| **CSV training log** | Per-epoch loss, accuracy, and learning rate exported from the Training panel |
| **XLSX results** | Full Batch Run results table with all metrics |
| **PNG figures** | Wilcoxon and Friedman plots at 300 DPI |

---

## Build

### Windows (PowerShell)
```powershell
# 1. Clean previous build (optional)
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue

# 2. Configure
cmake -S . -B build `
   "-DCMAKE_PROJECT_INCLUDE:FILEPATH=$PWD/cmake/neuralstudio.cmake" `
   "-DCMAKE_PREFIX_PATH=C:\Qt\6.10.1\msvc2022_64"

# 3. Build
cmake -B build && cmake --build build --config Release

# 4. Deploy Qt runtime dependencies
& "C:\Qt\6.10.1\msvc2022_64\bin\windeployqt.exe" `
   --release --no-translations --compiler-runtime `
   ".\build\Release\neuralstudio.exe"

# 5. Run
.\build\Release\neuralstudio.exe
```

> Adjust `C:\Qt\6.10.1\msvc2022_64` to match your Qt installation path.

---

### Linux
```bash
# 1. Clean previous build (optional)
rm -rf build

# 2. Configure
cmake -S . -B build \
   "-DCMAKE_PROJECT_INCLUDE:FILEPATH=$PWD/cmake/neuralstudio.cmake" \
   "-DCMAKE_PREFIX_PATH=/path/to/Qt/6.10.1/gcc_64"

# 3. Build
cmake --build build --parallel $(nproc)

# 4. Run
./build/neuralstudio
```

> No deployment step needed for local use. For distribution, use
> [linuxdeployqt](https://github.com/probonopd/linuxdeployqt) or AppImage tooling.

---

### macOS
```bash
# 1. Clean previous build (optional)
rm -rf build

# 2. Configure
cmake -S . -B build \
   "-DCMAKE_PROJECT_INCLUDE:FILEPATH=$PWD/cmake/neuralstudio.cmake" \
   "-DCMAKE_PREFIX_PATH=/path/to/Qt/6.10.1/macos"

# 3. Build
cmake --build build --parallel $(sysctl -n hw.logicalcpu)

# 4. Deploy Qt runtime dependencies (required for distribution)
/path/to/Qt/6.10.1/macos/bin/macdeployqt build/neuralstudio.app

# 5. Run
open build/neuralstudio.app
# or directly:
./build/neuralstudio.app/Contents/MacOS/neuralstudio
```

> Adjust `/path/to/Qt/6.10.1/macos` to match your Qt installation path.

---

## Project Structure

```
NeuralStudio/
├── CMakeLists.txt
└── src/
    ├── main.cpp
    ├── app/
    │   └── MainWindow.h/.cpp          ← Sidebar + QStackedWidget shell
    ├── core/
    │   ├── nn/
    │   │   ├── NeuralNetwork          ← MLP architecture
    │   │   ├── Layer                  ← Dense layer + dropout + backprop
    │   │   ├── Trainer                ← Gradient-based training (6 optimisers)
    │   │   ├── MetaTrainer            ← Metaheuristic training (17 methods)
    │   │   ├── RBFNetwork             ← RBF architecture
    │   │   ├── RBFTrainer             ← Analytic training (k-means++ + Cholesky)
    │   │   └── ModelSerializer        ← Weight persistence
    │   ├── optimizers/
    │   │   ├── optimizer.h            ← Abstract Optimizer base class
    │   │   ├── NSProblem              ← Network-as-optimization-problem adapter
    │   │   └── [23 concrete files]    ← de, pso, cmaes, jso, arq3, …
    │   ├── dataset/
    │   │   ├── Dataset                ← Core data structure + statistics
    │   │   ├── DatasetLoader          ← Format dispatcher
    │   │   └── loaders/               ← CsvLoader, NeuralDesignerLoader
    │   ├── validation/
    │   │   └── CrossValidator         ← K-fold CV
    │   ├── stats/
    │   │   └── Statistics             ← Wilcoxon, Friedman, box-plot stats
    │   ├── batch/
    │   │   └── BatchRunner            ← Multi-dataset × multi-optimiser engine
    │   └── export/
    │       ├── CppExporter            ← C++ header generation
    │       └── XlsxWriter             ← XLSX result tables
    └── ui/panels/
        ├── DataPanel
        ├── NetworkPanel
        ├── TrainingPanel
        ├── EvaluationPanel
        ├── PredictionPanel
        └── BatchPanel (+ WilcoxonTab, FriedmanTab, BatchPlotWidget)
```

---

## Version

Current release: **v0.8.0**

---

## License

See `LICENSE` for details.
