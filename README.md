# NeuralStudio

A no-code machine learning workbench built with **C++ 20** and **Qt 6**.  
Inspired by Neural Designer (Artelnics). Open-source, extensible, cross-platform.

---

## Command line (all modes)

Every mode also runs without the GUI through the console program `neuralstudio-cli`
(built with the application, Qt 6 Core only):

```bash
neuralstudio-cli train      --data wine.train --hidden 16 --optimizer adam --save-model wine.nsmodel   # Mode 1
neuralstudio-cli batch      --data sample_data/selected --optimizers adam,de,pso --runs 5 --out b.xlsx # Mode 2
neuralstudio-cli pretrain   --data wine.train --iw 6 --runs 10 --out p.xlsx                           # Mode 3
neuralstudio-cli bounds     --data pima.train --opt2 de --runs 10 --out bounds.xlsx                   # Mode 4
neuralstudio-cli archsearch --data wine.train --max-layers 3 --out arch.xlsx                          # Mode 5
neuralstudio-cli ge         --data wine.train --config unified-ge --seed 7                            # Mode 6
neuralstudio-cli ge experiment --experiment experiments/v4_smoke.json                                 # Mode 6 experiments
neuralstudio-cli predict    --model wine.nsmodel --data new_rows.csv --out predictions.csv
```

`neuralstudio-cli <command> --help` lists the options; see [docs/CLI.md](docs/CLI.md).

---

## Phase 1 — Data Loading & Exploration (current)

| Feature | Status |
|---|---|
| Neural Designer format (.train / .test) | ✅ |
| CSV / TSV (auto delimiter, auto header) | ✅ |
| Drag-and-drop file loading | ✅ |
| Data preview table (first 100 rows) | ✅ |
| Per-column statistics (min/max/mean/stddev/median) | ✅ |
| Sidebar step navigation | ✅ |

## Phase 2 — Coming next

- Visual network architecture editor (drag-and-drop layers)
- Training engine: backpropagation, SGD / Adam / RMSProp
- Real-time loss & accuracy charts (Qt Charts)
- Evaluation: confusion matrix, ROC/AUC, regression plots
- C++ / ONNX / Python export
- Workflow automation (batch runs)

---

## Supported Input Formats

| Extension | Format | Detection |
|---|---|---|
| `.train` `.test` `.data` | Neural Designer | header lines: `num_inputs\nnum_samples\n` |
| `.csv` `.tsv` `.txt` | CSV / TSV | auto-detects delimiter (`,` `;` `\t`) and header |

Adding a new format: implement `IDataLoader`, register in `DatasetLoader.cpp`.

---

## Build Requirements

| Dependency | Version | Notes |
|---|---|---|
| C++ compiler | C++20 | GCC 11+, Clang 14+, MSVC 2022 |
| CMake | 3.20+ | |
| Qt 6 | 6.4+ | Widgets, Charts, Gui, Core |
| Eigen 3 | 3.4+ | Optional — used later for matrix math |

---

## Build Instructions

### Linux / macOS
```bash
git clone <repo>
cd NeuralStudio

# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --parallel $(nproc)

# Run
./build/NeuralStudio
```

### With Qt installed in non-standard path
```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/Qt/6.x.x/gcc_64
```

### Windows (Qt Creator)
1. Open `CMakeLists.txt` in Qt Creator.
2. Configure with your Qt 6 kit.
3. Build & Run.

---

## Project Structure

```
NeuralStudio/
├── CMakeLists.txt
├── README.md
├── sample_data/
│   ├── wine.train          ← Neural Designer format example (89 samples)
│   └── wine.test           ← Neural Designer format example (90 samples)
└── src/
    ├── main.cpp
    ├── app/
    │   ├── MainWindow.h/.cpp       ← Shell: sidebar + QStackedWidget
    ├── core/
    │   └── dataset/
    │       ├── Dataset.h/.cpp          ← Core data structure + statistics
    │       ├── DatasetLoader.h/.cpp    ← Format factory / dispatcher
    │       └── loaders/
    │           ├── IDataLoader.h           ← Abstract interface
    │           ├── NeuralDesignerLoader    ← .train/.test parser
    │           └── CsvLoader               ← CSV/TSV parser
    ├── ui/
    │   └── panels/
    │       ├── DataPanel       ← Step 1: load + preview + stats  ✅
    │       ├── NetworkPanel    ← Step 2: architecture editor      🔜
    │       ├── TrainingPanel   ← Step 3: training controls        🔜
    │       └── EvaluationPanel ← Step 4: results dashboard        🔜
    └── utils/
        └── Logger.h            ← Thin QDebug wrapper
```

---

## Neural Designer File Format

```
<num_inputs>          ← integer: number of input features
<num_samples>         ← integer: number of data rows (used as hint)
v1_1 v1_2 … v1_N v1_out   ← space-separated doubles
v2_1 v2_2 … v2_N v2_out
…
```

**Wine dataset example** (13 inputs, 1 output, 3 classes: 0.0 / 0.5 / 1.0):
```
13
89
13.200000 1.780000 2.140000 … 1050.000000 0.000000
…
```

---

## Architecture Notes

- **`Dataset`** owns all data via `std::vector<std::vector<double>>` (row-major).  
  Switch to `Eigen::MatrixXd` in Phase 2 by wrapping in `Dataset::asMatrix()`.
- **`IDataLoader`** is the extension point — implement it and register in `DatasetLoader::allLoaders()`.
- **Panels** are `QWidget` subclasses in `ui/panels/`, decoupled from core via signals.
- The **sidebar** drives a `QStackedWidget`; each step button maps to a panel index.
