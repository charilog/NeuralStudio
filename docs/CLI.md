# neuralstudio-cli — every mode from the command line

`neuralstudio-cli` is the console program of NeuralStudio. It runs the six modes of the
graphical application without a window, so experiments can be scripted, run over SSH or on a
server, and repeated exactly. It is built from the same core sources as the application and
calls the same workers (Trainer, MetaTrainer, RBFTrainer, CrossValidator, BatchRunner,
PreTrainWorker, EvolBoundsWorker, ArchSearchWorker, GEStudioWorker, GEBatchRunner) with the
same defaults as the panels: a command line computes what the corresponding panel computes.

```
neuralstudio-cli <command> [options]
neuralstudio-cli <command> --help
```

| Command | Mode | What it does |
|---|---|---|
| `train` | 1 Single Workflow | train one MLP or RBF network, evaluate it, K-fold CV, save / export, predict |
| `batch` | 2 Batch Run | datasets × optimizers × runs; Holm-adjusted Wilcoxon, Friedman + Nemenyi |
| `pretrain` | 3 Pre-Train + NNC | small pre-trained network transferred into the main network |
| `bounds` | 4 Evolutionary Bounds | per-parameter weight bounds, training inside them (+ GE bounds) |
| `archsearch` | 5 Architecture Search | evolutionary search of layers, sizes and activations |
| `ge` | 6 GE Studio | grammatical evolution: single run, or `ge experiment` (datasets × folds × models) |
| `predict` | – | apply a saved `.nsmodel` network to a data file |
| `list` | – | optimizers, GE configurations, GE models |

The modes can also be given by number: `neuralstudio-cli 4 --data wine.train` is `bounds`.

## Build

The program is built together with the application (CMake option `NS_BUILD_CLI`, on by
default) and needs Qt 6 Core only:

```
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target neuralstudio-cli
```

On Windows it is a console program (`neuralstudio-cli.exe` next to `NeuralStudio.exe`).

## Data files

* Neural Designer files (`.train`, `.test`, `.data`) and CSV / TSV (delimiter and header detected).
* `--data` takes files, folders (every training file inside) and patterns (`"wine_*.train"`),
  comma separated or repeated; data files can also be listed at the end of the command.
* The companion test file is found as in the Data panel (`wine.train` → `wine.test`,
  `a_train.csv` → `a_test.csv`, `.trn` → `.tst`). It is never used for training or selection:
  Mode 1 shows it as the monitoring curve only (all training rows train and early stopping is
  off, as in the Training panel; `--no-monitor-test` trains with the validation split instead),
  Modes 2–6 score it after training.

## Mode 1 — `train`

```
neuralstudio-cli train --data wine.train --hidden 16 --act tanh --optimizer adam --epochs 500 \
                       --out wine.xlsx --save-model wine.nsmodel --export-cpp wine.h --export-onnx wine.onnx
neuralstudio-cli train wine.train --optimizer de --batch 50 --epochs 200 --weight-bound 5
neuralstudio-cli train housing.train --model rbf --centers 30 --kernel gaussian
neuralstudio-cli train pima.train --cv 10 --cv-only
```

* `--optimizer`: any of the 24 methods (`neuralstudio-cli list optimizers`). For the
  non-gradient methods `--epochs` is the number of iterations and `--batch` the population
  (default 30, as the Training panel).
* `--hidden 64,32` (`none` = no hidden layer), `--act relu` or one activation per layer
  (`--act relu,tanh`), `--dropout`, `--task auto|binary|multiclass|regression`.
* `--lr`, `--lr-schedule constant|step|exp`, `--lr-decay`, `--lr-step`, `--weight-decay`,
  `--val-split`, `--patience`, `--weight-bound`.
* RBF: `--model rbf --centers --kernel --width --width-scale --ridge --kmeans-iter`.
* `--cv K`: K-fold cross-validation on the training file (gradient training; a population
  method falls back to Adam, as in the GUI); `--cv-only` skips the final training.
* Output: the report on standard output (training / validation / test metrics, confusion
  matrices, precision / recall / F1, MSE / RMSE / MAE / R²); `--out FILE.xlsx` (settings, metrics,
  history, CV folds); `--history FILE.csv`; `--save-model`, `--export-cpp [--cpp-namespace]`,
  `--export-onnx`; `--predict FILE [--predict-out FILE.csv]`.

## Prediction — `predict`

```
neuralstudio-cli predict --model wine.nsmodel --data new_rows.csv --out predictions.csv
```

Each row holds the input values, optionally followed by the target (then accuracy / errors
are reported). Without `--out` the predictions are written to standard output as CSV:
`row, predicted class, class scores / probabilities [, target]` or `row, prediction [, target]`.

## Mode 2 — `batch`

```
neuralstudio-cli batch --data sample_data/selected --optimizers adam,sgd,de,pso --runs 5 --out batch.xlsx
neuralstudio-cli batch wine.train pima.train housing.train --optimizers gradient --epochs 300
```

* `--optimizers`: names, or `all`, `gradient`, `meta` (default `adam,sgd`, as the panel).
* The network and training options are shared by every job (`--hidden 64,32`, `--act`,
  `--epochs 200`, `--lr`, `--batch`, `--val-split`, `--patience`, `--weight-bound`).
* Statistics: `--stat-metrics auto` tests the test error % (classification) and the test RMSE
  (regression), or the validation figures when a file has no test companion; any of
  `testErrorPct, valErrorPct, trainErrorPct, testRMSE, valRMSE, testMAE, valMAE, trainMAE,
  testR2, valR2`, or `none`. `--stat-task` restricts the tests to one task.
  As in the Wilcoxon / Friedman tabs, repeated runs and the folds `name_1 … name_K` of one
  dataset are averaged into one value per dataset; the Wilcoxon p-values are Holm-adjusted
  over all pairs; the Friedman test (≥ 3 optimizers) is followed by the Nemenyi critical
  difference. R² is ranked with higher = better.
* `--out FILE.xlsx`: sheets Runs, Summary, Wilcoxon <metric>, Friedman <metric>.
* Ctrl+C stops after the training job that is running (the Batch panel stops between jobs too).

## Mode 3 — `pretrain`

```
neuralstudio-cli pretrain --data wine.train,pima.train --hidden 16 --iw 6 \
                          --p1-opt de --p2-opt de --p3-opt none --runs 10 --out pretrain.xlsx
```

`--hidden`, `--act`, `--val-split`, `--iw` (nodes of the small network), `--equal-budget 1|0`;
per phase `--pN-opt --pN-pop --pN-iters --pN-bound` (Phase 3 may be `none`). Optimizers:
`ga de pso cmaes jso lmcmaes acor clpso mlshaderl arq3 lbfgs bfgs sa neldermead`. Defaults as
the panel (GA 200 × 200, bound 10, Phase 3 off, equal-budget baseline on).

## Mode 4 — `bounds`

```
neuralstudio-cli bounds --data pima.train --opt2 de --opt3 ga --runs 10 --out bounds.xlsx
neuralstudio-cli bounds --data pima.train --opt2 ge_grammar --ns-ge 30 --nl-ge 10 --pl-ge 0.05
```

Phase 1 `--opt1 --nc1 --ng1 --lambda --a --f --initial-bound`; Phase 2 `--opt2` (an optimizer,
`ge_grammar` or `none`) `--nc2 --ng2 --ns-ge --nl-ge --pl-ge`; Phase 3 `--opt3` (or `none`)
`--nc3 --ng3 --bfgs 1|0`; `--equal-budget 1|0`. The `.xlsx` file also holds the Phase-1 bounds
of every parameter (sheet Bounds).

## Mode 5 — `archsearch`

```
neuralstudio-cli archsearch --data wine.train --max-layers 3 --sizes 4,8,16,32 \
                            --acts relu,tanh --outer-gens 30 --runs 5 --out arch.xlsx
```

Search space `--min-layers --max-layers --sizes --acts`; outer optimizer `--outer --outer-pop
--outer-gens`; inner training `--inner --inner-pop --inner-iters --inner-bound --polish`;
`--topk`. The report ends with the Mode 1 command that trains the best architecture (the
panel's "Apply to Mode 1").

## Mode 6 — `ge`

Single run (the "Single run" page):

```
neuralstudio-cli ge --data wine.train --config unified-ge --gens 30 --seed 7 --out ge_wine.xlsx
neuralstudio-cli ge --data wine.train --methods standard,ge1,ge1_nobounds,rf,svm
neuralstudio-cli ge --data pima.train --config baselines --methods knn,linear,svm
```

`--config <preset>-<genotype>` (`unified|unifiedpe|archbounds|archonly` × `ge|pge|gelocal|sage`,
or `baselines`), `--grammar FILE.bnf`, `--methods` (rows of the final comparison; reference
models are computed only when listed — `neuralstudio-cli list methods`), and every option of the
panel (evolution, inner training, final comparison, fixed total budget). A `--seed` makes the
run reproducible.

Experiments (the "Experiment" page) — the options are exactly those of `tools/ge_batch`:

```
neuralstudio-cli ge experiment --experiment experiments/v4_smoke.json
neuralstudio-cli ge experiment --dir sample_data/tenfolding --datasets wine,pima --folds 1-10 \
                               --configs unified-ge --seeds 3 --out results/run1
```

Running the same command again resumes; finished runs in `<prefix>_raw.csv` are skipped.
With the same arguments `neuralstudio-cli ge experiment` and `ge_batch` write identical result rows.

## General

* `--quiet`: results only (no progress); `--verbose`: also the log messages of the library.
* Results go to standard output, progress and messages to standard error.
* Ctrl+C stops after the current step and reports what was completed; press it again (after a
  second) to quit at once.
* Exit codes: 0 success, 1 a run failed, 2 invalid command line, 130 interrupted.
