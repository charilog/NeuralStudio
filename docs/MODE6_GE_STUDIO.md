# Mode 6 — GE Studio (Grammatical Evolution with a unified BNF grammar)

One BNF grammar derives a complete neural network:

* **architecture** — number of layers, sizes, activations
* **bound commands** — per-layer `SHRINK / EXPAND / ...` applied to the lower and upper bound of the
  layer's parameters (the commands of Tsoulos & Charilogis, *Big Data Cogn. Comput.* 2025, but acting on
  layer groups instead of every single weight, so the chromosome length does not grow with the number of weights)
* **constructed features** — optional formulas over `x1..xN`, appended to the original inputs

The chromosome never encodes weights. For every *distinct* phenotype an inner optimizer (DE, PSO, CMA-ES, jSO, ...)
trains the weights **inside the intervals the grammar produced**. Fitness = validation **loss** (continuous, so it
cannot saturate at 0 on small validation sets; validation error % is available as an option, with a 0.1 % train-error
tie-break), times an optional parsimony factor `1 + λ·params/1000`. Identical phenotypes are cached. The search stops
early when the best fitness has not improved for `patience` generations (default 8).

Constructed features are constant-folded (`log(exp(log(sin(3))))` becomes one constant), constant features are
rejected, and a feature is dropped when it exceeds `feature size` operations/operands (default 15).

## Genotypes

| | |
|---|---|
| **GE** | integer codons, `codon mod n`, wrapping, one-point crossover, optional **Local Crossover** (Tsoulos, Charilogis, Tsalikakis 2024), generation scheme "best (1-ps)·Nc survive" |
| **PGE** | every non-terminal owns a distribution over its productions, learned PBIL-style from the elite derivations (learning rate α, elite fraction, probability floor). Individuals are choice traces. Learned probabilities can be carried to the next dataset. |

Both share the same derivation, interpreter, cache and final comparison, so GE vs PGE is a controlled comparison.

## Grammar conventions

Meaning is read from rule **names**: `<layer>` (first `<size>`, first `<act>`, first two `<cmd>`), `<out>` (two `<cmd>`),
`<feat>` (its `<expr>`). `@VARS@` expands to `x1 | ... | xN` per dataset. Presets: unified, architecture+bounds,
architecture only. Custom grammars are validated in the panel. A depth limit forces the shallowest production
when needed, so every derivation terminates.

## Final comparison (per dataset, on the companion test file)

All rows use the same inner optimizer and the same final budget; L-BFGS polish (if on) is projected onto the bounds.

1. `Standard` — fixed network (default `16 relu`), bounds ±B
2. `Standard, equal total budget` — same network, budget = GE search evaluations + one final training
3. `GE #1 .. #K` — the K best phenotypes retrained
4. `GE #1 without bound commands` — ablation: same architecture/features, all bounds ±B
5. `GE search cost` — loss evaluations and seconds spent by the whole search

## Files

```
src/core/ge/Grammar.*          BNF parser, derivation tree, codon mapping            (no Qt)
src/core/ge/GEEngine.*         GE + PGE loop, crossovers                              (no Qt)
src/core/ge/NetSpec.*          tree -> network spec, bound commands, features, presets (no Qt)
src/core/ge/GEStudioWorker.*   dataset loading, inner training, caching, comparison
src/ui/panels/GEStudioPanel.*  Mode 6 panel (queue, grammar editor, results, xlsx export)
tests/ge_core_test.cpp         headless unit tests of the GE core
tests/ge_worker_smoke.cpp      end-to-end run without GUI
tests/ge_panel_smoke.cpp       offscreen GUI run with screenshots
```
Build tests: `cmake -B build -DNS_BUILD_TESTS=ON && cmake --build build`.

## Proposed experimental protocol (not yet run)

* **Data**: the `tenfolding` pairs (`name_1 .. name_10`). Queue all ten folds of a dataset; report mean ± std of test error.
* **Research questions**
  1. Does the unified grammar reach the same test error as Mode 5 + Mode 4 (run in sequence) with fewer loss evaluations?
  2. Effect of the bound commands: `GE #1` vs `GE #1 without bound commands`.
  3. GE vs PGE at equal evaluation budget; effect of carrying probabilities across datasets.
  4. Effect of Local Crossover (cost vs gain; it spends `Ncr` extra evaluations per chromosome).
  5. Value of constructed features (unified vs architecture+bounds preset).
* **Fairness**: report loss evaluations, not only seconds; use the equal-budget baseline row; same inner optimizer everywhere.
* **Statistics**: 30 independent seeds per fold and configuration (seed field), Wilcoxon / Friedman over datasets
  (export the xlsx and use Mode 2's statistics tabs or R).

## Batch runner (`tools/ge_batch.cpp`, target `ge_batch`)

Runs every fold x seed x configuration without the GUI and pools the results.

```
ge_batch --dir sample_data/tenfolding --datasets wine,pima,australian \
         --folds 1-10 --seeds 5 \
         --configs unified-ge,archbounds-ge,archonly-ge,unified-pge,unified-gelocal \
         --gens 40 --pop 40 --inner-pop 30 --inner-iters 60 --final-pop 80 --final-iters 200 \
         --out results/run1
```
* Config name = `<preset>-<genotype>` (`unified|archbounds|archonly` x `ge|pge|gelocal`).
* Reads `<dir>/<ds>_<fold>.train` (+ `.test`); missing/empty files are skipped. Seed = `seedIdx*1000 + fold`.
* Resumable: `<out>_raw.csv` is appended after each run; re-running the same command skips finished runs
  (`--no-resume` to start over, `--summarize-only` to just rebuild the reports).
* Methods per run: `standard`, `standard_eq` (equal total budget), `grid_eq` (best of 16 classic architectures searched with the GE search budget, then retrained like a GE phenotype), `ge1..geK`, `ge1_nobounds`, and the equal-total-budget rows `ge1_eq` / `ge1_nobounds_eq` (GE #1 retrained with the same budget as `standard_eq`; this is the fair comparison of architecture + bounds against the plain network). Default patience is 15.
* Outputs: `_raw.csv`, `_summary.csv`, `_paired_tests.csv`, `_summary.xlsx`
  (Summary, Search cost, Paired tests, Raw). Tests: paired Wilcoxon of GE #1 vs Standard / Standard at equal budget /
  GE #1 without bounds, between configs, across datasets, and Friedman over datasets.
* Build: `cmake -B build -DNS_BUILD_TESTS=ON && cmake --build build --target ge_batch`.

## Known limitations

* Smoke runs in `tests/` are single runs and show nothing about performance.
* A very small validation split is still noisy even with the continuous loss; prefer datasets with a few hundred rows or more.
* The companion test files of the 10-fold sets are small (wine: 17 rows, so one error = 5.9 %). Pool all folds before comparing methods.
* A network with constructed features cannot be saved/exported with the existing Save/ONNX actions (they lack the
  feature pre-processing). Mode 6 hands a network to the rest of the app only when the best phenotype has no features.
* Targeted / TargetedWorst crossovers of the GenCrossover paper are not implemented. No N-run loop (Mode 5 has one).
* Mode 4's `ge_grammar` option (a random-sampling surrogate, no real GE) is left untouched.
* Built and tested on Linux / GCC / Qt 6.4 only. `optimizer.h` needed `#include <stdexcept>` for GCC.

## v6 method (fixed total budget, Lamarckian GE)

Why v5 could not show a benefit of features or bound commands, and what changed:

| Problem in v5 | Change in v6 |
|---|---|
| Features computed and min-max scaled with heavy tails: most rows squeezed into a tiny part of [0,1] | features on raw inputs, non-finite replaced, clipped to the train 1st..99th percentile (test uses the train percentiles); grammar gains `/`, `sqrt`, `abs` |
| Bounds were an absolute ±B for every layer | scale-aware base per layer: `fanK * sqrt(6/(fanIn+fanOut))` (`boundMode=fan`); commands scale it |
| Networks under-trained: no overfitting, so a regulariser (bounds) has nothing to fix; cheap fitness favoured fast-training tiny nets | hybrid training everywhere (global optimizer + bound-respecting L-BFGS); a gradient evaluation is charged as 3 loss evaluations |
| Selection and final training used different trainers | the search uses the same hybrid training |
| Low-fidelity bias | **Lamarckian GE**: a phenotype that re-appears (GE's many-to-one mapping) resumes training from its stored weights; fitness kept only if validation improves. Then two-stage selection: top `finalists` refined to full depth and re-ranked |
| Budget accounting favoured whichever row got more training | `totalBudget T`: search stops at `searchFrac*T`; GE rows get `T - search`; `standard_T` gets T; `grid_T` = grid search with the same search share + the rest of T; ablations `ge1_T_nobounds`, `ge1_T_nofeat` get the same budget as `ge1_T` (an ablation with an identical phenotype copies the row) |

Statistics: paired Wilcoxon with folds as units (seeds averaged per fold), medians reported.

Hypotheses (state before running the confirmation datasets):
H1 ge1_T < standard_T, H2 ge1_T < grid_T, H3 ge1_T < ge1_T_nobounds, H4 ge1_T < ge1_T_nofeat (unified only).
Development datasets: airfoil, concrete, housing. Confirmation datasets must not be used for any tuning.

## Batch experiments from the GUI (v7)

The batch engine moved to `src/core/ge/GEBatch.{h,cpp}` and is shared by `tools/ge_batch.cpp` (CLI)
and the GE Studio panel. In the panel, expand **Batch experiment (command line)**, paste the same command
you would type in a terminal and press **Run batch**:

* the program name at the start (e.g. `.\build_batch\Release\ge_batch.exe`) is optional and ignored;
* paths may use `\` or `/` on Windows and Linux; quotes ("..." or '...') work for paths with spaces;
  multi-line commands with PowerShell `` ` ``, cmd `^` or bash `\` continuations can be pasted as they are;
* relative paths (`--dir`, `--out`) are resolved against **Working folder** (default: the folder that holds
  `sample_data`, found by searching upwards from the executable);
* it runs in-process on a worker thread: no console window; the progress bar follows generations and
  final training inside each run; the status line shows the run and the ETA; the **Log** tab shows the
  log and, at the end, the full summary (same text as the CLI);
* **Stop** discards only the unfinished run; running the same command again resumes;
* **Check** validates the command and reports how many runs are done / left; **Options** lists all options;
  **Open results folder** opens the output folder.

Do not run the same `--out` from the GUI and from a terminal at the same time.
Offscreen GUI test: `tests/ge_batch_gui_smoke.cpp`.

## Experiment files and overall tables (v8)

### Experiment file (`experiments/*.json`)
```json
{
  "name": "bench64",
  "data_dir": "sample_data/tenfolding",
  "output": "results/bench64",
  "folds": "1-10",
  "seeds": 1,
  "datasets": { "classification": ["pima", "wine"], "regression": ["airfoil", "concrete"] },
  "configs": ["unified-ge"],
  "models": [
    { "id": "standard",       "bnf": false, "run": true },
    { "id": "standard_T",     "bnf": false, "run": true },
    { "id": "grid_T",         "bnf": false, "run": true },
    { "id": "ge1",            "bnf": true,  "run": true },
    { "id": "ge1_T",          "bnf": true,  "run": true },
    { "id": "ge1_T_nofeat",   "bnf": true,  "run": true },
    { "id": "ge1_T_nobounds", "bnf": true,  "run": false }
  ],
  "options": { "total-budget": 600000, "gens": 40, "pop": 40, "inner-pop": 30, "inner-iters": 60,
               "final-pop": 80, "final-iters": 200, "patience": 15 }
}
```
* `models[].bnf` documents whether a model comes from the BNF grammar; a wrong flag is reported and corrected.
  `run: false` skips that model (passed to the engine as `--methods`). Models of the other budget mode are reported and ignored.
* Relative paths are resolved against the working folder. Loading checks that every dataset is read with the
  declared task (classification / regression) and lists mismatches.
* CLI: `ge_batch --experiment experiments/bench64.json` (other options on the command line override the file).
* `experiments/bench64.json`: the 39 classification + 25 regression datasets of the BNF paper available in
  `tenfolding` (hepatitis and plastic excluded: their files do not match the task of the paper).
  `experiments/mini_test.json`: a few-minute functional test.

### Overall tables
After every batch (and with **Load results…** for any `<prefix>_raw.csv`) the panel shows the tabs
**Classification** and **Regression**; `<prefix>_overall.xlsx` (or **Export tables (.xlsx)**) contains:
* `Classification`, `Regression`: one row per dataset, columns grouped **Without BNF** / **With BNF**, value = mean
  test error over folds (seeds of a fold averaged first; % for classification, MSE for regression); then AVERAGE,
  average Friedman rank, wins/ties/losses and Wilcoxon p of the main GE model (`GE #1 (T)`) against every column
  (datasets as units), Friedman p and the Nemenyi critical difference at 0.05; one block per grammar configuration;
* `… (std)`: the standard deviation over folds; `Methods`: id, label, uses BNF, description; `Cost`: mean time and
  parameters per model.

Jobs run smallest dataset first (by train-file size), so most datasets finish early and the long ones run last.

## Reference models (v9, configuration `baselines`)

Run as the grammar-free configuration `baselines` (e.g. `--configs unified-ge,baselines`, or listed in the
experiment file's `models`); in the overall tables they appear as extra columns next to the GE rows.
Same folds, same metrics (error % / MSE on the raw target), inputs min-max normalised with the train statistics.

| id | model | settings |
|---|---|---|
| `adam` | MLP H=10 sigmoid, Adam | lr 1e-3, batch 32, >= 500 epochs and >= 20 000 updates |
| `bfgs` | MLP H=10 sigmoid, L-BFGS | memory 20, <= 2000 iterations, random (Glorot) start |
| `genetic` | MLP H=10 sigmoid, GA of the BNF paper | Nc 500, Ng 200, ps 0.1 (keeps best (1-ps)Nc), pm 0.05, crossover a ~ U[-0.5,1.5], tournament 4, weights in [-10,10] |
| `neat` | NEAT (feed-forward) | 150 genomes, 200 generations, c1=c2=1, c3=0.4, delta_t=3, add node 0.03, add link 0.05, stagnation 15 |
| `rbf` | RBF network, 10 Gaussian centres | k-means++, sigma = d_max / sqrt(2k), ridge output layer |
| `knn` | k nearest neighbours | k in {1,3,5,7,9,11,15,21} chosen on an 80/20 split of the train file |
| `linear` | logistic (cls) / ridge (reg) | L2 strength in {1e-6..1} chosen on the split |
| `rf` | random forest | 100 CART trees, R defaults (mtry sqrt(d) / d/3, min leaf 1 / 5) |
| `svm` | RBF-kernel C-SVC (one-vs-one) / epsilon-SVR | SMO with WSS2 (as LIBSVM), gamma = 1/(d Var X), C in {1,10,100} on the split, epsilon 0.1 (standardised y) |
| `gnn` | the method of the BNF paper (uses BNF) | phase 1 GA; bounds [-F|w|,F|w|]; phase 2 GE (grammar of its Fig. 2) with interval fitness over Ns=50 random networks + local search (Nl=20, pl=0.001); phase 3 GA inside the bounds + bounded BFGS. Not given in the paper: F=2, 200 codons, tournament 4, phase-1 interval [-10,10] |

Classification uses cross-entropy for the gradient/evolutionary MLP models (regression: MSE on standardised targets).
kNN, logistic/ridge and SVM were checked against scikit-learn 1.9 on the same folds and hyper-parameters
(identical test errors on pima, wine, airfoil, housing; one sample of 231 different on segment).
Tools: `tests/baselines_check.cpp` (one fold, every model, chosen hyper-parameters).

## One panel, two pages; reopening and changing experiments (v10)

The panel now has a single **Run**, **Stop**, **Export (.xlsx)** and **Open results folder** row. What they act
on depends on the setup page:

| Page | Run | Export (.xlsx) |
|---|---|---|
| **Single run (queued files)** | GE Studio on the queued `.train` files with the settings of the page | comparison, phenotypes, convergence of that run |
| **Experiment (datasets × folds × models)** | the whole experiment (same engine as `ge_batch`, in-process, no console) | overall tables: Classification / Regression, with and without BNF |

The earlier separate controls (Run batch, Load/Save experiment, 64-dataset template, Load results, Export
tables) are merged into the Experiment page:

* **New ▾**: an empty experiment (opens the editor) or the 64-dataset benchmark of the BNF paper.
* **Open…**: an experiment file (`.json`) **or the results of an earlier experiment (`<prefix>_raw.csv`)**.
  Every run now writes its exact settings next to the results (`<prefix>_experiment.json`), so an experiment
  is reopened exactly. For older results without that file the datasets, configurations, models, folds and
  seeds are read from the rows; options are then the defaults (a note says so).
* **Edit…**: dialog with a *Datasets* tab (all datasets of the data folder, task detected from the file,
  filter, all classification / all regression, and how many folds already have results), a *Models* tab
  (run yes/no, uses BNF yes/no, result rows already present) and an *Options* tab.
* **Save / Save as…**: the experiment file. The summary shows models with / without BNF, datasets and
  "x of y runs complete".
* **Command line** (collapsible): the equivalent `ge_batch` command. A pasted or edited command replaces
  the experiment when Run or *Check / apply* is pressed.

Changing an experiment that already has results (same results prefix):

* **Add a reference model** (kNN, SVM, …): only that model is computed on every existing fold and appended;
  existing rows are not touched.
* **Add a BNF / GE model** (e.g. `ge1_T_nobounds`): the GE run of each fold is repeated so that all GE rows
  (and their ablations) refer to the same phenotype; the old GE rows of that fold are replaced.
* **Add a dataset or fold**: all requested models run on it.
* **Remove a model or dataset**: nothing is deleted; its rows stay in `<prefix>_raw.csv` and only leave the
  tables (the status line reports "N of M result rows belong to this experiment").
* Changing GE **options** of an experiment that has GE results makes old and new GE rows incomparable; use a
  new results prefix for that (the editor warns).

GUI tests: `tests/ge_batch_gui_smoke.cpp` (pasted command, progress, stop/resume),
`tests/ge_exp_open_smoke.cpp` (Open `<prefix>_raw.csv` → Edit: +model, −model, +dataset → Run only the
missing runs), `tests/ge_panel_smoke.cpp` (single run).

## Models lists (v11)

Models are chosen by clicking, like datasets, on a **Models** list (`src/ui/panels/GEModelPicker.*`): two groups,
**Without BNF** and **With BNF**, a check box per model (a group box ticks / clears the whole group),
buttons *All / None / With BNF / Without BNF* and the count "n selected: x with BNF, y without". The same
list appears in three places:

* **Single run page** (between the dataset queue and the grammar): the rows of the GE comparison
  (Standard, Standard (eq), Grid search (eq), GE #1, GE #2 … #K, ablations, equal-budget retraining) and every
  reference model (ADAM, BFGS, GENETIC, NEAT, RBF, kNN, Logistic / ridge, Random forest, SVM, GNN). Reference
  models are trained on the same train file and measured on the same test file, after the GE rows. With only
  reference models ticked no GE search runs. The selection is remembered. (The former check box
  "add equal-total-budget baseline" is replaced by the Standard (eq) / GE #1 (eq) entries.)
* **Experiment page**, next to the summary: a tick changes the experiment at once (the command line, the
  "x of y runs complete" line and the tables follow); with results present the list shows the result rows of
  every model. The next Run computes only what the added models lack.
* **Edit… → Models** tab, the same list.

## Scale-free tests, hypotheses and result tabs (v12)

* **Relative difference** per dataset: rd = (method − reference) / mean(|method|, |reference|), 0 when both are 0,
  rd > 0 = the reference (GE #1 (T)) is better. It is scale-free, so regression datasets with a large MSE
  (e.g. eu, baseball) do not dominate the test the way the raw-value Wilcoxon does. The overall tables gain the rows
  *Median relative difference %*, *Geometric mean ratio*, *Wilcoxon p on relative differences* and
  *Holm-adjusted p* (Holm over all columns compared with the reference).
* **Hypotheses** (tab and xlsx sheet): H1 Standard (T), H2 Grid search (T), H3 no bounds, H4 no features, H5 GNN
  (equal-budget mode: the _eq rows), per task and for both tasks together, Wilcoxon on rd, Holm over the hypotheses of
  the same scope. Left out of these tests: the **development datasets** and datasets without every fold.
* **Development datasets**: experiment key `"development_datasets": [...]`, field in Edit…, option `--dev-datasets`.
  They stay in the tables and are only left out of the hypothesis tests.
* **Result tabs follow the page**: Single run shows Convergence, Comparison, Phenotypes, Grammar statistics and Log;
  Experiment shows Log, Classification, Regression and Hypotheses (an experiment has no single convergence
  curve or phenotype list; its results are the overall tables). Since 1.0.6 the former **Log** and **Batch**
  tabs are one **Log** tab: the timestamped events and the experiment report (progress, summaries, notes).
* Fix: L-BFGS polish stops when an iteration evaluates nothing (zero gradient); it used to spin forever
  (BL fold 9 hung for hours).

## Self-adaptive grammar ensemble (v13, "v2" of the method)

Motivation from the 64-dataset results: GE #1 (T) under-fits (train error 11 % vs 0.9 % for RF), the
single network chosen on a small validation split leaves gain unused (best GE variant per dataset ≈ 8 % better),
and bounds / features help a single network little. Three self-adaptive components (all decisions come from
progress or noise measured during the run):

* **A1 noise-aware selection and ensemble** — the best `--ens-pool` (12) archive phenotypes are scored per
  validation sample with their search-time weights.
  *ge1_T_ad*: among the candidates within `--se-rule` (1) paired standard errors of the best, the simplest.
  *ens_T*: greedy selection with replacement (Caruana 2004), at most `--ens-max` (8) picks; a member is added
  unless it makes the validation loss worse by more than `--se-rule` standard errors (non-inferiority — averaging
  lowers variance, which the selection split cannot show); on equal gain an unused member is preferred.
* **A2 noise-aware search stop** (`<preset>-sage` configurations, e.g. `unified-sage`) — a generation counts as
  progress only if the best validation loss improves by more than `--se-rule` standard errors; with `--patience`
  stalled generations the search stops and the rest of T goes to training.
* **A3 self-adaptive training** (all `_ad` / `ens` rows) — global phase (DE) in blocks of 5 generations until a
  block gains < 1 % or 30 % of the budget is used; then bound-respecting L-BFGS with active-set projection, in
  chunks, measuring the validation loss (held-out split of the training rows, different per member) and keeping
  the best-validation weights; a descent ends when the validation loss has not improved for as long as it took to
  reach its best (≥ 20 chunks) or L-BFGS converges; then it restarts around the best weights (iterated local search)
  until the budget is spent.

Rows (fixed total budget T, computed only when listed in `--methods` or with a `-sage` configuration):
`standard_T_ad`, `std_ens_T` (as many standard networks as the GE ensemble has distinct members, T shared),
`ge1_T_ad`, `ens_T`, `ens_T_nofeat`, `ens_T_nobounds`. With `ens_T` present it is the reference of the tables;
hypotheses H1 std_ens_T, H2 ge1_T, H3 ens_T_nobounds, H4 ens_T_nofeat, H5 gnn, H6 ge1_T_ad.

Experiments: `experiments/v2_dev.json` (development: contraceptive, sonar, airfoil, concrete, housing, ele) and
`experiments/v2_confirm.json` (37 datasets never used before; run once, after the settings are frozen).
Old commands and experiment files keep their meaning: without `--methods` a configuration computes the classic rows.

## Code audit and corrections (v14)

Full list, effects and verification: `docs/AUDIT_v14.md`. For Mode 6 the important changes are:

* **Regression output scaling** — a regression network computes `y = mean + sd · (linear output)` with the mean /
  standard deviation of the train targets, so the hidden layers and the weight bounds of the grammar work on a
  standardised target (as every reference MLP does). Losses, predictions and reported errors stay on the scale of
  the data. All regression rows of the network methods (standard*, grid_T, ge1*, ens*) of earlier result files were
  computed without it and must be re-run; the reference models are unchanged.
* Multi-class gradient of the classic L-BFGS polish divided by n once (was twice).
* Nemenyi critical difference valid for any number of methods (was the k = 10 value for k > 10, too small);
  pair p-values are the family-wise Nemenyi p-values; Friedman p-value exact; scale-aware ties.

Results files written by v13 or earlier cannot be mixed with v14 runs in one table: start the experiments with a
new output prefix (or `--no-resume`).

## Classification improvements under development (v17, NeuralStudio 1.0.4)

Diagnosis from `v2_dev`: the GE ensemble wins on regression, but on some
classification sets (sonar, spiral, phoneme) plain MLPs lose to RF/SVM/kNN,
and the greedy ensemble collapses to 1-2 distinct members. Three switchable
additions were made. None of them changes existing configurations or rows.

- **Periodic input embedding chosen by the grammar** (preset
  `UnifiedEmbedding`, configuration `unifiedpe-sage`). The rule
  `<emb> ::= E(none) | E(<freq>)`, `<freq> ::= 0.5|1|2|4|8|16` lets GE decide
  per network whether to append sin/cos(2*pi*c*u) features, where u is the
  input min-max scaled with training statistics only and c = sigma * z with
  fixed N(0,1) draws z (seed 0x5EEDF00D), m = clamp(64/d, 1, 4) frequencies
  per input. Networks with an embedding are not exported (like networks with
  constructed features).
- **Diversity-aware ensemble selection** (row `ens_T_div`). In the greedy
  Caruana step an unused candidate is preferred over re-adding a used one when
  a paired test of the two ensemble validation losses gives
  `dm <= se-rule * dse`.
- **Cross-validation committees** (rows `ens_T_cv`, control
  `std_ens_T_cv`). Each selected member is trained K = 5 times, once per fold
  of a K-fold split of the training file (`NSProblem::setFoldSplit`), and all
  K x members networks vote. The total budget T is unchanged (each network
  gets T / (members * K)).

The development experiment is `experiments/v3_dev.json`. Its datasets do not
overlap with the confirmation datasets. The final method is chosen from it
and frozen before the confirmation run.

## SAGE, the final method (frozen 2026-10-23 after v3_dev)

Decision from `v3_dev` (15 development datasets x 5 folds): configuration
`unifiedpe-sage`, row `sage` = periodic input embedding chosen by the grammar +
diverse noise-aware ensemble selection + every member a 5-fold CV committee,
total budget T. In v3_dev this variant (then called `ens_T_cv`) had the best
average rank against the 10 reference models (2.6 of 11), beating RF on 10/15
and SVM on 11/15 datasets; known weak spots were tae, bands, sonar and ele
(embedding overfits). No dataset-specific tuning was done after that.

Rows (all share the same data splits, so each ablation differs from `sage`
only in what it removes):

| id | meaning |
|---|---|
| `sage` | the final method |
| `sage_std` | as many standard networks, same CV committees and budget (effect of the grammar) |
| `sage_noemb` | same members without the periodic embedding |
| `sage_nofeat` | same members without constructed features |
| `sage_nobounds` | same members without bound commands |
| `sage_one` | the best member alone (effect of the ensemble) |

Hypotheses with reference `sage` (one Holm family per configuration and
scope): P1-P10 SAGE < rf, svm, knn, linear, adam, bfgs, genetic, neat, rbf, gnn;
A1-A5 SAGE < sage_std, sage_noemb, sage_nofeat, sage_nobounds, sage_one.

Experiment files: `v3_smoke` (quick check), `v3_confirm` (37 never-used
datasets, 10 folds; the evidence), `v3_main` (64 datasets of the BNF paper;
the development datasets are excluded from the tests and reported separately).
