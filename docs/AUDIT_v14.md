# Code audit of the machine-learning models (v14)

Scope: the neural-network core (`NeuralNetwork`, `Layer`, `Trainer`, `NSProblem`), every optimizer used for
training, the reference models of Mode 6 (`Baselines.cpp`), the data loaders, the statistics (Wilcoxon,
Friedman, Nemenyi, Holm) and the exporters. Every fix below was verified with a test (section "Verification").

## A. Errors that change results of Mode 6 (GE Studio)

| # | Where | Error | Fix | Effect on `paper_main` (v12 results) |
|---|-------|-------|-----|--------------------------------------|
| A1 | `NeuralNetwork`, `NSProblem`, `Trainer`, all network builders | **Regression networks were trained on the raw target scale.** The output layer is linear and the weights are bounded (±10, grammar / fan-in bounds), so a target such as house prices (≈2·10⁵), stock (≈47) or housing (≈22) had to be produced by weights near their bounds; every reference MLP standardises the target instead. | The regression network now computes `y = mean + sd · (linear output)` with mean / sd of the **train** targets (`fitOutputScaling`). Hidden layers and bounds work on a standardised target; losses, predictions and reported errors stay on the data scale. Stored in saved models and in the C++ / ONNX exports. | All regression rows of the NeuralNetwork-based methods (standard, standard_T, grid_T, ge1, ge1_T and ablations, adaptive rows) must be **re-run**. The reference models (adam, bfgs, genetic, gnn, neat, rbf, knn, linear, rf, svm) already standardised y and are not affected. |
| A2 | `NSProblem::evalAndGrad` | **Multi-class gradient divided by n twice** (once per sample, once when collected). | Divided once. | L-BFGS polish of the classic rows on multi-class datasets. L-BFGS rescales its steps (γ = sᵀy / yᵀy), so the effect is mostly a weaker first step and a looser Armijo test — small, but those rows are re-run anyway with the new build. |
| A3 | `Statistics::nemenyi` | **Critical value for k > 10 methods used the k = 10 value**, so the critical difference was too small (k = 17: q = 3.164 instead of 3.458, CD 8.5 % too small → too many "significant" pairs). The per-pair p-values were unadjusted normal p-values. | q is computed for any k from the studentized range distribution (infinite df); pair p-values are the family-wise Nemenyi p-values (p < 0.05 exactly when the gap exceeds CD). Checked against SciPy for k = 2…50. | CD figures in the overall tables / Friedman tab change (larger). No hypothesis test changes (they use Wilcoxon + Holm). |
| A4 | `Statistics::chi2pvalue` | Friedman p-value from the Wilson–Hilferty approximation (up to ~2× off for small p). | Exact regularised incomplete gamma Q(df/2, χ²/2). Max relative error vs SciPy 4·10⁻¹⁰. | Friedman p-values (all were < 0.001; conclusions unchanged). |
| A5 | `Statistics` (Wilcoxon, Friedman, Nemenyi) | Ties detected with an **absolute** tolerance 10⁻⁶, which also merges real differences of small regression errors (BL MSE ≈ 3.5·10⁻⁴). | Scale-aware tolerance (10⁻⁹ of the magnitude). | No false ties existed in `paper_main` (checked); preventive. |

## B. Errors in the rest of NeuralStudio (Modes 1–5)

| # | Where | Error | Fix |
|---|-------|-------|-----|
| B1 | `Trainer::trainBatch` | Multi-class gradient divided by the batch size twice (here and in the layer update): SGD / RMSProp / AdaGrad / Nadam learnt ~bs times slower on multi-class data. | Divided once. |
| B2 | `Trainer` | Regression trained on raw targets: with SGD the effective step grew with the square of the target scale (unstable or very slow for large targets). | Gradient of the standardised squared error (A1); reported losses stay on the data scale. |
| B3 | `Trainer` | Early stopping left the network at the **last** epoch (patience epochs after the best). | The best-validation weights are restored. |
| B4 | `TrainingPanel` → `Trainer` / `MetaTrainer` | With a test file loaded, the **test file was the validation set of early stopping** (the stopping epoch was chosen on the test data). | The test file is now a monitoring curve only; early stopping is disabled in that case (message in the log). Without a test file the internal validation split is used as before. |
| B5 | `CrossValidator` | Min–max input scaling used the statistics of the **whole** dataset (held-out fold included). | Statistics of the training fold only. |
| B6 | `Optimizer::localSearch` stub | Returned the candidate with the fitness of the global best: with in-run local search enabled, unchanged candidates received a false fitness. | Returns +∞ (nothing is replaced). Not enabled in Mode 6. |
| B7 | `Optimizer::setup` | Single-point methods (L-BFGS, CG, LM, SA, Nelder–Mead) without `setInitialPoint` started from the **all-zero vector** — a symmetric saddle point of a neural network (all hidden units stay identical). Affected Mode 3/4/5 when such an optimizer was chosen. | Small random start around the centre of the box. |
| B8 | `mLSHADE_RL` | Strategy tables were allocated only in `configure()`; the Mode-6 / architecture-search factories never call it → **crash** (out-of-range write) when mLSHADE-RL was selected. | Allocated in `init()`. |
| B9 | `SimulatedAnnealing` | Moves were judged against the best point instead of the current one, the temperature was in absolute loss units, and cooling 0.97 per evaluation froze the search after ~1000 evaluations whatever the budget. | Metropolis rule on the current point, temperature relative to |f(x₀)|, cooling derived from the budget. Sphere-20D with 20 000 evaluations: 21.8 → 0.036. |
| B10 | `LBFGS` | Armijo test against the best value instead of the value at the current iterate. | Uses f(current). |
| B11 | `CppExporter` | Generated code wrote each hidden layer into an input buffer of size `nIn` → **buffer overflow** when a hidden layer is wider than the input; weights written with 6 significant digits. | Buffer sized to the widest layer; 17 digits. |
| B12 | `NeuralDesignerLoader` | Columns split on a single space (tab-separated files failed); unparsable values became NaN silently. | Any whitespace; unparsable values are reported in the log. |
| B13 | `NSProblem` | Regression loss summed over outputs in the fitness but averaged in the gradient (inconsistent with several outputs). | Averaged everywhere (single-output datasets unchanged). |

## C. Checked and correct

* GE grammar, bound commands, feature parser, genetic operators; DE bounds handling; fixed-budget accounting.
* All optimizers (DE, PSO, GA, jSO, CMA-ES, CLPSO, ACOR, LM-CMA-ES, mLSHADE-RL, SA, Nelder–Mead, L-BFGS) on a
  bounded 20-D test: budget respected exactly, no bound violation, reported best = re-evaluated best.
* Reference models: RF and ADAM agree with scikit-learn on 8 datasets (pima, wine, segment, housevotes, housing,
  abalone, stock, BK; ours equal or slightly better); kNN, linear, SVM checked earlier; BFGS / GA / GNN / NEAT / RBF
  reviewed (standardised targets, train-only statistics, evaluations counted).
* Wilcoxon signed-rank (exact without ties, normal approximation with tie and continuity correction) matches SciPy.
* Data: no NaN, no train/test overlap (except duplicate rows in `magic`), no class of a test file missing from its
  train file, and the task detected from the train file agrees across the 10 folds of all 117 datasets.

## D. Known limitations (not changed)

* The task is detected from the train file: integer targets with ≤ 20 values are classes, other targets are
  regression. Class labels stored as fractions (e.g. `glass`: 0, 0.167, …) are therefore treated as regression,
  while `SN`, `pl`, `QU` (regression with few distinct values) are correctly regression. The experiment dialog shows
  the detected task of every dataset.
* Wilcoxon uses the exact distribution for any n when there are no ties (SciPy switches to the normal
  approximation above n = 50; the exact value is the more accurate one).

## Verification

* `grad_check` (binary, multi-class, regression incl. california): serial and parallel gradients agree to 10⁻¹⁴
  and with finite differences; fitness, gradient loss and loss from predictions identical; the `Trainer` (SGD and
  Adam, early stopping) beats the trivial predictor on every dataset.
* Statistics against SciPy: Friedman p (4·10⁻¹⁰ relative), Nemenyi q (k = 2…50, 4 decimals), Wilcoxon.
* `ge_core_test`, `ge_experiment_test`, GUI smoke tests.

---

# Second pass (v15): every mode, file by file

Scope: the data layer, the network core, RBF, every optimizer, the workers of Modes 2–5, Mode 6 (grammar,
engine, worker, reference models, reports) and the metric computations of the panels.

## E. Errors found and fixed

| # | Mode | Where | Error | Fix |
|---|------|-------|-------|-----|
| E1 | 1 | `RBFTrainer`, `RBFNetwork` | **RBF classification was broken.** The output weights were fitted to the *raw* label column (one column even for K classes): binary predictions went through a sigmoid, so a score of 0 became 0.5 and almost every sample was predicted as class 1 (pima: 54 % test error, the majority class alone gives 37 %); for K classes the K-output network read a 1-column weight matrix out of range. Inputs were not scaled, so k-means distances were dominated by the features with large units. | Targets encoded per task (binary 0/1, one-hot for K classes); inputs min-max scaled with train statistics inside the network; binary score compared with 0.5 directly. Test error now: pima 29 %, wine 6 %, segment 13 %, housing MSE 27 vs 83 of the mean predictor. |
| E2 | 1 | `EvaluationPanel` (RBF) | True class of a sample taken as `label >= 0.5` (binary) or the argmax of a one-column label (always 0, K classes), so the reported RBF accuracy was meaningless. | Index of the label among the train class values. A test file loaded by hand is now also evaluated for an RBF model. |
| E3 | 3, 4, 5 | `ArchSearchWorker`, `EvolBoundsWorker`, `PreTrainWorker` | `loadCombined` **merged the companion test file into the training data**; the architecture choice, the bounds, the training and the accept/reject decisions all saw the test rows. | Train file only. Modes 4 and 5 additionally report the error on the test file ("Test Error %" / "Test MSE" rows), which never influences anything. |
| E4 | 4, 5 | same | When a phase was worse than the previous one on train or validation, **its row was overwritten with the previous numbers**, so the comparison table could never show a phase losing. | Each row shows what that phase achieved; only the weights carried forward fall back (the final network is never worse on the validation split). |
| E5 | 5 | `PreTrainWorker::adaptW` | The pre-trained small network was copied as a **prefix of the flat weight vector**: after the first-layer weights, its biases and output weights landed on unrelated parameters of the large network. | Neuron-wise transfer: hidden neurons 0..Iw−1 get the input weights and biases, the output weights of the copied neurons are copied and the extra neurons get output weight 0, so the large network starts as exactly the pre-trained function. |
| E6 | 4 | `EvolBoundsWorker::runPhase3` | The final L-BFGS was given an absolute budget of 2000 evaluations while the evaluation counter already included Phase 3, so it **usually never ran**. | 2000 evaluations from the current count. |
| E7 | all | `NelderMead` | Trial points (reflection, expansion, contraction) were clamped *before* being computed, so the simplex **left the bounds** (400 000 out-of-box evaluations in a 20-D test, and a reported optimum the box does not contain); a warm start that was the best vertex could be replaced by a worse one. | Clamp after computing; the start point counts. |
| E8 | all | `GA`, `CLPSO`, `PBIL`, `UMDA` | The initial population (and each PBIL/UMDA generation) was evaluated **ignoring the evaluation budget** (budget 10 → 40 evaluations), which matters for every fixed-budget comparison; a GA generation cut by the budget left empty individuals with fitness 0 that then ranked first. | Budget respected everywhere; unfilled slots keep old individuals. |
| E9 | all | `Optimizer::setInitialPoint` | A warm start outside the box (e.g. weights trained under wider bounds, Mode 4 Phase 2→3) was evaluated and could be returned as the result. | Projected onto the box. |
| E10 | all | `PSO::init` | Returning on budget exhaustion skipped the best-so-far update. | Fixed. |
| E11 | 1, 2 | `ConjugateGradient` | Armijo test against the best value instead of the current iterate (as L-BFGS in v14). | Fixed. |
| E12 | 1, 2 | `LevenbergMarquardt` | Re-evaluated loss and gradient at the unchanged point after every rejected step; the comment claimed equivalence with (JᵀJ + λI)⁻¹Jᵀr, which is not what it computes. | Gradient kept from the last accepted step; documented as a diagonal trust-region heuristic. |
| E13 | all | `CMAES::updateEigenSystem` | Jacobi method searching the whole matrix before every rotation: O(D⁴) per decomposition, prohibitive for network weights. | Cyclic Jacobi sweeps, O(D³) per sweep (same result). |
| E14 | 1, 2 | `MetaTrainer` | With early stopping the network was left at the best *training* fitness, not at the best validation iteration (the gradient Trainer restores the best epoch since v14). | Best-validation weights restored. |
| E15 | 1 | `TrainingPanel` (cross-validation) | The optimizer of the CV was `index 0 ? Adam : SGD`: RMSProp, AdamW, Nadam, AdaGrad were run as SGD. Fold training errors were ignored (fold reported as 0). | Selected optimizer used (a population method falls back to Adam, stated in the status line); errors stop the CV. |
| E16 | 1 | `Layer::updateAdamW` | Weight decay also applied to the biases. | Biases not decayed (Loshchilov & Hutter). |
| E17 | 2 | `WilcoxonTab`, `FriedmanTab` | A value was "missing" when negative, so **negative R² was dropped** (removing exactly the datasets where some optimizer failed); several results of one optimizer on one dataset overwrote each other; the folds `name_1 … name_K` counted as separate datasets (dependent observations treated as independent). | Shared builder: validity by task, repeats and folds averaged into one value per dataset (shown in the info line). |
| E18 | 2 | `BatchPanel` | Means over repeated runs included failed runs (their −1 entries, e.g. for R²). | Successful runs only. |
| E19 | all | `CompanionFinder::roleOf` | `.trn` / `.tst` were returned as "trn"/"tst", so clicking a `.tst` file loaded it as the **training** set. | Roles "train"/"test" for every naming pair. |
| E20 | all | `CsvLoader` | `1,,3` (missing value) dropped the whole row; non-numeric values became NaN silently. | Empty fields kept as missing; counts reported in the log. |
| E21 | all | `NeuralNetwork::normalize` | A missing value (NaN) propagated through the network and every loss. | Replaced by the train mean of its column (stored in saved models). |
| E22 | all | `ModelSerializer::load` | No size checks: a damaged model file wrote out of range. | Layer count, sizes and activations checked. |

## F. Effect on the Mode 6 results

None of E1–E22 changes the Mode 6 pipeline numerically: GE Studio uses DE + L-BFGS, its own loader path
(`loadBundle`), `NSProblem` and the reference models of `Baselines.cpp`, none of which had these errors; the
datasets contain no missing values. The only effect is that the random number streams of the optimizers are
consumed differently (start point drawn in `setup`), so runs are not bit-identical to v14 — statistically the
same. The v14 conclusion stands: re-run `paper_main` with v14 or later.

## G. Known limitations (unchanged, documented)

* Mode 4 "Phase 2 = GE grammar" draws an independent bound command per weight at random (random search over
  per-weight factors), not a grammar derivation; Mode 6 is the grammar-based method.
* Modes 3–5 choose and accept on a validation split of the train file that is also the reported validation
  error (optimistic for the chosen configuration); the test-file rows of Modes 4–5 are the unbiased figures.
* The Wilcoxon tab shows unadjusted pairwise p-values (the Friedman tab gives the family-wise Nemenyi test).

## Verification (v15)

* `grad_check`: gradients and losses as in v14; Trainer (SGD, Adam) and RBF beat the trivial predictor on
  pima, wine, segment and housing; `.tst` clicked → roles correct; CSV with an empty field keeps the row;
  model save/load gives identical predictions; a NaN input equals the train-mean input.
* Optimizer harness (20-D, bounded, optimum inside and outside the box, budgets 10 / 41 / 100 / 20 000):
  every optimizer respects the budget and the bounds, and its reported best equals the re-evaluated best.

---

# Third pass (v16): the places that produce or transform numbers

Re-checked, with independent recomputation where possible: the overall tables and hypothesis tests of
`GEReport` (fold means, common folds, relative difference, Holm, which datasets enter a test), the raw-file
writer / reader and resume (`GEBatch`), the parameter order of the grammar's bounds against `flatten`, the
constructed-feature parser and evaluator, the data files themselves, and every reference model.

## H. Errors found and fixed

| # | Where | Error | Effect | Fix |
|---|-------|-------|--------|-----|
| H1 | `GEStudioWorker` (`makeSeed`) | The network trainings of Mode 6 (DE population, L-BFGS) took **clock-based seeds**, although the batch engine sets a seed per job (`seed·1000 + fold`). The GE search itself was seeded, the trainings were not. | Two runs of the same job gave different numbers (paper_main is a single realisation that cannot be reproduced). No bias. | With a configured seed every training seed comes from a stream seeded by (seed, dataset): verified, two runs of the same jobs (wine, housing; standard, GE #1, ADAM, NEAT, kNN, RF) give **bit-identical** rows. |
| H2 | `Baselines.cpp` | The random stream of each reference model was seeded with `qHash(id)`; Qt does not guarantee that value across versions / platforms. | Possible run-to-run / machine-to-machine differences of the stochastic baselines. | Fixed FNV-1a hash of the id. |

## I. Checked and correct (numbers recomputed)

* Reference models against scikit-learn on 8 datasets × 10 folds (same grids, own 80/20 selection split):
  kNN, logistic / ridge and RBF-SVM agree within the noise of the hyper-parameter choice
  (e.g. segment: kNN 2.94 / 2.94, linear 5.07 / 5.02, SVM 3.25 / 2.99; abalone: 4.91 / 4.93, 5.10 / 5.07,
  4.55 / 4.54; stock SVM 0.507 / 0.525). RF and ADAM agree as reported in v14.
* SMO solver (WSS2, LIBSVM formulation for SVC and ε-SVR), random-forest split criteria (Gini through
  Σc²/n, SSE through s²/n), NEAT (speciation, crossover, acyclicity), GNN phases.
* `flatten` order (per layer: weights row-major, then biases) matches the per-layer bound vectors of `trainOne`.
* Feature expressions: variables x1..xN map to the raw input columns, protected log / sqrt / division,
  clipping to the train 1st–99th percentiles (the test file uses the train percentiles).
* Raw results: 10 significant digits; resume replaces a GE run as a whole (ablations keep their phenotype);
  a stopped run is never written.
* Overall tables: per dataset the mean over the folds that every method has, then over seeds; Friedman / ranks
  / Wilcoxon / rd only on datasets with every method; hypotheses only on confirmation datasets with all folds;
  Holm step-down with monotonicity.
* Data: 117 datasets × 10 folds, one output column each, no ragged rows, no missing values.

## J. Notes for the article

* The kNN "train error" includes each training sample as its own neighbour (0 % for k = 1); only test errors
  enter the tables.
* The geometric ratio is computed over the datasets where both errors are positive (a 0 % error has no
  ratio); the relative difference rd and its Wilcoxon test use all datasets.
* Results of v14/v15 and v16 differ only by the random streams (H1, H2); v16 makes every run reproducible.

---

# Fourth pass (v17): the training losses, the line searches, the single-point methods and the data layer

## K. Errors found and fixed

| # | Mode | Where | Error | Effect | Fix |
|---|------|-------|-------|--------|-----|
| K1 | all | `NSProblem` (fitness, `evalAndGrad`, `evalAndGradParallel`, loss on the test file), `Trainer` | **Binary cross-entropy was computed on the clipped probability** `-log(max(1e-7, p))` and its gradient was `dL/dp · sigmoid'(z)`. For a saturated output (\|z\| > ~16, common with weights in ±10) the loss of a confidently **wrong** sample is capped at 16.1 and its gradient is ~1e-11 instead of ~1: exactly the samples that should be corrected contribute nothing, and L-BFGS / CG / LM (Mode 6 polish, adaptive training, Modes 1-5) stop on a point that only looks converged. The multi-class loss was clipped the same way while its gradient was not, so loss and gradient disagreed for those samples. | Mode 6 (search fitness, L-BFGS phase, adaptive rows) and every binary / multi-class training in Modes 1-5. Regression is unchanged. | Losses from the logits: BCE = max(z,0) − t·z + log(1+e^−\|z\|), CCE = logsumexp(z) − z_t; gradient w.r.t. the output pre-activation p − t (new `Layer::backwardFromZ`). Identical to the old values when no output is saturated. Probability-space losses of averaged ensemble predictions (`predLoss`, `predLossK`) are unchanged. |
| K2 | 1, 2 | `NSProblem::evalAndGrad` | Switched the network to training mode, so with dropout > 0 the gradient of L-BFGS / BFGS / CG / LM belonged to a randomly thinned network while `evaluate()` returned the loss of the full one; the line searches compared incompatible quantities. | Mode 1/2 with dropout and a gradient-based meta-optimizer. | Training mode off (dropout belongs to the mini-batch `Trainer` only). |
| K3 | all | `LBFGS::lineSearch`, `ConjugateGradient::lineSearch` | When no Armijo step was found the last trial step (2^-25 ≈ 3e-8) was returned and **taken** although it did not decrease f; the iterate drifted to worse points and the same 25 evaluations were repeated in every later iteration until the budget ran out. | Wasted budget (Mode 6 polish / adaptive phase, Mode 4 final polish) and a current iterate that could only get worse once the search failed. | Failure returns 0: the curvature memory is dropped and steepest descent is tried; if that fails too the method stops evaluating (the drivers already stop on an iteration that evaluates nothing). |
| K4 | 1-6 | `SimulatedAnnealing`, `NelderMead`, `MetaTrainer` | One iteration made ONE move, while every driver runs `iterations` iterations of a budget `iterations × population` (MetaTrainer: `iterations × 20`). SA used 1/population of its budget and its cooling, derived from that budget, never went below t ≈ 0.7 (no annealing); Nelder-Mead likewise used a small fraction. | SA / Nelder-Mead rows were not run with the budget they reported. | One iteration = `population` evaluations (as the population methods); MetaTrainer gives them population 20. |
| K5 | all | `NelderMead::init` | The axis vertices were built from `best_x_`, which changes when a vertex improves, so later vertices were moved along two axes; a start coordinate on the upper bound produced a flat (degenerate) simplex. | Poorer start simplex. | Vertices from the fixed start point; step inward at the upper bound. |
| K6 | 3-6 | `NSProblem::valLoss`, `GEStudioWorker::trainOne`, Modes 3-5 | With no validation rows (tiny data set) the validation loss was 0 (a perfect score) and the validation error (1 − (−1))·100 = 200 %. | Early stopping / GE fitness / architecture choice on a non-existent validation set. | −1 ("none"); the callers fall back to the training figures. |
| K7 | 4 | `EvolBoundsWorker::runPhase2Opt` | Refined half-width f·0.6·\|w\|: for f < 1.67 the refined box excluded the Phase-2 weights themselves, so Phase 3 started from a clipped, worse network. | Mode 4 with a small f. | Half-width at least \|w\|. |
| K8 | 4 | `EvolBoundsWorker::runPhase2GE` | The GE bound programs were scored on all rows, including the validation rows that later judge the result. | Optimistic validation figures of Mode 4 (GE phase). | Scored on the training part of the same split (seed 42). |
| K9 | 5→1 | `NetworkPanel::applyArchitecture` | Compared `itemData()` of a combo whose items have no data, so the activations found by the architecture search were **never** applied (every layer stayed ReLU). | "Use this architecture" built a different network. | Compares the item text. |
| K10 | all | `CompanionFinder` | The pairs `_train/_test`, `-train/-test` were matched against the whole file name, so `iris_train.csv` never found `iris_test.csv` (and its role was unknown). | Test file of CSV data never loaded automatically. | Also matched on the name without its extension. |
| K11 | all | loaders, `Dataset` | A row with a missing target (NaN) was kept: NaN propagated into every loss, and inserting NaN into the `std::set` of `uniqueOutputValues()` breaks its ordering (undefined behaviour in the task detection). | Corrupted training / task detection for files with an empty target. | Such rows are removed (reported in the log). |
| K12 | all | `Dataset::computeStatistics` | Variance as E[x²] − E[x]² (catastrophic cancellation for large values with a small spread). | Wrong standard deviation in the statistics. | Two-pass variance. |
| K13 | all | `NeuralNetwork::normalize` | Indexed past the normalisation arrays when a model has fewer entries than inputs. | Out-of-range read. | Such columns pass through. |
| K14 | 1 | `NetworkPanel`, `PredictionPanel`, `PreTrainWorker` | Info line printed the number of distinct target values as "outputs" (e.g. "506 outputs"); a binary prediction showed 0/1 instead of the class label of the data; the Phase 2 title of Mode 5 lost the optimizer name (`%2` used twice). | Display only. | Fixed. |

## L. Effect on Mode 6

K1 and K3 change the numbers of every classification row of Mode 6 that trains a network (search fitness, L-BFGS
polish, adaptive / ensemble / SAGE rows); regression rows change only through K3 (fewer wasted evaluations).
The reference models of `Baselines.cpp` already used exact logit losses and are unaffected. Results produced
before v17 should be re-run for the classification datasets.

## M. Checked and correct (no change)

* All 16 optimizers on shifted sphere / ellipsoid / Rastrigin (20-D, 20 000 evaluations, 3 seeds): every one beats
  random search by orders of magnitude, respects the bounds, and its reported best equals the re-evaluated best.
* Wilcoxon (exact / tie-corrected normal), Friedman (tie correction), Nemenyi (studentized range for any k), Holm.
* SMO / RF / kNN / logistic-ridge / NEAT / GNN reference models; GE grammar, mapping, constant folding.

## N. Known limitations (unchanged)

* Task detection from the target column cannot tell class labels stored as equally spaced fractions (glass:
  0, 0.167, …) from regression targets with few levels (SN, pl, QU); the experiment files declare the task.
* `PSO` accepts a particle move only when it improves the particle (greedy, "DE template"), a deliberate
  variant of the canonical PSO.
* Mode 4 "Phase 2 = GE grammar" only produces bounds; its "Phase 1+2" row shows the Phase-1 network.

---

# Fifth pass (v18, NeuralStudio 1.0.6): Modes 1-5 only

Constraint of this pass: nothing that Mode 6 computes may change (v4 experiments are running). No file of the
Mode 6 computation path (NSProblem, optimizers, NeuralNetwork, loaders, grammar, GE worker / batch / report,
reference models, statistics) was edited; `ge_batch` builds from unchanged sources. Mode 6 got only the
merged **Log** tab (the former Log and Batch tabs; duplicate "Stop requested" / error lines removed).

## O. Errors and inconsistencies fixed

| # | Mode | Where | Problem | Fix |
|---|------|-------|---------|-----|
| O1 | 3 | `PreTrainWorker` | **Unequal budgets**: the baseline got the Phase-2 budget only, the pre-training route Phase 1 + 2 (+ 3), i.e. 2-3× more evaluations; "pre-training helps" could just mean "more evaluations help". | Option *Baseline with equal total budget* (default on): the baseline gets the evaluations of all phases. Off = former protocol. |
| O2 | 4 | `EvolBoundsWorker` | Same: Standard got the Phase-2 budget (40 000 evaluations with the defaults) against ~242 000 for Phases 1 + 2 + 3 + BFGS. | Same option (default on); the GE phase is counted with its expected number of scoring evaluations. |
| O3 | 3, 4 | workers, panels | When Phase 1+2 was worse than Standard, Phase 3 continued from the **Standard** weights but the row was still called "Phase 1+2+3". | Row labelled "Standard + Phase 3 (Phase 1+2 rejected)"; with N runs / several datasets the column keeps its name and a row "Phase 3 started from" says how often it happened. |
| O4 | 4 | `EvolBoundsWorker` | With Phase 2 = GE grammar the "Phase 1+2" row is the Phase-1 network (GE only chooses bounds). | Labelled "Phase 1+2 (GE bounds, Phase-1 weights)". |
| O5 | 5 | `ArchSearchWorker` | The **test file was ignored**: the only error shown was the validation error on which the search selected (optimistic). | Every trained architecture is also scored on the companion test file (never used by the search); new column *Test Err* (and mean ± std over runs). |
| O6 | 5 | `ArchSearchWorker` | Every architecture got the same few thousand derivative-free evaluations whatever its number of weights: large networks were compared half-trained and the search favoured tiny ones. | Bound-respecting L-BFGS polish after the inner optimizer (*L-BFGS polish*, default 400 evaluations, 0 = off). |
| O7 | 5 | `ArchSearchWorker::run` | No exception handling: an unreadable dataset file **terminated the application**. | Reported as an error. |
| O8 | 5 | `ArchSearchPanel` | Numeric columns sorted as text ("10.5" before "9.2"); MSE printed with 4 decimals (0.0000). | Sort by value; MSE with 4 significant digits. |
| O9 | 2 | `BatchRunner` | "Val error %" was the accuracy of the **last** epoch while early stopping restores the **best** epoch (another network); the regression validation metrics were never filled (Val MAE / RMSE / R² of the statistics tabs were empty). | Validation metrics of the final network on the trainers' validation rows (same split as Trainer / NSProblem). |
| O10 | 2 | `BatchRunner` | A training error was thrown from inside a Qt slot (through signal emission, unsupported by Qt). | Recorded and raised after `run()`. |
| O11 | 2 | `BatchRunner` | Dataset name = `baseName()`: "a.v1.train" and "a.v2.train" were both "a" and were averaged together. | `completeBaseName()`. |
| O12 | 2 | `WilcoxonTab` | k optimizers → k(k-1)/2 pairwise tests with **unadjusted** p-values (28 tests for 8 optimizers: about one false "significant" pair expected). | Holm-adjusted p-values (shown next to the raw ones); stars, colours and plot brackets use the adjusted value. |
| O13 | 2 | `BatchPanel` | Comment "only Adam + SGD checked by default", code checked all six gradient methods. | Adam + SGD. |
| O14 | 1 | `CrossValidator`, `TrainingPanel` | Stop did nothing during cross-validation (and was checked only between folds); the std over folds used n instead of n − 1; an interrupted fold was reported as a result. | Stop reaches the running fold; sample std; a stopped CV reports only the completed folds and says so. |
| O15 | 1 | `TrainingPanel` | Nothing showed that a population method had e.g. 9 600 evaluations for 2 700 weights (≈ random search). | Banner with weights, budget and evaluations per weight (warning below 20). The LR spin box that was created twice is created once. |
| O16 | 1 | `DataPanel` | The task every mode uses was not shown; a "class balance" was shown for regression targets with few values (SN, pl); class labels with one decimal (0.167 → 0.2). | Detected task in the info bar; balance only for classification; labels as stored. |
| O17 | 3, 4 | panels | Regression errors printed with two decimals (an MSE of 0.00035 showed as "0.00"). | Four significant digits for regression. |
| O18 | 3 | `PreTrainNNCPanel` | Network, validation split and Iw stayed editable during a run (they do not affect it). | Locked while running, as the phase boxes. |

## P. Mode 6
* Log and Batch tabs merged into one **Log** tab (both pages).
* Computations unchanged (no source of `ge_batch` / the GE worker / the reference models was edited).
