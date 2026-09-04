# Interpretable UK Credit Risk: Recovering Macro-by-Micro Interactions

This project investigates how borrower characteristics and UK macroeconomic conditions combine to influence next-quarter loan default risk, and tests whether interpretable models can recover these interaction effects accurately and keep their predictions, calibration and explanations reliable when the economy is put under stress.

Real lending data cannot answer this question cleanly, because nobody knows the true interaction structure hiding inside it. So the project builds its own answer instead. A synthetic quarterly loan panel is generated with a known, controllable relationship between borrower attributes and 8 real UK macroeconomic series. Because the true relationship is written by the researcher, every model's recovery of it can be measured directly rather than assumed.

## The core idea: a dual-truth design

2 independent sources of ground truth are used, and both are compared against the same 3 glass-box models.

The **planted arm** injects 5 hidden interactions directly into the default hazard as fixed product terms (for example, income multiplied by wage growth, weighted by a chosen coefficient). These 5 pairs, their signs and their strengths are sealed away from the modelling stage and only revealed for grading. This arm is the exact instrument. It proves whether the recovery machinery works at all, because the correct answer is known down to the decimal.

The **emergent arm** plants nothing. Instead, the macroeconomic state is wired into each borrower's affordability accounting quarter by quarter. For example CPI raises essential spending, Bank Rate reprices variable debt, unemployment triggers income shocks, and the default hazard is a convex function of the resulting squeeze. The same 5 interactions emerge from this mechanism on their own, without any product term being written into the code. This arm answers a harder and more honest question: does a model recover an interaction that arises from real economic structure, not just from an artificial injection?

3 interpretable glass-box models are compared, which are Logistic Regression with explicit interaction terms, a Generalised Additive Model with tensor smooths, and an Explainable Boosting Machine restricted to the same candidate pairs. All 3 see an identical 35-pair candidate grid (7 borrower axes crossed with 5 macro axes) with no truth labels attached, so none of them is told in advance which 5 pairs are real.

## Project structure

```
credit_dgp.py              the data-generating process (the simulator)
pipeline_utils.py          shared preprocessing, model wrappers, and the interaction ruler
macro_data/                8 real UK macro series (ONS, BoE, BIS), bundled locally
ground_truth_params.json   public hazard parameters (main effects, reference moments)
synthetic_credit_panel.csv the null-condition reference panel written by Stage 1

stage1_dgp.ipynb                    Stage 1: protocol and the sealed benchmark
stage2_data_prep_eda.ipynb          Stage 2: preprocessing and exploratory analysis
stage3_modelling_interaction.ipynb  Stage 3: model fitting and interaction recovery
stage4_regime_stability.ipynb       Stage 4: stress testing and explanation stability
stage5_model_comparison.ipynb       Stage 5: synthesis and research answers

outputs/                   every saved result file (JSON), plus the dev/val/conf reference CSVs Stage 2 exports for provenance
figures/                   plot data and a consolidated summary table
requirements.txt           pinned package versions
```

Each notebook is self-contained. All 5 only import `credit_dgp.py` and `pipeline_utils.py`, and every heavy computation is checkpointed: a single `REGENERATE` flag near the top of each notebook decides whether the cell reloads a saved result in seconds or reruns the real experiment from scratch. This means the notebooks can be opened and read without waiting for anything to recompute, while still being fully reproducible if `REGENERATE` is switched on.

## Stage 1: Synthetic Data Generation and Experimental Protocol

This stage sets up the benchmark before any modelling happens. 5 hidden active pairs are chosen from the 35-pair candidate grid using a seeded random draw, so the choice is not hand-picked or favourable. Their identity, signs and the interaction strength ladder are written to a sealed private file that later stages are not allowed to peek at during fitting.

A public, truth-free registry of all 35 candidate pairs is published alongside it, listing pair names only, with no indication of which ones are real. Stage 1 also generates a null-condition reference panel (no interaction switched on at all) and exports the true hazard parameters that later stages use to reconstruct the oracle. Using a null panel here matters because it stops the exploratory analysis in Stage 2 from accidentally revealing which interactions were planted.

## Stage 2: Data Preparation and Preprocessing

The reference panel from Stage 1 is turned into a model-ready feature matrix here. Categorical variables are one-hot encoded, continuous drivers are z-scored, and the 35 interaction product columns are built. The split into development, validation and confirmation partitions is both loan-disjoint and time-blocked, so no loan appears in 2 partitions and no future quarter leaks into training.

The scaler is fitted on development data only, then applied unchanged to validation and confirmation, which keeps the later stress tests honest. This stage also derives a feature marking whether a loan began the quarter already in arrears, since the true hazard depends on this and it needs to be visible to every model that follows. A short exploratory section checks macro collinearity and compares the generated population against real UK plausibility anchors, mostly as a sanity check rather than a core result.

## Stage 3: Modelling and Interaction Recovery

This is where the 3 models are actually fit and graded. Hyperparameters are tuned once by validation log loss, never by how well a setting happens to recover the interaction, which keeps the later recovery numbers a fair test rather than a tuned-to-win result.

Detection is calibrated against 10 interaction-free panels first, so a model's claim of finding something real is judged against its own noise floor rather than an arbitrary cutoff. The main experiment then runs 5 seeds across 4 interaction strengths (none, weak, moderate, strong), and for every condition each model is scored with the same ruler: Friedman's H-squared for interaction strength, and a coefficient-equivalent b-hat for sign and magnitude. This ruler is model-agnostic, since it only needs a predict function, which is what lets logistic regression, the GAM and the EBM be compared on one shared scale.

The same ruler is then applied to the emergent arm, comparing each model's whole interaction fingerprint across all 35 pairs against the economically-grounded truth. Inline figures at the end of the stage show the recovered interaction surfaces, the active-versus-decoy separation, the null thresholds and the calibration behaviour.

## Stage 4: Regime Stability and Stress Testing

The models fitted on the emergent, structural world are frozen here and rolled forward through economic scenarios they were never trained on. A benign and a severe 16-quarter macro path are constructed, and the same simulated borrowers are pushed through both, so any difference in predicted default is attributable to the macro shock and not to who happens to be in the loan book. A fixed-composition variant is also run, which keeps the exact same borrowers and only changes the macro inputs, and this is compared against the population-consistent version to see how much a naive stress test overstates the shock.

3 real historical windows are tested too: the 2008 financial crisis, the COVID-19 shock, and the 2022 inflation spike. The financial crisis window sits entirely inside the training period under the temporal split, so scoring it would be testing the models on data they already learned from. It is excluded from the results and the reason is stated plainly, rather than silently dropped or, worse, reported as if it were a fair test.

Finally, this stage asks a question that goes beyond prediction: does each model's explanation stay the same when the economy moves? The interaction fingerprint measured in Stage 3 is recomputed under each stress regime and compared back to the benign baseline, averaged over the same 5 seeds used for the forward scenarios so the result carries a proper variance estimate rather than resting on a single draw.

## Stage 5: Synthesis, Model Comparison and Governance

The final stage does no new computation of its own. It reads every result file written by the earlier stages and pulls them into one consolidated table, so a reader can see discrimination, calibration, recovery magnitude, structural agreement, stress error and explanation stability side by side for each model. It then writes out plain-language answers to the 3 research questions and states the project's governance position: what is transparent, what is reproducible, and where the claims stop (this is methodological recovery inside a UK-informed simulator, not a claim about real UK or Lloyds coefficients).

## Headline findings

No single model wins on every axis, which is the central result of the comparison.

- The Explainable Boosting Machine gives the best prediction, the best calibration and the best structural agreement with the emergent truth, but recovers the smallest share of the planted interaction magnitude.
- Logistic regression and the GAM recover the planted magnitude almost exactly, but predict slightly less accurately overall.
- Under real historical stress all 3 models transport well, with small gaps between predicted and realised default. Under a severe scenario that pushes macro conditions beyond anything seen in training, the models diverge and fail in opposite directions: the simpler models over-predict risk, while the EBM under-predicts it.
- The same pattern shows up in the explanations. Logistic regression keeps almost the same interaction story under stress as it does normally. The EBM's explanation shifts the most, meaning the model that predicts best is also the one whose reasoning becomes least trustworthy exactly when trust matters most.

## Running the project

Install the pinned dependencies from `requirements.txt`, then open the 5 notebooks in order from Stage 1 through Stage 5. With `REGENERATE` left at its default of `False`, every notebook reloads its saved results and opens in seconds. Setting it to `True` in any stage reruns that stage's real computation from scratch, which takes from a few minutes up to a few hours depending on the stage, since Stage 3 and Stage 4 refit the models many times over.

The project needs nothing outside this folder to run. The macro data is bundled locally, and the 2 shared modules (`credit_dgp.py` and `pipeline_utils.py`) sit in the same directory as the notebooks.
