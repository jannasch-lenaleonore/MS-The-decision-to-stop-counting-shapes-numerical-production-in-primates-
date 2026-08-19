# A drift-diffusion model of the stop decision in sequential number production

MATLAB code and behavioural data for the modelling in

> Seidler, L. E., Jannasch, L. L., Westendorff, S. & Nieder, A.
> *The decision to stop 'counting' shapes numerical production in primates.*

Two rhesus macaques produced an instructed number of hand movements and ended
each sequence themselves. This repository fits and evaluates the stochastic
**stopping model** of that termination decision, compares it against the two
classical mental-number-line accounts, and produces the model-derived figure
panels, tables and numbers reported in the manuscript.

## Quick start

```matlab
>> run_all           % every figure and every table, from the fitted
                     % parameters shipped in results/ (under a minute)
>> run_all('refit')  % refit everything from the raw data first, overwriting
                     % results/ (several hours)
```

The default reuses `results/` because those are the fits the manuscript
reports. `'refit'` regenerates them from the behavioural data; the fitting is
stochastic, so a refit of the RT model in particular settles on a different
optimum and shifts Table 1.

`run_all.m` sets up the paths itself, so it works from any working directory,
and it prints a banner before each stage. Most of the manuscript's numbers are
**printed, not plotted**, so capture the log:

```matlab
>> diary('run_all_log.txt'); run_all; diary off
```

Every stage is also an ordinary script that can be run on its own — see the
order in `run_all.m`.

## Requirements

- MATLAB R2023a or later with the Statistics and Machine Learning Toolbox
  (`quantile`, `signrank`, `corr`, `fminsearch`, `boxplot`).
- No other dependencies, no toolboxes beyond that, nothing to install.

The behavioural data are included in `data/`: `bhv_data_tbl_m1.mat` (Monkey 1,
36 sessions) and `bhv_data_tbl_m2.mat` (Monkey 2, 40 sessions), each holding a
`dataTable` with per-session `RespMat`, `RT_conf` and `RT_err_conf`. In
`RespMat`, column 2 is the target numerosity, column 3 the produced number and
column 5 the outcome (`0` correct, `6` error); all-`9` rows are padding and are
dropped on load. `RT_conf` and `RT_err_conf` hold the confirmation reaction
times of that session's correct and error trials, in order.

## What produces what

Every file in `figures/` is one manuscript panel. The numbers quoted in the
text are printed to the console by the script named here.

| Manuscript item | Produced by | Output |
|---|---|---|
| Fig. 3E — median RT ± IQR per response offset, per monkey | `diffusion_model_fig4.m` | `figures/fig3E_rt_per_offset.pdf` |
| Fig. 4A — example decision-variable trajectories | `diffusion_model_fig4.m` | `figures/fig4A_model_traces.pdf` |
| Fig. 4B — what the three accounts predict | `schematic_row2_MS.m` | `figures/fig4B_model_predictions.pdf` |
| Fig. 4C — the three fitted models on one release axis | `model_comparison_fig4.m` | `figures/fig4C_model_fits.pdf` |
| Fig. 4D — per-trial log-likelihood per animal | `model_comparison_fig4.m` | `figures/fig4D_loglikelihood.pdf` |
| Fig. 4E — held-out log-likelihood per session | `model_comparison_fig4.m` | `figures/fig4E_crossvalidation.pdf` (left panel) |
| Fig. 4F — median RT ± IQR, monkeys vs model | `diffusion_model_fig4.m` | `figures/fig4F_rt_model_vs_monkey.pdf` |
| Fig. 4G — RT quantiles per offset | `diffusion_model_fig4.m` | `figures/fig4G_rt_quantiles.pdf` |
| Table 1 — G² per offset, ΔG² and its 95% CI | `compare_rt_fit_MS.m` | printed |
| Results — RT trend and asymmetry (median ρ, both *p*) | `diffusion_model_fig4.m` | printed |
| Results — per-trial log-likelihoods per animal, % of explainable deviance | `model_comparison_fig4.m` | printed |
| Results — cross-validated model comparison (59/76, 70/76, ΔL, *p*) | `model_comparison_fig4.m` | printed |
| Results — pooled vs per-animal fits | `cv_pooled_vs_animal_MS.m` and `model_comparison_fig4.m` | printed |
| Methods — fitted parameters *v*, σ_z and their log-likelihoods | `fit_diffusion_mle_MS.m` | printed, saved to `results/params_perf.mat` |
| Methods — number-line parameters *b*, σ and their log-likelihoods | `fit_gaussian_mle_MS.m` | printed, saved to `results/params_gauss.mat` |
| Methods — session SD 7.9%, design effects, χ² and G² heterogeneity | `session_variability_MS.m` | printed |
| Methods — held-out means ± SD, Pearson *r* between models | `model_comparison_fig4.m` | printed |

`figures/diagnostics/` holds the three figures `compare_rt_fit_MS.m` draws of
the RT comparison. They are the visual form of Table 1 and are not manuscript
panels; `figures/` itself contains manuscript panels only.

**Not in this repository.** Fig. 1 (task schematic), Fig. 2 (response
functions by stimulus format), Fig. 3A–D (RT by numerosity, correct vs error)
and Supplementary Fig. 1 (performance by format, condition and temporal
arrangement) come from the behavioural analysis of the full session records,
not from the model. This repository is the code behind the modelling, which is
what the manuscript's Data and Code Availability statement points to.

## Pipeline

| Stage | Script | Writes | Runtime |
|---|---|---|---|
| 1 | `fit_diffusion_mle_MS.m` | `results/params_perf.mat` | ~80 min |
| 2 | `fit_gaussian_mle_MS.m` | `results/params_gauss.mat` | seconds |
| 3 | `cv_models_MS.m` | `results/cv_results.mat` | hours |
| 4 | `cv_pooled_vs_animal_MS.m` | `results/cv_pooled_vs_animal.mat` | hours |
| 5 | `model_comparison_fig4.m` | Fig. 4C, 4D, 4E + tables | seconds |
| 6 | `session_variability_MS.m` | prints only | seconds |
| 7 | `schematic_row2_MS.m` | Fig. 4B | seconds |
| 8 | `diffusion_model_fig4.m` | Fig. 3E, 4A, 4F, 4G + RT tests | ~40 s |
| 9 | `fit_diffusion_rt_MS.m` | `results/params_rt.mat` | ~20 min |
| 10 | `compare_rt_fit_MS.m` | Table 1 + diagnostics | ~15 s |

Stages 5–8 and 10 only read the `.mat` files, so with `results/` populated (as
shipped) a bare `run_all` reproduces every figure and every table in under a
minute. Stages 1–4 and 9 are the ones `run_all('refit')` adds; their runtimes
are the ones quoted in each script's own header.

Two helper functions in `helpers/` are shared by the RT scripts:
`run_diffusion_sim.m` (one vectorised sweep of the model) and `compute_g2.m`
(the G² objective). The scripts add `helpers/` to the path themselves.

## The model

Each numerosity step is one second of a bounded accumulation process. The
animal "counts" by accumulating evidence across discrete steps and stops when
the decision variable (DV) first crosses a fixed upper bound `θ = 1`:

- **Drift** is constant within a trial (`drift_sigma = 0`, i.e. `fix_drift =
  true`, the manuscript's variant). The code also supports Ratcliff's
  across-trial drift variability `η`, drawn per trial from
  `N(drift_mean, drift_sigma)`.
- **Starting point** is a folded normal `|N(0, sigma_z)|`; a reflecting lower
  border at 0 keeps the DV non-negative. The accumulator carries over across
  counting steps.
- **Within-trial noise** is Gaussian (`noise = 0.1`, step `dt = 0.01`).
- The **response offset** is the step at which the bound is crossed relative to
  the target (`stop_k - target`), binned to `-2 … +2`. Trials that never stop
  within range fall into a "no-stop" category that carries an observed count of
  zero, so producing them costs likelihood.
- **Reaction time** is the time since the last step onset, scaled to ms by a
  factor drawn from the task's timing protocols
  (`{200, 500, 800, 1100}`, `{200, 200}`, `{800, 400}`).

Offsets are pooled over target numerosities 3 and 4 across both monkeys:
16 537 of 16 960 trials, the rest being errors outside `-2 … +2`.

## The comparison models

`model_comparison_fig4.m` compares three accounts of the response distribution
over the offset bins −2:+2. Each emits one probability per produced number
given the target, so all three are scored with the same multinomial
likelihood, on the same trials, with the same optimiser.

| Model | k | Number line |
|-------|---|-------------|
| Stopping (accumulation to bound) | 2 | — (target-invariant by construction) |
| Linear number line (shared) | 2 | `mu_n = n + b`, one `sigma` in count units, so `sigma_3 = sigma_4` |
| Log number line (shared) | 2 | `mu_n = log(n) + b`, one `sigma` in log units, so the distribution is symmetric on a log scale and widens with `n` |

All three are matched in flexibility: two free parameters, a location and a
width, that must account for both targets at once. Because the likelihood is
multinomial the predicted probabilities are normalised, so a free peak
amplitude would scale all categories equally and cancel — it is not
identifiable and is deliberately not used. In every model the proportion of
correct trials is the offset-0 probability, i.e. a *prediction* of location and
width, never a fitted quantity.

For the two number lines the probability of producing `r` is the **height** of
the curve at `r` (at `log(r)` on the logarithmic line), normalised over the
possible responses — the reading standard in numerical cognition, in which the
curve *is* the response distribution. The alternative, integrating the curve
between the half-integer edges around `r`, carries a Jacobian that on a log
scale down-weights larger `r` by roughly `1/r`; it fitted these data clearly
worse at equal parameter count and is not used anywhere. The height rule lives
in `predict_gauss` in `fit_gaussian_mle_MS.m` and is duplicated as
`gauss_predict` in both cross-validation scripts — **the three copies must stay
identical**.

`fit_gaussian_mle_MS.m` also fits numerosity-specific number lines (`mu`,
`sigma` refitted per target, 4 parameters each) and `cv_models_MS.m` still
scores them, but they are deliberately not compared in the manuscript: the
saturated ceiling drawn on Fig. 4E already bounds what any numerosity-specific
account could reach, without introducing a fitted competitor that would have to
be penalised for its extra parameters.

## Cross-validation and the unit of analysis

Trials within a session are not independent. `session_variability_MS.m`
quantifies this: the session-wise proportion correct has a between-session SD
of 0.079 against the 0.038 expected for independent trials, a design effect of
~4.3 (`deft` = 2.08), and the sessions are heterogeneous well beyond chance
(χ²(75) = 366.0; G²(300) = 1585.3 for the full offset distribution). The
**session is therefore the unit** — for holding out, and for the statistics.

Cross-validation is **leave one session out**: 76 folds, one per session, each
model refitted on the other 75. There is no random partition to choose and
nothing to average over, so the result is deterministic and the paired test has
exactly one observation per session.

Scores are read as a **held-out log-likelihood per trial**: the log of the
probability the model assigned to the response that actually occurred, on the
session it was not fitted to, divided by that session's trial count so that
long and short sessions weigh equally. Its exponential is the geometric-mean
probability given to the observed response. Two references put it on a scale —
a uniform model over the five observed categories (0.200) and the saturated
ceiling that knows each target's exact response frequencies (0.311) — and
`model_comparison_fig4.m` prints each model's position between them as
"% of range".

Sessions differ from each other far more than the models differ from each
other, and they do so in step: per-session scores correlate at *r* = 0.997
between the stopping model and the linear number line. That is why the
comparison is made **within** session, and why the reported statistic is a
signed-rank test over the 76 paired per-session differences.

## Reproducibility

Every script seeds the RNG (`rng(42)`, with fixed seeds for the final
simulations and for the bootstrap), so the results are reproducible run to run.
The one quantity carrying Monte Carlo noise is the stopping model's
log-likelihood, since it has no closed form; it is evaluated at 500 000
simulated trials and its across-seed SD (≈ 5.8 nats on ≈ 19 768) is printed
next to it.
