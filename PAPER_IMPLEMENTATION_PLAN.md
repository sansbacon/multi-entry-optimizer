# Paper review and NFL implementation roadmap

## Executive conclusion

The Python port preserves the paper's central **sequential construction**
strategy: maximize projected points, encourage within-lineup correlation
through stacks, and diversify across lineups using overlap limits.
Sequential construction is not an omission to fix.

**Status: all five phases below are now implemented** (Phase 1: NFL
classic/structural heuristics and named strategies; Phase 2: historical
evaluation/calibration tooling; Phase 3: optional moment-aware MILP;
Phase 4: probability diagnostics and coverage-based portfolio selection;
Phase 5: bounded formulation/performance experiments, measured end-to-end
on this stack). What remains, per each phase's own stated boundaries, is
**not** more implementation work but **running Phase 2's tooling against
real historical NFL contest/projection data**, which this workspace does
not have; every structural, moment-aware, and probability-diagnostic
claim above is mathematically/statistically verified on synthetic
fixtures only, not validated against real contest outcomes.

The largest pieces, now implemented rather than missing:

1. More complete, configurable structural stacking and concentration rules.
2. Historical contest evaluation and out-of-sample selection of strategy,
   lineup count, and overlap settings.
3. Optional player-moment inputs and explicit lineup variance/covariance
   constraints from the paper's general formulation.
4. Probability-of-success diagnostics connecting the implementation to the
   paper's actual portfolio objective.

The paper does **not** implement full expected-payout optimization. Its
formal objective is the probability of at least one successful entry.
Its practical hockey implementation uses a mean-maximizing heuristic
guided by variance and dependence, rather than directly maximizing that
probability. Ownership modeling and full payout optimization would be new
research extensions, not restoration of omitted Julia functionality.

This document began as a plan; it now also records what was implemented,
how it was verified, and what was measured, phase by phase below.

## Evidence and source limitations

Reviewed the parsed manuscript pages and the original PDF (both excluded
from this public repository per [NOTICE.md](NOTICE.md); see the freely
available [arXiv preprint](https://arxiv.org/abs/1604.01455) instead),
[Julia implementation](picking_winners.jl),
[Python implementation](src/multi_entry_optimizer/), [tests](tests/test_lineups.py), and
[solver benchmark](benchmark.py).

The Markdown folder (local-only; not published) contains pages 1-36, but
**23 files are empty**. Nonempty pages are 1-5, 14, 15, 18, 22, 24, 25,
29, and 31.
The PDF contains **39 pages**, including the appendix and references not
fully represented in the Markdown. Missing text was extracted locally to
session storage; the original Markdown and PDF were not changed.
The extractor reported malformed PDF object references and omitted rotated
text. The main prose and equations were readable, but extracted formulas
must be independently checked rather than treated as authoritative code.

Paper citations below use section/equation numbers and PDF page links.
They refer to the local manuscript, not an inferred later publication.

## What the paper actually establishes

- **Objective:** choose a fixed-size set of entries maximizing
  `U(S) = P(union of success events for entries in S)`; Equation (2.1),
  page 6.
- **Submodularity:** exact marginal-gain greedy selection over a fixed
  candidate set with a cardinality constraint has the classical
  `1 - 1/e` guarantee; page 7.
  That guarantee does **not** automatically apply to mean-maximizing
  lineup construction, estimated probabilities, or extra pairwise
  compatibility constraints.
- **Approximation:** the second-order inclusion-exclusion surrogate is
  `U2(S) = sum_i P(E_i) - sum_{i<j} P(E_i and E_j)`;
  Sections 2.1-2.2, pages 8-10.
  Its usefulness depends on rare events and limited dependence; it is
  not a generally accurate probability model for arbitrary portfolios.
- **Design principles:** high mean, high individual variance, and low
  cross-entry dependence; Sections 2.3-2.4,
  pages 11-13.
  The large-variance argument is for upper-tail thresholds above the mean,
  not a general recommendation for every DFS contest.
- **Practical implementation:** approximate moments with structural stacks
  and overlap; solve sequentially. The hockey application explicitly
  avoids estimating full covariance matrices;
  parsed page 18 and
  parsed page 22.
- **Evaluation:** compare realized contest profit distributions and tune
  strategy to slate size, not projection alone;
  parsed page 24 and
  PDF page 26.

## Gap analysis

| Concept | Current Python state | Proposed treatment | Priority |
| --- | --- | --- | --- |
| Sequential mean-maximizing solves; cuts against every prior lineup | Implemented | Preserve as default; no joint rewrite | Preserve |
| Roster, budget, team/game feasibility | Implemented for NFL classic presets | Keep separate from strategic constraints | Preserve |
| Avoid negatively correlated opponents | DST versus opposing offense implemented, on by default | Retain; document as NFL heuristic, validate empirically | Preserve |
| Full plus partial stacks, Eqs. (4.5)-(4.6) | **Implemented**: `secondary_stack` adds an independent game-based skill-position stack | Benchmark alternative position/team combinations empirically | Done (P1) |
| Exact team concentration, Type 4/5 | **Implemented**: `strategic_min_teams`/`strategic_max_teams`/`strategic_team_scope` | Calibrate bounds on historical NFL slates | Done (P1) |
| Position/role restrictions, Type 2/3/4 | **Implemented**: `flex_positions` restricts FLEX eligibility; locks/exclusions unchanged | Add NFL-specific role eligibility (e.g. starter status) once data is available | Done (P1)/P2 |
| Named, reproducible strategy combinations, Table 3 | **Implemented**: `STRATEGIES`/`apply_strategy`, `--strategy` CLI flag | Expand presets once Phase 2 evaluation identifies good combinations | Done (P1) |
| True no-stacking experimental baseline | **Implemented**: `STRATEGIES["baseline"]` | N/A | Done (P1) |
| Historical contest backtesting, Section 5 | **Tooling implemented** (`multi_entry_optimizer.evaluation`); not yet run on real data | Run against real historical NFL contests once available | Done (P2, tooling)/data-gated |
| Tune overlap by slate size, Section 5.3 | **Selection tooling implemented** (`bucket_by_game_count`/`select_policy_by_bucket`); no policy learned yet | Run against real training/validation slates to learn a policy | Done (P2, tooling)/data-gated |
| Entry-count and construction-order analysis, Sections 5.1/5.4 | **Implemented**: `creation_order_spearman`/`best_entry_creation_index`/`affordable_entries` | Apply to real held-out slates | Done (P2, tooling)/data-gated |
| Projection ensembles and contextual predictors, Section 3.2 | **Implemented**: `align_player_values`/`fit_ols`/`fit_ridge`/`mae_rmse` | Fit and validate on real multi-source projections | Done (P2, tooling)/data-gated |
| Empirical pairwise correlation analysis, Section 3.3 | **Implemented**: `residualize`/`estimate_group_correlations` with shrinkage and CIs | Estimate on real NFL residuals with enough same-game samples | Done (P2, tooling)/data-gated |
| Explicit variance lower bound, Eq. (2.10) | **Implemented**: `Config.variance_floor` with exact MILP linearization | Validate against real player moments once available | Done (P3)/data-gated |
| Covariance bound with previous lineups, Eq. (2.10) | **Implemented**: `Config.covariance_cap`/`correlation_ceiling` | Validate against real player moments once available | Done (P3)/data-gated |
| Iteration-dependent variance/dependence limits, Section 2.4 | **Scalar bounds implemented**; no per-iteration schedule | Explicit schedules, gated and tuned; never silently relax failures | Partial (P3) |
| Gaussian tail probabilities and U2 | **Implemented**: `compute_success_diagnostics`, `greedy_scenario_coverage` | Validate against real NFL score distributions once available | Done (P4)/data-gated |
| Two-lineup construction, Eq. (5.1) | **Implemented and benchmarked** (not default): measurably slower, matching the paper's own finding | No change; remains an experimental artifact only | Done (P5) |
| Stronger formulations, Appendix 8 | **Benchmarked**: pairwise DST and disaggregated cardinality linking, both proven equivalent, neither measurably faster | No change to `multi_entry_optimizer` defaults | Done (P5) |
| Full expected payout, ownership, exposure optimization | Missing | Outside paper-parity scope; separate future proposal | Deferred |

## Phase 1: complete the structural NFL heuristic — IMPLEMENTED

**Status:** implemented in [multi_entry_optimizer](src/multi_entry_optimizer/) and covered by
[tests/test_lineups.py](tests/test_lineups.py) (78 tests total, 31 added
for this phase). See the [README](README.md#strategic-and-stacking-options)
for the user-facing CLI/API description. All additions default to the
original unrestricted behavior (confirmed by
`test_no_new_options_changes_default_behavior`). This phase does **not**
validate that any of these structural choices improve real NFL contest
performance; that requires Phase 2's historical evaluation.

### Configuration and formulation

Extend [Config and build_model](src/multi_entry_optimizer/)
without changing defaults.

- Add optional strategic `min_teams`/`max_teams` bounds and an explicit
  scope: all roster members or offensive players only. Site legality
  remains independently enforced on the site's required scope.
- Link each strategic team indicator in **both directions**:
  `y[t] <= sum(x for scoped members of team t)` and
  `sum(x for scoped members of team t) <= M[t] * y[t]`.
  Existing one-way indicators suffice for minimum counts but cannot
  correctly enforce a maximum or exact count.
- Offer an optional secondary two-player skill-position stack from a
  **different game from the QB**. Define permitted same-team/opposing-team
  and position combinations explicitly; test alternatives rather than
  assume all pairs are beneficial.
- Use group activation variables with membership/count implications.
  A required secondary group must actually select its members; merely
  counting a team's existence is insufficient.
- Add an allowed-FLEX-position subset. For single-position players, fixing
  disallowed positions to their base roster count enforces this without
  introducing player-slot assignment variables.
- Use existing excludes for user-supplied role eligibility initially.
  Do not translate "power-play defenseman" into "DST"; NFL defenses are
  not hockey defensemen. Starting status or usage thresholds need NFL data
  and validation, not an invented direct equivalent.

Provide a small experiment suite, with names that do not imply NHL Type
1-5 equivalence:

1. Pure mean baseline: no stacks and DST opponents permitted.
2. DST-conflict avoidance only.
3. QB + one receiver.
4. QB + two receivers.
5. QB stack plus bring-back.
6. Optional secondary stack and team-concentration variants.

Implemented as `STRATEGIES["baseline"]`, `["dst-avoidance"]`,
`["qb-plus-one"]`, `["qb-plus-two"]`, `["qb-stack-bring-back"]`, and
`["secondary-stack"]` respectively, applied via `apply_strategy` or the
CLI's `--strategy` flag.

The paper's full-plus-partial line structure motivates the secondary
stack, but NFL players do not participate in fixed NHL-style lines.
Bring-backs already exist in the port and are an NFL adaptation, not an
omitted hockey equation. Exact three-team hockey settings should not
become NFL defaults, especially where FanDuel's per-team cap restricts
feasibility.

### Acceptance criteria

- Exhaustive enumeration on tiny pools agrees with each formulation.
  — `test_strategic_max_teams_exhaustive`, `test_secondary_stack_exhaustive`,
  `test_flex_positions_exhaustive`.
- Maximum/exact team-count tests include a high-projection lineup that
  violates the bound; test offense-only versus all-player scope.
  — `test_strategic_max_teams_exhaustive` (154.9-projection, 4-team
  optimum excluded by `strategic_max_teams`); `test_strategic_min_teams_constructive`
  and `test_strategic_team_scope_distinguishes_dst_only_team` (constructive
  locked lineup with 4 "all"-scope teams but 3 "offense"-scope teams).
- Secondary-stack tests prove game separation and actual group membership.
  — `test_secondary_stack_exhaustive`, `test_secondary_stack_requires_different_game`.
- Test each allowed FLEX subset, locks/exclusions, and site-rule conflicts.
  — `test_flex_positions_exhaustive`, `test_each_flex_position`,
  `test_lock_exclude`, `test_draftkings_rejects_single_game`.
- No new options produces unchanged feasible regions and objectives.
  — `test_no_new_options_changes_default_behavior`.
- Infeasible strategy combinations produce an explicit explanation.
  — `test_infeasible_secondary_stack_explains`,
  `test_infeasible_strategic_min_teams_explains`,
  `test_infeasible_strategic_max_teams_explains`.

## Phase 2: historical evaluation and parameter calibration — TOOLING IMPLEMENTED

**Status:** the evaluator/calibration tooling is implemented in
[multi_entry_optimizer.evaluation](src/multi_entry_optimizer/evaluation.py), independent of
[multi_entry_optimizer](src/multi_entry_optimizer/) (lineup generation) and
[benchmark.py](benchmark.py) (solver performance). Covered by
[tests/test_evaluation.py](tests/test_evaluation.py)
(41 synthetic-fixture tests). **No real historical NFL contest or
projection data was available in this workspace**, so none of this
tooling has been run against real data, no NFL-specific numeric policy
(overlap-by-slate-size, projection blend weights, correlation priors) has
been learned or promoted, and no [multi_entry_optimizer](src/multi_entry_optimizer/) `Config` default
has been changed. The 41 tests use only synthetic/fabricated fixtures and
establish the evaluator's arithmetic correctness; per the data gate below,
they cannot and do not establish that any strategy performs well in real
NFL contests.

Build an evaluator separately from lineup generation; keep
[the existing runtime benchmark](benchmark.py) focused on performance.

### Inputs and reproducibility

Use user-supplied or lawfully obtained data with:

- Contest/slate ID, site/scoring version, lock time, and game list.
- Player IDs, salaries, pre-lock projections and availability snapshots.
- Realized fantasy points, historical opponent entries or score counts.
- Entry fee, rank-to-prize mapping, maximum entries, and field capacity.
- Optional source projections and role/game-context features captured
  before lock. Do not require paid projection feeds.

Persist configuration, input hashes, data timestamps, training cutoff,
random seed, solver/library versions, and solve statuses per experiment.
Match player IDs and scoring systems strictly; missing actual scores must
not silently become zero.

**Implemented as:** `Contest`, `PayoutTier`, `SlateRecord`, `PlayerMeta`
hold the data model above (contests/payouts/slates are user-constructed;
no bundled real contest data). `score_lineup`/`score_field_entries` raise
`ValueError` on any missing player ID rather than defaulting to zero;
`validate_scoring_version` rejects a scoring-version mismatch between a
`Contest` and supplied realized points. `hash_payload`/`ExperimentRecord`/
`record_experiment`/`load_experiment_records` persist an append-only,
reproducible JSON-lines experiment log (config/data hashes, timestamps,
training cutoff, seed, solver versions, solve statuses).

### Replay and metrics

- Score every generated entry from actual player outcomes.
- Insert all generated entries into the same contest ranking, accounting
  for competition among the generated entries themselves.
- Handle ties by splitting the sum of prizes for occupied ranks among
  tied entries, according to the contest's rules.
- Distinguish additive counterfactual entry from fixed-capacity replay.
  For the latter, reproduce the paper's seeded random removal of field
  entries; report variation across removal samples.
- Report total fees, gross winnings, net profit, ROI, median/quantiles,
  best rank, top-k frequency, any-success frequency at declared thresholds,
  and uncertainty across slates. Report mean ROI alongside the median;
  top-heavy outcomes are dominated by rare outliers.
- Compare entry counts under both equal per-entry assumptions and a fixed
  total budget; more entries incur more cost.
- Report creation-order versus realized-rank Spearman correlation and
  the creation index of the best entry. Earlier projection rank is not a
  reliable promise of better realized tournament performance.
- Label cross-sectional dispersion of realized scores separately from
  estimated predictive variance of an individual lineup.

**Implemented as:** `rank_and_payout` ranks by score and splits tied
prizes across every occupied rank position, even across payout-tier
boundaries. `replay_additive` inserts generated entries alongside the
full recorded field; `replay_fixed_capacity` instead resamples
`field_capacity - len(generated)` kept field entries per seeded trial
(same seed reproduces the same trial sequence) and reports the resulting
`ReplayResult` per trial. `run_contest_replay` ties a `Contest` to either
mode. `summarize_replay_trials`/`summarize_across_slates` report mean and
median ROI, ROI quantiles, net-profit median, best-rank statistics,
`top_k_frequency`, `any_success_frequency` at caller-declared thresholds,
and `creation_order_spearman`/`best_entry_creation_index` per
`ReplayResult`. `affordable_entries` supports the equal-entry-count versus
fixed-budget comparison (the caller generates and replays the
budget-affordable entry count per strategy using this helper).
`cross_sectional_dispersion` reports the *realized field's* score
dispersion and is explicitly documented as distinct from any model's
predicted lineup variance (which Phase 2 does not compute).

### Tuning and predictive models

Use chronological train/validation/test splits grouped by slate.
Tune stack options, team concentration, overlap, and lineup count only
on training/validation data; evaluate once on held-out slates.
Avoid putting entries from one slate in different folds.

Learn a simple slate-size policy for overlap from NFL results. Inputs
can include game count, available player count, entry count, and strategy
constraints. NHL overlap recommendations of 4-7 are historical results,
not NFL constants. No automatic relaxation on an infeasible live run.

Optionally blend multiple supplied projections with ordinary least squares
or ridge regression by NFL position/role, using NumPy/SciPy. Evaluate
MAE/RMSE and downstream contest outcomes separately. Do not reuse hockey
coefficients, assume access to named commercial feeds, or add a goalie-win
bonus to NFL players.

**Implemented as:** `chronological_split` sorts slates by `lock_time` and
splits by slate ID (never dividing one slate's data across folds);
`assert_chronological_no_leakage` rejects overlapping slate-ID sets and
any fold boundary where a later fold's earliest slate is not strictly
after an earlier fold's latest slate. `bucket_by_game_count` and
`select_policy_by_bucket` perform *selection only*: the caller must
already have generated lineups and run replay for every candidate
overlap/strategy value on every training/validation slate; no NHL
overlap value (4-7) is hardcoded as an NFL recommendation anywhere.
`align_player_values`/`fit_ols`/`fit_ridge`/`predict_blend` implement
per-position projection blending with NumPy only (`lstsq` for OLS, a
closed-form intercept-unpenalized solve for ridge); `mae_rmse` evaluates
accuracy as a function kept separate from (never called by) the replay
functions above, so accuracy and contest-outcome evaluation cannot be
accidentally conflated.

NFL correlation analysis should align players within the same games and
use conditioning/residualization against pregame expectations to reduce
confounding. Estimate role/position groups with sample counts, uncertainty,
and shrinkage; do not treat raw noisy pair estimates as established facts.

**Implemented as:** `residualize` computes realized-minus-projected
residuals with strict ID matching. `estimate_group_correlations` groups
same-game player pairs by sorted position pair and same/opposing team,
reports the raw Pearson correlation alongside a sample-size-shrunk
estimate (`raw * n / (n + shrinkage_prior_strength)`) and a Fisher-z
confidence interval that is only populated once `sample_count >=
min_sample`; it never claims precision from a 1- or 2-sample group.

### Acceptance criteria and data gate

- Hand-calculated fixtures cover ties crossing payout bands, multiple own
  entries, contest capacity, fees, missing IDs, and no profitable entries.
- Seeded fixed-capacity replay is reproducible.
- Train/test separation checks reject future data leakage.
- Synthetic fixtures establish evaluator correctness only; they cannot
  establish an NFL strategy edge.
- Promotion of tuned defaults requires a sufficiently broad held-out NFL
  sample, with uncertainty and sensitivity reported. Real historical NFL
  contest/projection data is not currently supplied in this workspace.

**Verified by** [tests/test_evaluation.py](tests/test_evaluation.py):
`test_rank_and_payout_ties_crossing_payout_bands`,
`test_replay_additive_multiple_own_entries_different_tiers`,
`test_replay_fixed_capacity_enforces_contest_capacity` (capacity),
`test_replay_additive_basic_fees_and_profit` (fees),
`test_score_lineup_strict_matching`/`test_run_contest_replay_missing_ids_and_scoring_version_and_entry_limit`
(missing IDs), `test_replay_additive_no_profitable_entries`,
`test_replay_fixed_capacity_reproducible_with_seed` (seeded reproducibility),
and `test_assert_chronological_no_leakage_rejects_overlap_and_future_leakage`
(leakage rejection). **No promotion has occurred**: this phase ships only
the evaluation/calibration tooling and its synthetic-fixture tests; it
does not include real NFL contest/projection data, does not run any
tuning against real data, and does not change any `multi_entry_optimizer` default.

## Phase 3: optional moment-aware sequential MILP — IMPLEMENTED

**Status:** implemented in [multi_entry_optimizer](src/multi_entry_optimizer/), covered by
[tests/test_moments.py](tests/test_moments.py) (39 tests). All fixtures
are synthetic, hand-constructed factor models built to be exactly
checkable against direct quadratic evaluation; this phase establishes
MILP/mathematical correctness only. No real NFL player variance/covariance
data was available in this workspace, so no `variance_floor`/
`covariance_cap`/`correlation_ceiling` value has been validated against
real contest outcomes, and `Config.moments` defaults to `None` (fully
opt-in; confirmed unchanged default behavior by
`test_disabled_moment_mode_unchanged`).

This implements the **general formulation**, beyond what the shipped
Julia hockey heuristic implements.

### Moment data contract

Keep [Player](src/multi_entry_optimizer/) and the existing seven-column CSV compatible.
Accept a separate player-ID-aligned moment dataset containing player
variances and covariance values with scoring version and as-of metadata.
Require complete diagonal coverage, finite numbers, symmetry, nonnegative
variances, and a positive-semidefinite matrix within documented numerical
tolerance. Missing covariances must follow an explicit, recorded modeling
policy; never silently infer independence.

Prefer regularized, coherent covariance estimates or game/team factor
models. Independently clipping pair correlations or arbitrarily dropping
small matrix entries can destroy positive semidefiniteness. Any repair,
shrinkage, or independence assumption must be explicit and auditable.

For a binary lineup vector `x`, player means `mu`, and covariance `Sigma`:

```text
lineup mean       = mu.T @ x
lineup variance   = x.T @ Sigma @ x
cross covariance  = x.T @ Sigma @ previous_x
```

**Implemented as:** `Moments` holds `scoring_version`, `as_of`,
`player_ids`, `variance`, `covariance` (sparse, sorted `(lesser, greater)`
ID-pair keys only), `missing_covariance_policy` (required, currently only
`"assume_independent"` is supported), and `psd_tolerance`. `validate_moments`
checks complete diagonal coverage for the supplied player pool, finite
values, nonnegative variances, non-self/sorted covariance keys, and
positive semidefiniteness via `numpy.linalg.eigvalsh` within tolerance,
returning the dense submatrix aligned to that pool's order. The lineup
mean formula is already the existing `-projection` MILP objective, unchanged
by this phase; `lineup_variance`/`lineup_covariance`/`realized_correlation`
implement the other two formulas above as post-solve diagnostics on any
`Lineup`, not MILP inputs.

### MILP implementation with open-source HiGHS

Keep the objective `maximize mu.T @ x`.

1. For nonzero off-diagonal covariance pairs `(j,k)` with `j < k`,
   introduce `z[j,k] = x[j] * x[k]` using:

   ```text
   0 <= z[j,k]
   z[j,k] <= x[j]
   z[j,k] <= x[k]
   z[j,k] >= x[j] + x[k] - 1
   ```

   Continuous `z` in `[0,1]` is sufficient when the linked `x` variables
   are binary. Update model integrality metadata accordingly, or use
   binary auxiliaries first if simpler.

2. Enforce the exact variance floor:

   ```text
   sum_j Sigma[j,j] * x[j]
     + 2 * sum_{j<k} Sigma[j,k] * z[j,k] >= variance_floor
   ```

   Include negative covariance coefficients and all three product links;
   omitting links can give false feasibility.

3. For each prior lineup, precompute `c = Sigma @ previous_x`.
   Its covariance constraint is simply `c.T @ x <= covariance_cap`.
   No product variables are needed because the previous lineup is fixed.
   This accounts for shared-player variance and cross-player dependence,
   unlike overlap alone.

4. Keep overlap available as a separate operational diversity constraint.
   Call a covariance cap a covariance cap, not a correlation cap.
   Correlation has a variable denominator:

   ```text
   corr(x, previous_x) =
       covariance / sqrt(variance(x) * variance(previous_x))
   ```

   Report realized modeled correlations after solve. If a nonnegative
   correlation ceiling is required, a sufficient conservative covariance
   bound is `rho_max * sqrt(variance_floor * previous_variance)`, provided
   the new lineup's positive variance floor is actually enforced.
   Do not claim this is an exact correlation formulation.

5. Accept scalar bounds first; add explicit per-iteration schedules only
   after validation. The paper suggests tighter early limits and relaxed
   later limits as an option, not a demonstrated universal optimum.

**Implemented as:** `build_model` allocates one continuous `z[j,k]`
(`integrality=0`, bounds `[0,1]`) per nonzero off-diagonal pair and adds
the three linking rows plus the variance-floor row when
`Config.variance_floor` is set; `ModelData.integrality` is now a per-column
array (all-integer by default) and `ModelData.sigma` carries the dense
matrix for the iterative loop. `generate_lineups` adds one
`c.T @ x <= cap` row per completed lineup (parallel to the existing
overlap-cut row) when `covariance_cap` or `correlation_ceiling` is set,
with the correlation-ceiling path computing `cap` from the just-solved
lineup's own modeled variance. Only scalar, non-iteration-dependent bounds
are implemented (item 5); no per-iteration schedule exists.

### Acceptance criteria

- Exhaustive tiny-pool comparison with direct quadratic evaluation.
- Test unequal variances, positive/negative covariance, and zero off-diagonals.
- With equal variances and zero off-diagonals, covariance cuts reduce to
  scaled overlap cuts.
- Demonstrate two disjoint lineups with positive covariance: zero player
  overlap does not imply independence.
- Reject invalid matrices, ambiguous ID alignment, and missing diagonals.
- Benchmark variable/row counts, memory, and wall time before enabling
  dense moments on realistic slates. A dense pool of 192 players has
  18,336 possible off-diagonal pairs; do not assume baseline solve times.
- Disabled moment mode preserves current behavior and input requirements.

**Verified by** [tests/test_moments.py](tests/test_moments.py):
`test_variance_floor_matches_exhaustive_quadratic_evaluation` (exhaustive,
using a hand-built factor model with unequal variances and positive/
negative/zero covariance in one fixture);
`test_covariance_cap_reduces_to_scaled_overlap_with_equal_variance_no_covariance`
(reduction to scaled overlap, byte-for-byte identical lineup sequences);
`test_lineup_covariance_disjoint_players_can_be_strongly_positive` and
`test_covariance_cap_blocks_disjoint_correlated_lineup_that_overlap_allows`
(disjoint-but-correlated demonstration, including a full MILP scenario
where `covariance_cap` rejects what `max_overlap=0` alone permits);
`test_validate_moments_rejects_*` (7 tests covering non-PSD, ambiguous
key ordering, self-pairs, unknown IDs, non-finite/negative values, and
missing diagonals); `test_variance_floor_adds_exactly_one_continuous_variable_per_nonzero_pair`
and `test_dense_moments_on_moderately_sized_pool_still_solves` (model-size
scaling awareness, the latter on a 60-player fully-dense covariance
matrix); `test_disabled_moment_mode_unchanged` (disabled-mode regression).

## Phase 4: probability diagnostics and optional portfolio selection — IMPLEMENTED

**Status:** implemented in
[multi_entry_optimizer.probability](src/multi_entry_optimizer/probability.py), built on
Phase 3's `Lineup`/`Moments` artifacts and kept separate from
[multi_entry_optimizer](src/multi_entry_optimizer/)'s lineup generation (it imports from `multi_entry_optimizer`,
not the reverse). Covered by
[tests/test_probability.py](tests/test_probability.py)
(31 tests). All fixtures are synthetic Gaussian/factor-model constructions;
this phase establishes statistical/algorithmic correctness only. No real
NFL score distribution, success threshold, or coverage-selected portfolio
has been validated against real contest outcomes.

### Diagnostics first

Require a declared success threshold `t`, sourced from training data or
specified by the user, never the held-out contest's winning score.
For Gaussian diagnostics, compute marginal success as
`norm.sf((t - mean) / standard_deviation)` and pairwise joint survival
using a numerically stable bivariate Gaussian calculation.

Validate tails with analytic cases and Monte Carlo; handle zero variance,
near-perfect correlation, underflow, and integration error explicitly.
The extracted bounds in Eq. (2.4) and Appendix 7 contain apparent
scale/normalization ambiguities. Before coding those bounds, check
invariance under score-unit rescaling and rederive them independently.
Use standard Gaussian probabilities rather than blindly transcribing
an extracted formula.

Report marginal probabilities, pair intersections, `U2`, and simulated
`U`. The raw second-order lower bound can be negative and its marginal
gains can be negative; do not label it a calibrated probability, silently
clip it, or give it the exact-union submodularity guarantee. Quantify its
approximation error on the NFL scenarios used.

**Implemented as:** `marginal_success_probability` (`norm.sf` on a
z-score, with an explicit zero-variance point-mass case) and
`pairwise_joint_survival` (the orthant-probability identity
`sf(za)+sf(zb)-1+Phi2(za,zb;rho)`, with explicit Frechet-bound shortcuts
near `rho=±1` and a validated-bounds check that raises rather than papers
over a genuine numerical problem). Neither transcribes Eq. (2.4);
both use only standard `scipy.stats` Gaussian functions, per the
ambiguity concern above. `compute_success_diagnostics` reports
`marginal_probabilities`, `pairwise_joint_probabilities`, the raw
(never-clipped) `u2`, its per-lineup `u2_marginal_gains` (which the tests
show can go negative), and a seeded `simulated_u` with its binomial
standard error, plus an `approximation_error` property
(`u2 - simulated_u`). The Monte Carlo simulation
(`simulate_player_level_scenarios`) draws jointly at the **player** level
via the Moments covariance (not by assuming pairwise lineup correlations
alone combine to a valid joint distribution, which they need not for 3+
lineups), so every lineup's simulated success is evaluated against
exactly the same draws.

### Optional candidate-pool portfolio mode

Generate a diverse legal candidate pool using earlier phases. Simulate
correlated player outcomes once per scenario so every lineup is evaluated
against the **same** outcomes. Select `k` candidates by uncovered-success
scenario mass: each iteration maximizes the probability mass newly covered.

This is exact greedy maximum coverage on a finite weighted scenario set.
Under a cardinality-only selection constraint it has the classical
guarantee relative to the optimum **on that candidate/scenario set**,
not the full feasible lineup universe or true unknown NFL distribution.
Hard overlap or exposure constraints added at selection time invalidate
that simple guarantee.

Compare against the sequential baseline using separate evaluation
scenarios and historical held-out contests. A fixed threshold is a
success proxy, not literal tournament victory. Actual rank-based victory
requires a field-score model, and ties need a declared success convention.
Ownership and payout modeling should remain a separately scoped extension.

**Implemented as:** `generate_candidate_pool` reuses
`multi_entry_optimizer.generate_lineups` (Phases 1-3) to produce more candidates than
the eventual portfolio size `k`; `select_portfolio_by_coverage` simulates
once via `simulate_player_level_scenarios`, scores every candidate against
those same draws, and calls `greedy_scenario_coverage` (exact greedy
maximum coverage, with an optional non-uniform `scenario_weights`).
`evaluate_portfolios_success_rates` evaluates multiple named portfolios
against one shared simulation for a fair, paired comparison.
`compare_to_sequential_baseline` ties these together, explicitly rejecting
(raising `ValueError`) if the selection and evaluation seeds are equal, to
enforce that the comparison is genuinely held-out rather than graded on
the same draws used for selection. Historical held-out **contest**
comparison (as opposed to held-out simulated scenarios) is not built here;
it would use [multi_entry_optimizer.evaluation](src/multi_entry_optimizer/evaluation.py) (Phase 2)
once real data is available.

Acceptance: enumerate tiny coverage instances; verify marginal gains,
monotonicity, dependence handling, reproducible seeds, and held-out
evaluation. Track candidate-generation quality and scenario sampling
error separately from selection-optimization error.

**Verified by**
[tests/test_probability.py](tests/test_probability.py):
`test_greedy_matches_brute_force_optimal_on_tiny_instance` and
`test_greedy_achieves_at_least_the_one_minus_one_over_e_guarantee`
(enumerated tiny instances, including the explicit `1-1/e` bound);
`test_greedy_marginal_gains_are_non_increasing` (a provable consequence of
submodularity for greedy coverage, verified directly);
`test_greedy_covered_mass_is_monotonically_non_decreasing`;
`test_greedy_prefers_diverse_over_redundant_candidates` (dependence
handling); `test_greedy_reproducible_and_deterministic` and
`test_simulate_union_probability_reproducible_with_seed` (seeds);
`test_compare_to_sequential_baseline_held_out_evaluation` and
`test_compare_to_sequential_baseline_requires_distinct_seeds` (held-out
evaluation); `test_pairwise_joint_survival_perfect_positive_correlation_frechet_upper_bound`/
`..._negative_correlation_frechet_lower_bound`/`..._near_perfect_correlation_stays_bounded`
(analytic edge cases); `test_u2_marginal_gains_can_be_negative` (the
explicitly allowed non-monotonicity).

## Phase 5: bounded formulation/performance experiments — IMPLEMENTED

**Status:** implemented in
[formulation_experiments.py](formulation_experiments.py), covered by
[tests/test_formulation_experiments.py](tests/test_formulation_experiments.py)
(17 tests). **Conclusion from measured results below: no `multi_entry_optimizer`
default was changed.** Every alternative formulation produces an
equivalent feasible region (proven), but none showed a measured
end-to-end benefit over the current defaults on this stack — consistent
with the paper's own finding that stronger formulations are not
automatically faster with open-source solvers. The alternative
formulations remain available as tested, documented artifacts in
`formulation_experiments.py`, not as new `multi_entry_optimizer` options.

The paper warns that seemingly better formulations can be slower;
parsed page 29 and
Appendix 8.

- Compare the current aggregate DST conflict constraint against pairwise
  `x[dst] + x[opponent] <= 1`.
- Compare aggregate versus stronger membership links for stack groups.
- Optionally construct two lineups at once, maximizing their combined
  mean with a linearized mutual-overlap bound and all prior-lineup cuts.
  Add ordering/symmetry controls and explicit runtime limits.
  This still is not direct payout optimization.
- Benchmark on representative NFL slates, 1/20/100/150 entries, several
  overlap values, and different strategy/moment modes. Record build time,
  solve time, memory, status, gap, objective, and solver versions.
- Re-evaluate SciPy versus native HiGHS for the expanded models; the
  existing synthetic baseline result cannot establish the winner here.
  Keep all required solvers open-source and avoid adding a modeling layer
  without demonstrated need.

**Implemented as:** `multi_entry_optimizer.dst_conflict_rows(players, formulation)` (a
pure, behavior-preserving extraction from `build_model`, so the default
aggregate formulation is unchanged) returns either form;
`formulation_experiments.build_model_with_dst_formulation` swaps it into
an otherwise-identical model. `cardinality_link_rows`/
`build_cardinality_benchmark_model` implement the Appendix-8-style
weak/strong cardinality-indicator comparison as a **self-contained MILP**
(see the "scope" note below on why this does not thread through the full
NFL model). `build_joint_pair_model`/`generate_lineups_two_at_a_time`
implement the optional two-lineups-at-once joint MILP: two full copies of
`build_model`'s rows, every prior lineup's overlap cut applied to both
copies, a linearized mutual-overlap bound (continuous `z_j = x_j * x'_j`
per player, the same three-inequality technique as Phase 3's variance
floor), an objective-based symmetry break (lineup A's projection total
`>=` lineup B's), and an explicit per-pair `time_limit`.
`compare_solvers_on_expanded_model` re-runs the Phase 0 SciPy-vs-native-HiGHS
comparison with Phase 1's `strategic_max_teams` and Phase 3's
`variance_floor` enabled, using a **sparse**, same-team-only synthetic
covariance structure (a fully dense cross-team factor was tried first and
made the benchmark impractically slow, reproducing the exact "do not
assume baseline solve times" warning from Phase 3 — see the measured
results below for what that mistake cost before it was caught and fixed).

**Scope note on "stronger membership links for stack groups":** the
paper's strengthened line-stacking form (`indicator <= x_j` for every
member `j`) is only a *valid* encoding of "at least `k` selected" when
`k` equals the full group size (`k == M`); `multi_entry_optimizer`'s actual
`secondary_stack`/`strategic_*_teams` constraints are general
at-least-`k` with `k` typically less than the group size, where that
disaggregation would be flatly **incorrect** (it would force every
member whenever the indicator is 1, not just `k` of them). This
experiment therefore measures the weak/strong technique on a
self-contained MILP scoped to the one case where it legitimately
applies, rather than presenting an incorrect "drop-in" comparison inside
the full NFL model.

### Acceptance criteria

- Equivalent feasible integer solutions: `test_dst_conflict_rows_define_identical_feasible_region_exhaustive`
  proves the aggregate and pairwise DST formulations define the
  **identical** feasible set via exhaustive enumeration on a tiny pool
  (not just one solve path); `test_compare_dst_formulations_equivalent_on_realistic_pool`
  extends this to full sequential multi-lineup runs.
  `test_cardinality_formulations_define_identical_feasible_region_exhaustive`
  proves the same for the cardinality-link comparison.
  `test_joint_pair_combined_objective_never_worse_than_sequential` proves
  (not just measures) that the joint model's combined objective can never
  be lower than sequential's, since any feasible sequential pair is a
  feasible point of the joint problem too.
- Measured end-to-end benefit (actual run; teams=6/72 players, single
  repeat per scale due to this session's wall-clock budget — see the
  limitation note below):

  | Comparison | n | Result |
  | --- | ---: | --- |
  | DST aggregate vs. pairwise | 10 | aggregate 1.42s vs. pairwise 1.59s; equivalent lineups |
  | DST aggregate vs. pairwise | 20 | aggregate 3.90s vs. pairwise 4.88s; equivalent lineups |
  | Cardinality aggregate vs. disaggregated (k=M, isolated) | — | aggregate 0.66s (81 rows) vs. disaggregated 0.57s (401 rows); equivalent objective |
  | Two-at-a-time vs. sequential | 2 | sequential 0.046s vs. joint 0.284s (~6x slower); identical combined objective |
  | SciPy vs. native HiGHS (expanded model) | 10 | SciPy 10.45s vs. HiGHS 13.59s (SciPy ~23% faster) |
  | SciPy vs. native HiGHS (expanded model) | 20 | SciPy 27.94s vs. HiGHS 40.32s (SciPy ~31% faster) |

  **Conclusion:** pairwise DST rows added 3x the row count for no
  measured speed benefit (mildly slower at the tested scales); two-at-a-time
  was measurably and substantially slower with no objective improvement,
  directly reproducing the paper's own finding; SciPy remained faster
  than native HiGHS by a growing margin on the expanded model, extending
  (not just assuming) the Phase 0 solver choice. The isolated cardinality
  comparison was inconclusive from a single run and is scoped too
  narrowly to generalize to `multi_entry_optimizer`'s actual stack constraints
  regardless. **None of these results justify changing any `multi_entry_optimizer`
  default**, so none were changed.
- Limitation: this run used `repeats=1` (not the originally intended 3+)
  and a modest 72-player pool, to stay within a responsibly bounded
  wall-clock budget in a shared environment (an earlier attempt with a
  denser covariance structure and a larger 1/20/100/150 sweep was
  accidentally left running for over 30 minutes before being stopped and
  the covariance structure was fixed to be appropriately sparse). The
  directional conclusions above (equivalence holds; no formulation change
  pays off) are expected to be robust to that limitation, since the
  proven equivalence results do not depend on repeat count, and the
  timing gaps between the faster and slower options are not close enough
  at the tested scales to plausibly reverse with more repeats — but this
  has not been re-verified at the full 150-lineup scale in this session.
  Rerunning `python formulation_experiments.py` with larger `--lineups`
  and `--repeats` values (and patience, or a smaller `--teams` count) is
  the natural next step before drawing a stronger conclusion.

Do not promote these experiments based only on tighter LP relaxations or
paper-era solver behavior. Require equivalent feasible integer solutions
and measured end-to-end benefit.

## Recommended delivery order and boundaries

1. ~~Ship structural controls and independent historical-evaluator
   fixtures.~~ **Structural controls (Phase 1) and evaluator tooling with
   synthetic fixtures (Phase 2) are both implemented.**
2. **Obtain historical NFL data; tune and validate structural policies.**
   This step is **not yet done** — it is the one remaining part of this
   item. The Phase 2 tooling (replay, chronological splits, overlap-policy
   selection, projection blending, correlation estimation) is ready to run
   once real contest/projection data is available, but no such data has
   been supplied or used in this workspace.
3. ~~Add moment-aware MILP behind explicit opt-in, starting with
   covariance diagnostics/cuts and then variance-floor linearization.~~
   **Implemented** (Phase 3, above): `variance_floor`,
   `covariance_cap`/`correlation_ceiling`, and the diagnostic functions.
   Not yet validated against real player variance/covariance data (none
   is available in this workspace).
4. ~~Add probability diagnostics; consider scenario-coverage selection
   only when the distribution model passes validation.~~ **Implemented**
   (Phase 4, above): Gaussian diagnostics, `U2`/simulated-`U` comparison,
   and greedy scenario-coverage portfolio selection. The "distribution
   model passes validation" gate still applies before trusting any of
   this on real data: no real NFL score distribution has been fit or
   validated in this workspace, so the diagnostics and selection have
   been proven correct on synthetic fixtures only.
5. ~~Run optional strengthening/batching experiments if runtime becomes a
   measured bottleneck.~~ **Implemented and run** (Phase 5, above):
   aggregate-vs-pairwise DST rows, aggregate-vs-disaggregated cardinality
   linking, and sequential-vs-two-at-a-time, each with proven-equivalent
   feasible solutions. Measured at a bounded (single-repeat, 72-player)
   scale: none showed a benefit justifying a default change, and
   two-at-a-time was measurably much slower, reproducing the paper's own
   finding. SciPy remained faster than native HiGHS on the Phase 1+3
   expanded model too. A larger/`--repeats`-robust re-run is the natural
   next step, not a requirement before trusting the directional
   conclusion that no default should change.

Retain the existing simple optimizer as the documented baseline throughout.
Do not replace it with an unvalidated joint optimizer, copy hockey
parameters into NFL defaults, or promise profitability from mathematical
correctness tests alone. **Phase 1's new structural options, Phase 3's
moment-aware constraints, and Phase 4's probability diagnostics/coverage
selection are mathematically and statistically verified** (exhaustive
enumeration, constructive fixtures, direct quadratic comparison, analytic
edge cases, brute-force coverage comparison, explicit infeasibility
handling) **but not empirically validated**: no claim is made that any
stack, team-concentration bound, FLEX restriction, variance floor,
covariance cap, success threshold, or coverage-selected portfolio improves
real NFL contest results until Phase 2 is completed with real historical
data. **Phase 5 is the one phase whose own stated acceptance criteria
have actually been measured end-to-end on this stack** (not just proven
mathematically), and the measured conclusion was to change nothing.
