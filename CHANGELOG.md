# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this
project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

Initial release: a from-scratch Python port of
[`picking_winners.jl`](picking_winners.jl) to NFL classic-contest daily
fantasy sports, converted from hockey rules to NFL rules and then
extended across five phases derived from reviewing the paper the Julia
code implements (Hunter, Vielma, and Zaman, *Picking Winners Using
Integer Programming*). Full technical detail, acceptance criteria, and
measured benchmark numbers for every phase are in
[PAPER_IMPLEMENTATION_PLAN.md](PAPER_IMPLEMENTATION_PLAN.md); this entry
is a summary.

### Phase 0: initial conversion

- Preserved the paper/Julia algorithm's sequential construction:
  maximize projected points for one lineup, add player-overlap cuts
  against every previous lineup, repeat.
- Replaced hockey rules with NFL classic rules: one player pool,
  QB/RB/WR/TE/DST positions, a skill-player FLEX, and DraftKings/FanDuel
  salary, team, and game constraints.
- Replaced hockey line/power-play stacking with NFL-appropriate
  correlation heuristics: QB + same-team WR/TE stack, optional opposing
  bring-back, and optional DST-vs-offense avoidance.
- Benchmarked native `highspy` against `scipy.optimize.milp` on the same
  sparse model and synthetic slate; SciPy was faster in both measured
  runs (10.05s vs. 11.47s median in the final timed run of a
  20-lineup/192-player benchmark) and was chosen as the production
  solver interface. Both are backed by the open-source HiGHS solver.
- Typed API, validated CSV input/output, CLI, lock/exclude controls,
  deterministic roster-slot assignment, atomic CSV writes, and explicit
  failure (never a silent partial portfolio) on infeasibility or a time
  limit.
- 47 tests: independent roster-legality checks, exhaustive-enumeration
  comparison on small pools, error-path coverage, and a 100-lineup
  end-to-end CLI run cross-checked against all constraints and all 4,950
  pairwise overlap limits.

### Phase 1: structural NFL heuristics

- Added strategic team-concentration bounds
  (`strategic_min_teams`/`strategic_max_teams`/`strategic_team_scope`),
  a secondary same-team stack from a different game than the QB's
  (`secondary_stack`/`secondary_stack_positions`), FLEX-position
  restriction (`flex_positions`), and named strategy presets
  (`STRATEGIES`/`apply_strategy`/`--strategy`).
- All options default to the original, unrestricted behavior.
- 31 new tests (78 total), with expected values computed via exhaustive
  enumeration fixtures.

### Phase 2: historical evaluation and calibration tooling

- New independent module (`multi_entry_optimizer.evaluation`): strict-matching
  contest replay (`run_contest_replay`, `replay_additive`,
  `replay_fixed_capacity`) with tie-splitting payouts (`rank_and_payout`),
  trial/cross-slate summaries, chronological train/validation/test
  splitting with leakage detection, overlap/strategy policy selection by
  slate bucket, NumPy-only OLS/ridge projection blending, shrinkage-aware
  pairwise correlation estimation, and reproducible experiment logging.
- 41 new tests (119 total), all on synthetic fixtures.
- **Tooling only**: no real historical NFL contest/projection data was
  available to run this against, and no lineup-generation default
  changed as a result of building it.

### Phase 3: optional moment-aware MILP

- Opt-in `Moments` dataset (player variances/covariances) plus
  `variance_floor`, `covariance_cap`, and `correlation_ceiling` `Config`
  fields, linearized into the MILP via continuous product variables for
  nonzero covariance pairs; post-solve `lineup_variance`,
  `lineup_covariance`, and `realized_correlation` diagnostics.
- 39 new tests (158 total) using a hand-built factor-model covariance
  fixture, including exhaustive comparison against direct quadratic
  evaluation and rejection of non-PSD/ambiguous/incomplete moment data.
- **Disabled by default**: with `moments=None`, behavior is unchanged
  from Phase 0/1.

### Phase 4: probability diagnostics and portfolio selection

- New module (`multi_entry_optimizer.probability`), Python-API only, built on
  Phase 3's `Lineup`/`Moments`: exact Gaussian marginal/pairwise joint
  survival probabilities, the `U2` second-order inclusion-exclusion
  surrogate (reported raw, **never clipped**, since it can be negative
  and non-monotonic), seeded player-level Monte Carlo simulation of the
  true union probability, exact greedy scenario-coverage portfolio
  selection (`1 - 1/e` guarantee relative to the finite candidate/
  scenario set used, not the full lineup universe), and a held-out
  baseline comparison against plain sequential top-k.
- 31 new tests (189 total): analytic Fréchet-bound edge cases, a
  demonstration that `U2` can go negative while the true simulated
  probability is near 1, greedy-vs-brute-force exact comparisons, and
  reproducible seeding.

### Phase 5: bounded formulation and performance experiments

- New script (`formulation_experiments.py`, dev-only, not packaged):
  aggregate vs. pairwise DST-conflict rows, aggregate vs. disaggregated
  cardinality-indicator linking, an optional two-lineups-at-once joint
  MILP, and a SciPy-vs-native-HiGHS re-run on an expanded model (Phase 1
  strategic constraints + Phase 3 moments enabled).
- Every alternative formulation was proven to define an identical
  feasible region to the existing default (exhaustive comparison on tiny
  pools, or an always-true inequality for the joint model).
- **Measured results** (single repeat, 72-player pool, bounded scale):
  no end-to-end benefit from any alternative. Pairwise DST rows added 3x
  the constraint rows for no speed gain; two-lineups-at-once was ~6x
  slower with no better combined objective (reproducing the paper's own
  finding); SciPy stayed faster than native HiGHS on the expanded model
  too (23-31% faster at the tested scales). **No lineup-generation
  default was changed** as a result.
- 17 new tests (206 total).
- This phase's first run (dense, same-loading-for-every-player synthetic
  covariance across 16 teams) hung for 37+ minutes before being stopped;
  the cause was a nearly-dense cross-team covariance matrix creating
  ~15,400 unnecessary MILP product variables. Fixed by using a sparse,
  same-team-only block-diagonal covariance structure for the benchmark,
  plus added wall-clock budgeting and progress logging to the script.

### Packaging

- Restructured from flat top-level modules (`nfl_dfs.py`,
  `historical_evaluation.py`, `probability_diagnostics.py`) into a
  `src/multi_entry_optimizer/` package (`lineups.py`, `evaluation.py`,
  `probability.py`), so `pip install multi-entry-optimizer` pairs with
  `import multi_entry_optimizer` with no name/import mismatch.
- Added full `pyproject.toml` metadata (license, authors, classifiers,
  keywords, project URLs), an MIT `LICENSE`, and `NOTICE.md` documenting
  this project's relationship to the academic paper and to
  `picking_winners.jl` (the original authors' MPL-2.0-licensed Julia
  reference implementation).
- Fixed two pre-existing type-checker diagnostics:
  `Player.game`'s return type (now provably `tuple[str, str]`, not
  `tuple[str, ...]`), and `ModelData.add`'s `coefficients` parameter
  (now `Mapping[int, float]`, covariant in value type, instead of an
  invariant `dict[int, float]`).
- Excluded `Picking-Winners.pdf` and `documents/*.md` (substantial
  extracted manuscript text) from version control; see
  [NOTICE.md](NOTICE.md).
