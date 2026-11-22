# multi-entry-optimizer

Open-source, mixed-integer NFL classic-contest daily fantasy sports (DFS)
lineup generation in Python, using only open-source solvers (SciPy's
`milp`, backed by HiGHS). No commercial solver, account, network service,
or license key is required.

Not affiliated with, endorsed by, or sponsored by the National Football
League (NFL), DraftKings, FanDuel, or any fantasy sports operator.

This project implements and extends the integer-programming approach of
Hunter, Vielma, and Zaman's *Picking Winners Using Integer Programming*
(freely available on [arXiv](https://arxiv.org/abs/1604.01455)), adapted
from the authors' own fantasy-hockey reference implementation
([`picking_winners.jl`](picking_winners.jl)) to NFL classic-contest rules.
See [NOTICE.md](NOTICE.md) for full attribution and licensing details,
and [PAPER_IMPLEMENTATION_PLAN.md](PAPER_IMPLEMENTATION_PLAN.md) and
[CHANGELOG.md](CHANGELOG.md) for the paper-to-code gap analysis and
full development history.

## Install

Requires Python 3.11 or newer.

```powershell
python -m pip install multi-entry-optimizer
```

From a source checkout instead:

```powershell
python -m pip install -e .
```

## Quickstart

```powershell
multi-entry-optimizer examples\players.csv lineups.csv --lineups 20 --max-overlap 7
```

Or from Python:

```python
from pathlib import Path
from multi_entry_optimizer import Config, generate_lineups, load_players, write_lineups

players = load_players(Path("examples") / "players.csv")
lineups = generate_lineups(players, Config(num_lineups=20, max_overlap=7))
write_lineups(Path("lineups.csv"), lineups)
```

## Features

- **Core lineup generation** (`multi_entry_optimizer`): sequential,
  projection-maximizing MILP with DraftKings/FanDuel classic-contest
  rules, overlap control, locks/excludes, QB stacking, bring-back, and
  DST-vs-offense avoidance.
- **Strategic and stacking options** (opt-in): team-concentration bounds,
  a secondary same-team stack from a different game, and FLEX-position
  restriction. See [Strategic and stacking options](#strategic-and-stacking-options).
- **Named strategy presets** (`STRATEGIES`/`apply_strategy`): common
  stack/bring-back/avoidance combinations by name.
- **Optional moment-aware constraints**: variance floors, covariance
  caps, and correlation ceilings via a linearized MILP, plus post-solve
  variance/covariance/correlation diagnostics. See
  [Moment-aware constraints](#moment-aware-constraints-optional-python-api-only).
- **`multi_entry_optimizer.probability`** (optional, Python API only):
  Gaussian and Monte Carlo success-probability diagnostics, plus greedy
  scenario-coverage portfolio selection. See
  [Probability diagnostics](#probability-diagnostics-optional).
- **`multi_entry_optimizer.evaluation`** (optional tooling): historical
  contest replay, chronological calibration splitting, projection
  blending, and correlation estimation, fully independent of lineup
  generation. See [Historical evaluation](#historical-evaluation-optional-tooling).

All optional features default to the original, unrestricted behavior;
nothing above changes a default unless you set it explicitly.

## Input

Provide a UTF-8 CSV with these columns (additional columns are ignored):

```csv
id,name,position,team,opponent,salary,projection
12345,Example Receiver,WR,BUF,NYJ,6500,18.75
```

- `id`: unique player ID, retained as text, including leading zeroes.
- `position`: exactly one of QB, RB, WR, TE, DST. Multi-position strings,
  kickers, and showdown captain slots are not supported.
- `team`, `opponent`: consistent abbreviations for a single slate, without
  home/away prefixes. A team can have only one opponent.
- `salary`: positive integer, no currency symbol or thousands separator.
- `projection`: finite floating-point fantasy points, including zero or
  negative values. Supply projections using the selected contest's scoring.

CSV headers are lowercase and case-sensitive. Position and team values
are normalized to uppercase. Empty pools, missing values, duplicate IDs,
invalid positions, and inconsistent opponents are rejected.
Supply only available players from the selected contest; the optimizer
does not fetch projections, injuries, start times, or salary feeds.
[`examples/players.csv`](examples/players.csv) is entirely fictional and
illustrates the schema.

## NFL classic presets

Both presets use **QB, RB, RB, WR, WR, WR, TE, FLEX, DST**.
FLEX permits RB, WR, or TE, never QB or DST.

| Preset | Salary cap | Minimum teams | Minimum games | Maximum players per team |
| --- | ---: | ---: | ---: | ---: |
| `draftkings` | $50,000 | 2 | 2 | 9 (no additional team cap) |
| `fanduel` | $60,000 | 3 | 1 | 4, including DST |

These are classic-format assumptions, not automatic discovery of current
contest rules. Three FanDuel teams necessarily span at least two games
in a consistent slate. Check the specific contest's requirements before
entry. Single-game/showdown, college, best-ball, late-swap, and unusual
contest formats are outside this implementation.

## CLI usage

```powershell
# 100-lineup portfolio, with at most seven shared players.
multi-entry-optimizer examples\players.csv lineups.csv --lineups 100 --max-overlap 7

# QB + two WR/TE teammates, plus an opposing skill player.
multi-entry-optimizer examples\players.csv lineups.csv --lineups 20 --stack 2 --bring-back

# FanDuel with per-lineup lock/exclude controls and a minimum spend.
multi-entry-optimizer examples\players.csv lineups.csv --site fanduel --lineups 5 --lock A_QB --exclude B_DST --min-salary 45000
```

`python -m multi_entry_optimizer` also works, without adding the scripts
directory to PATH.

Options:

- `--lineups`: positive count; default 1.
- `--max-overlap`: 0 through 8, default 7. Applied to every pair of
  generated lineups; 8 prevents duplicate player sets.
- `--stack`: 0 (default), 1, or 2 same-team WR/TE players with the QB.
- `--bring-back`: requires a nonzero stack; at least one opposing RB/WR/TE.
- DST-versus-offense avoidance is **on by default**, even without a QB
  stack. Use `--allow-dst-opponents` for a pure positional/salary baseline.
  This is a strategy constraint, not a site eligibility rule.
- `--min-salary`: optional minimum total salary.
- `--lock`, `--exclude`: repeat for multiple exact input IDs. Locks apply
  to every lineup and can make restrictive overlap settings infeasible.
- `--time-limit`: positive seconds **per lineup**, default 60.

### Strategic and stacking options

These are all opt-in; omitting them reproduces the original behavior.

- `--strategic-min-teams`, `--strategic-max-teams`: bound the number of
  distinct teams used, independent of each site's own minimum-teams rule
  and per-team player cap. Validated against the number of teams in the
  pool. A maximum/exact count requires a genuinely two-way-linked team
  indicator; a simple minimum-only indicator (as used for the site's own
  minimum-teams rule) cannot correctly enforce a maximum.
- `--strategic-team-scope`: `all` (default) counts every roster member
  toward the team bound above, or `offense` to exclude DST, so a team
  represented only by its DST does not count as a "used" team.
- `--secondary-stack {0,1,2}`: requires at least that many same-team
  RB/WR/TE players (configurable via `--secondary-stack-positions`) from
  a team in a **different game than the selected QB's game**. This is an
  NFL adaptation of the paper's full/partial line-stacking idea; NFL
  players do not participate in fixed lines, so this stacks a second
  correlated game instead. Requires enough eligible players on some
  non-QB-game team or the request is infeasible.
- `--flex-positions`: restrict the single FLEX slot to a subset of
  `RB WR TE` (default: all three). Implemented by fixing the restricted
  position's roster count to its base count (no "+1" flex allowance),
  not by adding slot-assignment variables.

### Named strategy presets

`--strategy NAME` applies a named combination of `stack`, `bring_back`,
`secondary_stack`, and `avoid_dst_opponents`. It cannot be combined with
`--stack`, `--bring-back`, `--allow-dst-opponents`, or `--secondary-stack`
directly (the CLI rejects that combination). These are NFL-specific names;
they intentionally do not reuse the Julia hockey "Type 1-5" naming, since
NFL has no line or power-play structure to stack.

| Name | stack | bring_back | secondary_stack | avoid_dst_opponents |
| --- | ---: | --- | ---: | --- |
| `baseline` | 0 | No | 0 | No |
| `dst-avoidance` | 0 | No | 0 | Yes |
| `qb-plus-one` | 1 | No | 0 | Yes |
| `qb-plus-two` | 2 | No | 0 | Yes |
| `qb-stack-bring-back` | 2 | Yes | 0 | Yes |
| `secondary-stack` | 2 | Yes | 2 | Yes |

```powershell
# Named strategy preset.
multi-entry-optimizer examples\players.csv lineups.csv --lineups 20 --strategy secondary-stack

# Manual strategic team-concentration and secondary-stack controls.
multi-entry-optimizer examples\players.csv lineups.csv --lineups 20 --strategic-max-teams 3 --secondary-stack 2 --flex-positions WR TE
```

Output columns are `QB,RB1,RB2,WR1,WR2,WR3,TE,FLEX,DST,salary,projection`,
with input IDs in the roster columns. This is an analysis CSV, **not a
site-upload template**: official templates may require repeated position
headers, contest metadata, or different player-ID formatting. Roster-slot
assignment follows input order within positions; it is not late-swap-aware.
Projection output is rounded to six decimals, but optimization uses the
original floating-point inputs.

The command exits nonzero with an explanation if the complete requested
portfolio cannot be solved. An existing output is preserved on failure.
No incumbent from a time-limited solve is reported as optimal.

## Python API

```python
from pathlib import Path
from multi_entry_optimizer import Config, apply_strategy, generate_lineups, load_players, write_lineups

players = load_players(Path("examples") / "players.csv")
config = apply_strategy(Config(num_lineups=20, max_overlap=7), "secondary-stack")
lineups = generate_lineups(players, config)
write_lineups(Path("lineups.csv"), lineups)
```

### Moment-aware constraints (optional, Python API only)

There are no CLI flags for `Moments`/`variance_floor`/`covariance_cap`/
`correlation_ceiling`: a full covariance dataset doesn't fit a CLI flag,
so this is Python-API only (consistent with `multi_entry_optimizer.evaluation`
having no CLI either).

```python
from multi_entry_optimizer import Config, Moments, generate_lineups, lineup_variance, realized_correlation

# Build a (symmetric, PSD) covariance matrix however you like; a factor
# model is one option. Only nonzero off-diagonal pairs need keys, and keys
# must be a sorted (lesser, greater) player-ID tuple.
ids = tuple(p.id for p in players)
covariance = {("playerA", "playerB"): 0.4}  # positive: same-team correlation
moments = Moments(
    scoring_version="v1", as_of="2024-09-08T17:00:00Z", player_ids=ids,
    variance=tuple(0.0 for _ in ids),  # replace with real per-player variances
    covariance=covariance, missing_covariance_policy="assume_independent",
)
config = Config(num_lineups=5, max_overlap=6, moments=moments,
                variance_floor=20.0, covariance_cap=5.0)
lineups = generate_lineups(players, config)
lineup_variance(lineups[0], moments)                    # realized modeled variance
realized_correlation(lineups[0], lineups[1], moments)    # realized modeled correlation
```

`variance_floor` adds a lineup variance floor (linearized via continuous
`z[j,k]` product variables for nonzero covariance pairs); `covariance_cap`
bounds each new lineup's covariance against every previously generated
lineup; `correlation_ceiling` (requires `variance_floor`) converts a
correlation ceiling into a conservative covariance bound (documented as
conservative, not an exact correlation constraint). **Disabled by
default**: with `moments=None`, behavior is unchanged. See
[PAPER_IMPLEMENTATION_PLAN.md](PAPER_IMPLEMENTATION_PLAN.md#phase-3-optional-moment-aware-sequential-milp--implemented)
for the full constraint derivation.

### Probability diagnostics (optional)

`multi_entry_optimizer.probability` builds on the `Lineup`/`Moments`
artifacts above. It never infers its success threshold `t` from a
held-out contest's actual winning score — pass `t` explicitly, sourced
from your own data.

```python
import multi_entry_optimizer.probability as mp

# Diagnostics for a fixed set of lineups (requires the same Moments used
# to generate them, or any compatible one):
diagnostics = mp.compute_success_diagnostics(lineups, moments, threshold=150.0,
                                              simulation_count=20_000, seed=0)
diagnostics.u2                     # second-order surrogate; CAN be negative, never clipped
diagnostics.simulated_u            # seeded Monte Carlo estimate of the true union probability
diagnostics.approximation_error    # u2 - simulated_u

# Optional candidate-pool portfolio selection: generate more candidates
# than you need, then pick k by incremental scenario coverage.
candidates = mp.generate_candidate_pool(players, config, pool_size=50)
selection = mp.select_portfolio_by_coverage(candidates, moments, threshold=150.0,
                                             k=5, scenario_count=20_000, seed=0)

# Compare against the plain sequential top-k on a genuinely HELD-OUT
# scenario set (a different seed than was used for selection).
comparison = mp.compare_to_sequential_baseline(
    players, config, moments, threshold=150.0, k=5, pool_size=50,
    selection_seed=0, evaluation_seed=1,
)
comparison.sequential_success_rate
comparison.coverage_success_rate
```

`greedy_scenario_coverage`'s `1 - 1/e` approximation guarantee applies to
the specific finite candidate/scenario set used, not the true unknown NFL
score distribution or the full feasible lineup universe. See
[PAPER_IMPLEMENTATION_PLAN.md](PAPER_IMPLEMENTATION_PLAN.md#phase-4-probability-diagnostics-and-optional-portfolio-selection--implemented)
for the full function reference and every stated limitation.

### Historical evaluation (optional tooling)

`multi_entry_optimizer.evaluation` replays generated lineups against a
historical contest's actual field and payouts. It is fully independent of
`multi_entry_optimizer`'s lineup generation: convert a `Lineup`'s player
IDs to a tuple yourself when calling it.

```python
import multi_entry_optimizer.evaluation as me

contest = me.Contest(
    contest_id="...", slate_id="...", site="draftkings", scoring_version="v1",
    lock_time="2024-09-08T17:00:00Z", games=(("BUF", "NYJ"),), entry_fee=5.0,
    payouts=(me.PayoutTier(1, 1, 500.0), me.PayoutTier(2, 5, 50.0)),
    field_capacity=200, max_entries_per_user=150,
)
generated_entries = [tuple(p.id for p in lineup.players) for lineup in lineups]
results = me.run_contest_replay(
    contest=contest, generated_entries=generated_entries,
    realized_points=realized_points,  # {player_id: actual fantasy points}, strict ID matching
    realized_scoring_version="v1", field_scores=historical_field_scores,
    mode="fixed_capacity", trials=200, seed=42,  # seeded, so reproducible
)
summary = me.summarize_replay_trials(results, top_k=(1, 10), success_thresholds=(150.0,))
```

Every realized-score and projection lookup is strict: a missing player ID
raises `ValueError` rather than silently scoring zero. `mode="additive"`
instead adds generated entries to the recorded field as-is, ignoring the
contest's capacity limit. See
[PAPER_IMPLEMENTATION_PLAN.md](PAPER_IMPLEMENTATION_PLAN.md#phase-2-historical-evaluation-and-parameter-calibration--tooling-implemented)
for the full function reference, chronological train/validation/test
splitting, overlap-policy selection, projection blending, and pairwise
correlation estimation.

**This module is tooling only.** No real historical NFL contest or
projection data has been run against it in this project; no strategy or
parameter tuned with it should be assumed to perform well in a real
contest without your own validation against real data.

## Formulation and scope

For each player, binary `x[i]` indicates selection. Minimize
`-sum(projection[i] * x[i])` with:

- Exactly 9 total players, exactly 1 QB and 1 DST.
- 2-3 RB, 3-4 WR, and 1-2 TE. Together with roster size these allow
  exactly one additional RB/WR/TE in FLEX. If `flex_positions` excludes a
  position, its upper bound is fixed to its base count (2, 3, or 1), so it
  cannot absorb the FLEX slot.
- Salary bounds, minimum represented teams/games, and site team caps.
- If QB `q` is selected, at least `stack` teammates in WR/TE.
- If bring-back is enabled, at least one opposing RB/WR/TE for selected `q`.
- For DST `d`, `8*x[d] + sum(opposing offensive x) <= 8` when enabled.
- If `strategic_min_teams`/`strategic_max_teams` is set, a two-way-linked
  indicator `y[t]` per team `t` (scoped by `strategic_team_scope`) enforces
  `sum(y) >= strategic_min_teams` and/or `sum(y) <= strategic_max_teams`.
- If `secondary_stack` is set, a per-team indicator `z[t]` satisfies
  `secondary_stack*z[t] <= sum(eligible members of t)`, with
  `sum(z) >= 1` and `z[t] + x[q] <= 1` for every QB `q` and every team `t`
  in `q`'s own game.
- For every previous selected set `L`, `sum(x[i] for i in L) <= max_overlap`.
- Locked/excluded players have fixed binary bounds.

A relative MIP gap of zero is requested; "optimal" means the solver's
optimal status within its numerical tolerances, not exact rational
arithmetic.

This is **sequential greedy optimization**: each lineup is optimal
subject to prior cuts, but the entire portfolio is not jointly optimized
for tournament payout. QB stacks, the secondary stack, and strategic team
concentration are correlation/diversification heuristics, not a full
covariance model, unless `Moments` is explicitly supplied (see above).
Maximizing projections does not by itself maximize tournament win
probability or expected payout. Ownership, exposure percentages, contest
simulation, and payout structures beyond `multi_entry_optimizer.evaluation`
are not modeled. Overlap controls lineup similarity but does not impose
portfolio exposure limits.

## Solver choice and performance

Use **SciPy's `scipy.optimize.milp`**, backed by the open-source **HiGHS**
mixed-integer solver. This retains floating-point projections, uses sparse
constraint matrices, and avoids a separate modeling-language dependency.
NumPy and SciPy are the only runtime dependencies. HiGHS is MIT-licensed;
SciPy uses a BSD license.

Measured on Windows 11, Python 3.12.13, NumPy 2.4.4, SciPy 1.18.0, and
highspy 1.15.1: SciPy's bundled HiGHS was faster than native `highspy`
bindings on the benchmarked workloads (roughly 12% faster in the final
timed run of a 20-lineup/192-player synthetic benchmark; see
[CHANGELOG.md](CHANGELOG.md) for full numbers from this and the later
expanded-model benchmark). This is the fastest **measured candidate for
this project's workloads**, not proof of the fastest library for every
slate; CBC, GLPK, SCIP, and OR-Tools CP-SAT were not benchmarked.
Re-run `benchmark.py`/`formulation_experiments.py` on representative real
slates before changing backends.

## Development

```powershell
python -m pip install -e ".[dev]"
python -m pytest tests\ -q
python benchmark.py --teams 16 --lineups 20 --repeats 3
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development workflow,
[CHANGELOG.md](CHANGELOG.md) for release history, and
[PAPER_IMPLEMENTATION_PLAN.md](PAPER_IMPLEMENTATION_PLAN.md) for the
paper-to-code gap analysis behind every phase above.

## License

MIT (see [LICENSE](LICENSE)) for this project's own Python source. This
does **not** cover `picking_winners.jl`, which remains under its original
authors' Mozilla Public License 2.0 — see [NOTICE.md](NOTICE.md).
