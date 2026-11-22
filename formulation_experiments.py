"""Phase 5: bounded formulation and performance experiments.

Compares alternative MILP formulations against
multi_entry_optimizer.lineups's defaults on measured, reproducible
benchmarks, per PAPER_IMPLEMENTATION_PLAN.md Phase 5. This module does
NOT change any multi_entry_optimizer.lineups default by importing it;
promoting an alternative formulation requires measured end-to-end
benefit, not just a tighter LP relaxation. The paper itself found
stronger formulations were sometimes slower with open-source solvers
(parsed page 29 / Appendix 8), which is exactly the kind of claim this
module is built to check on our own stack rather than assume.

What this covers:
  * Aggregate vs. pairwise-disaggregated DST-conflict rows, directly
    analogous to the paper's goalie-stacking weak/strong comparison
    (Appendix 8.2). Both forms define the IDENTICAL binary feasible
    region; only solver performance should differ. See
    `multi_entry_optimizer.dst_conflict_rows`.
  * Aggregate vs. disaggregated cardinality-indicator linking (the same
    mathematical technique as the paper's line-stacking weak/strong
    comparison, Appendix 8.2), tested as a self-contained MILP. The
    disaggregated "indicator <= each member" form is only a VALID
    encoding of "at least k selected" when k equals the full member
    count; multi_entry_optimizer.lineups's actual stack/team constraints
    are general at-least-k (k can be less than the group size), so this
    techinque does not generalize to a drop-in replacement there. This
    experiment measures the technique in the one case where it
    legitimately applies, not inside the full NFL model.
  * An optional two-lineups-at-a-time joint MILP (Eq. 5.1 / Appendix 8.1),
    with a linearized mutual-overlap bound and an objective-based
    symmetry-breaking constraint, benchmarked at a capped scale: the
    paper reports open-source solvers taking "longer than two hours" for
    two-at-a-time at scale, so this is not swept to the full 1/20/100/150
    range used for the other comparisons.
  * Re-running SciPy vs. native HiGHS on EXPANDED models (Phase 1
    strategic constraints plus Phase 3 moments enabled), since the
    original Phase 0 benchmark (benchmark.py) used a plain model only.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass, replace

import highspy
import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

from benchmark import native_lineups, synthetic_pool
from multi_entry_optimizer import (
    Config, Lineup, ModelData, Moments, OptimizationError, Player,
    build_model, dst_conflict_rows, generate_lineups,
)


# ---------------------------------------------------------------------------
# 1. DST-conflict: aggregate vs. pairwise
# ---------------------------------------------------------------------------


def build_model_with_dst_formulation(
    players: tuple[Player, ...], config: Config, formulation: str,
) -> ModelData:
    """Identical to multi_entry_optimizer.build_model(players, config)
    except the DST-conflict rows use `formulation` instead of the
    hardcoded default ("aggregate")."""
    if not config.avoid_dst_opponents:
        raise ValueError("avoid_dst_opponents must be True to compare DST formulations")
    data = build_model(players, replace(config, avoid_dst_opponents=False))
    for coefficients, lower, upper in dst_conflict_rows(players, formulation):
        data.add(coefficients, lower, upper)
    return data


def _solve_sequential(players: tuple[Player, ...], config: Config, data: ModelData) -> tuple[Lineup, ...]:
    """Mirrors multi_entry_optimizer.generate_lineups's sequential loop,
    but against an
    already-built ModelData (so callers can swap in an alternative
    formulation before solving)."""
    lineups = []
    for number in range(1, config.num_lineups + 1):
        starts, indices, values = data.sparse_rows()
        matrix = csr_matrix((values, indices, starts), shape=(len(data.rows), len(data.costs)))
        result = milp(
            data.costs, integrality=data.integrality,
            bounds=Bounds(data.lower, data.upper),
            constraints=LinearConstraint(matrix, data.row_lower, data.row_upper),
            options={"time_limit": config.time_limit, "mip_rel_gap": 0.0},
        )
        if result.status != 0 or result.x is None:
            raise OptimizationError(f"Lineup {number}/{config.num_lineups}: {result.message}")
        selected = [i for i in range(len(players)) if result.x[i] > 0.5]
        lineups.append(Lineup(tuple(players[i] for i in selected)))
        if number < config.num_lineups:
            data.add({i: 1 for i in selected}, -math.inf, config.max_overlap)
    return tuple(lineups)


def compare_dst_formulations(players: tuple[Player, ...], config: Config) -> dict:
    """Solves the same scenario with both DST formulations and returns
    timing plus an equivalence check (identical lineup ID sequences, since
    both formulations define the same feasible region)."""
    results = {}
    for formulation in ("aggregate", "pairwise"):
        data = build_model_with_dst_formulation(players, config, formulation)
        row_count = len(data.rows)
        start = time.perf_counter()
        lineups = _solve_sequential(players, config, data)
        elapsed = time.perf_counter() - start
        results[formulation] = {"seconds": elapsed, "row_count": row_count,
                                "lineup_ids": [sorted(lu.ids) for lu in lineups]}
    equivalent = results["aggregate"]["lineup_ids"] == results["pairwise"]["lineup_ids"]
    return {"equivalent_lineups": equivalent, **{k: {kk: vv for kk, vv in v.items() if kk != "lineup_ids"}
                                                 for k, v in results.items()}}


# ---------------------------------------------------------------------------
# 2. Cardinality-indicator linking: aggregate vs. disaggregated
# (self-contained, since the disaggregated form is only valid at k==M)
# ---------------------------------------------------------------------------


def cardinality_link_rows(
    member_count: int, k: int, indicator_index: int, member_start: int, formulation: str,
) -> list[tuple[dict[int, float], float, float]]:
    """Rows linking `indicator_index` to "at least k of member_start..
    member_start+member_count-1 are selected", in the aggregate (current
    multi_entry_optimizer) form, valid for any k, or the disaggregated
    form, valid ONLY when k == member_count (seeing why is the point of
    this experiment)."""
    if formulation == "aggregate":
        members = {member_start + j: 1 for j in range(member_count)}
        return [({**members, indicator_index: -k}, 0, math.inf)]
    if formulation == "disaggregated":
        if k != member_count:
            raise ValueError("disaggregated form is only a valid 'at least k' encoding when k == member_count")
        return [({indicator_index: 1, member_start + j: -1}, -math.inf, 0) for j in range(member_count)]
    raise ValueError(f"Unknown formulation: {formulation!r}")


def build_cardinality_benchmark_model(
    group_count: int, member_count: int, seed: int,
) -> tuple[dict, int]:
    """A self-contained MILP with `group_count` independent groups of
    `member_count` binary members each, all sharing one knapsack-style
    budget constraint (so the groups are not trivially separable). Each
    group has an activation indicator worth a random bonus when "at least
    k == member_count" (i.e., all) of its members are selected. Returns a
    (shared_row_template, n_variables) pair; `formulation` rows are added
    separately per comparison run."""
    rng = np.random.default_rng(seed)
    n_members = group_count * member_count
    n_variables = n_members + group_count  # members, then one indicator per group
    member_costs = rng.uniform(1, 5, size=n_members)
    indicator_bonus = rng.uniform(5, 15, size=group_count)
    weights = rng.uniform(1, 3, size=n_members)
    budget = float(weights.sum()) * 0.4  # tight enough to force real tradeoffs
    costs = np.concatenate([-member_costs, -indicator_bonus])  # milp minimizes
    template = {
        "costs": costs, "n_variables": n_variables, "n_members": n_members,
        "budget_row": ({i: float(weights[i]) for i in range(n_members)}, -math.inf, budget),
    }
    return template, n_variables


def compare_cardinality_formulations(
    group_count: int = 80, member_count: int = 5, seed: int = 0, time_limit: float = 60.0,
) -> dict:
    results = {}
    for formulation in ("aggregate", "disaggregated"):
        template, n_variables = build_cardinality_benchmark_model(group_count, member_count, seed)
        rows, row_lower, row_upper = [], [], []
        for row, lower, upper in (template["budget_row"],):
            rows.append(row); row_lower.append(lower); row_upper.append(upper)
        for g in range(group_count):
            indicator_index = template["n_members"] + g
            member_start = g * member_count
            for row, lower, upper in cardinality_link_rows(
                member_count, member_count, indicator_index, member_start, formulation,
            ):
                rows.append(row); row_lower.append(lower); row_upper.append(upper)
        starts = [0]
        indices: list[int] = []
        values: list[float] = []
        for row in rows:
            indices.extend(row)
            values.extend(row.values())
            starts.append(len(indices))
        matrix = csr_matrix((np.asarray(values), np.asarray(indices), np.asarray(starts)),
                            shape=(len(rows), n_variables))
        start_time = time.perf_counter()
        result = milp(
            template["costs"], integrality=np.ones(n_variables),
            bounds=Bounds(np.zeros(n_variables), np.ones(n_variables)),
            constraints=LinearConstraint(matrix, row_lower, row_upper),
            options={"time_limit": time_limit, "mip_rel_gap": 0.0},
        )
        elapsed = time.perf_counter() - start_time
        if result.status != 0 or result.x is None:
            raise OptimizationError(f"Cardinality benchmark ({formulation}): {result.message}")
        results[formulation] = {"seconds": elapsed, "row_count": len(rows),
                                "objective": float(-result.fun)}
    equivalent = abs(results["aggregate"]["objective"] - results["disaggregated"]["objective"]) < 1e-6
    return {"equivalent_objective": equivalent, **results}


# ---------------------------------------------------------------------------
# 3. Two-lineups-at-a-time joint MILP (capped scale)
# ---------------------------------------------------------------------------


def build_joint_pair_model(
    players: tuple[Player, ...], config: Config, prior_selections: list[list[int]],
) -> tuple[ModelData, int]:
    """Builds a single MILP selecting TWO lineups (x, x') at once: each
    independently satisfies every normal feasibility/strategy constraint
    (via two independent copies of build_model's rows), both respect every
    prior lineup's overlap cut, and a new linearized mutual-overlap bound
    (Eq. 5.1/Appendix 8.1) links them. An objective-based symmetry-break
    (lineup x's own projection total >= lineup x''s) halves the redundant
    search space from simply swapping the pair's labels. Returns
    (model, n_single), where n_single is lineup x's column count (so
    lineup x''s columns start at that offset)."""
    single_config = replace(config, num_lineups=1)
    data_a = build_model(players, single_config)
    data_b = build_model(players, single_config)
    n_single = len(data_a.costs)
    n_players = len(players)

    costs = np.concatenate([data_a.costs, data_b.costs])
    lower = np.concatenate([data_a.lower, data_b.lower])
    upper = np.concatenate([data_a.upper, data_b.upper])
    integrality = np.concatenate([data_a.integrality, data_b.integrality])
    rows: list[dict[int, float]] = list(data_a.rows)
    row_lower = list(data_a.row_lower)
    row_upper = list(data_a.row_upper)
    for row, rlo, rhi in zip(data_b.rows, data_b.row_lower, data_b.row_upper):
        rows.append({i + n_single: c for i, c in row.items()})
        row_lower.append(rlo)
        row_upper.append(rhi)

    for selected in prior_selections:
        rows.append({i: 1 for i in selected}); row_lower.append(-math.inf); row_upper.append(config.max_overlap)
        rows.append({n_single + i: 1 for i in selected}); row_lower.append(-math.inf); row_upper.append(config.max_overlap)

    # Linearized mutual overlap: z_j = x_j * x'_j for every player j.
    z_start = len(costs)
    costs = np.concatenate([costs, np.zeros(n_players)])
    lower = np.concatenate([lower, np.zeros(n_players)])
    upper = np.concatenate([upper, np.ones(n_players)])
    integrality = np.concatenate([integrality, np.zeros(n_players)])  # continuous; exact for binary x,x'
    for j in range(n_players):
        z = z_start + j
        rows.append({z: 1, j: -1}); row_lower.append(-math.inf); row_upper.append(0)
        rows.append({z: 1, n_single + j: -1}); row_lower.append(-math.inf); row_upper.append(0)
        rows.append({z: 1, j: -1, n_single + j: -1}); row_lower.append(-1); row_upper.append(math.inf)
    rows.append({z_start + j: 1 for j in range(n_players)})
    row_lower.append(-math.inf); row_upper.append(config.max_overlap)

    # Symmetry break: lineup A's projection total >= lineup B's.
    projections = {i: p.projection for i, p in enumerate(players)}
    rows.append({**projections, **{n_single + i: -proj for i, proj in projections.items()}})
    row_lower.append(0.0); row_upper.append(math.inf)

    return ModelData(costs, lower, upper, integrality, rows, row_lower, row_upper), n_single


def generate_lineups_two_at_a_time(
    players: tuple[Player, ...], config: Config, pair_time_limit: float | None = None,
) -> tuple[Lineup, ...]:
    """Generates config.num_lineups lineups two at a time via
    build_joint_pair_model, falling back to a single solve for an odd
    final lineup. pair_time_limit bounds each PAIR's solve explicitly
    (the paper reports open-source solvers taking hours per pair at
    scale; never let one experimental pair run unbounded)."""
    players = tuple(players)
    pair_time_limit = config.time_limit if pair_time_limit is None else pair_time_limit
    lineups: list[Lineup] = []
    prior_selections: list[list[int]] = []
    remaining = config.num_lineups
    while remaining > 0:
        if remaining == 1:
            single_config = replace(config, num_lineups=1)
            data = build_model(players, single_config)
            for selected in prior_selections:
                data.add({i: 1 for i in selected}, -math.inf, config.max_overlap)
            lineup, = _solve_sequential(players, single_config, data)
            selected = [i for i, p in enumerate(players) if p.id in lineup.ids]
            lineups.append(lineup)
            prior_selections.append(selected)
            remaining -= 1
            continue
        data, n_single = build_joint_pair_model(
            players, replace(config, time_limit=pair_time_limit), prior_selections,
        )
        starts, indices, values = data.sparse_rows()
        matrix = csr_matrix((values, indices, starts), shape=(len(data.rows), len(data.costs)))
        result = milp(
            data.costs, integrality=data.integrality,
            bounds=Bounds(data.lower, data.upper),
            constraints=LinearConstraint(matrix, data.row_lower, data.row_upper),
            options={"time_limit": pair_time_limit, "mip_rel_gap": 0.0},
        )
        if result.status != 0 or result.x is None:
            raise OptimizationError(f"Joint pair solve failed: {result.message}")
        selected_a = [i for i in range(len(players)) if result.x[i] > 0.5]
        selected_b = [i for i in range(len(players)) if result.x[n_single + i] > 0.5]
        lineups.append(Lineup(tuple(players[i] for i in selected_a)))
        lineups.append(Lineup(tuple(players[i] for i in selected_b)))
        prior_selections.append(selected_a)
        prior_selections.append(selected_b)
        remaining -= 2
    return tuple(lineups)


def compare_sequential_vs_two_at_a_time(
    players: tuple[Player, ...], config: Config, pair_time_limit: float,
) -> dict:
    start = time.perf_counter()
    sequential = generate_lineups(players, config)
    sequential_seconds = time.perf_counter() - start

    start = time.perf_counter()
    joint = generate_lineups_two_at_a_time(players, config, pair_time_limit)
    joint_seconds = time.perf_counter() - start

    return {
        "sequential_seconds": sequential_seconds, "joint_seconds": joint_seconds,
        "sequential_total_projection": sum(lu.projection for lu in sequential),
        "joint_total_projection": sum(lu.projection for lu in joint),
        "joint_at_least_as_slow": joint_seconds >= sequential_seconds,
    }


# ---------------------------------------------------------------------------
# 4. SciPy vs. native HiGHS on an EXPANDED model (Phase 1 + Phase 3)
# ---------------------------------------------------------------------------


def expanded_moments(players: tuple[Player, ...]) -> Moments:
    """A same-team-only (block-diagonal) positive-correlation factor model,
    for exercising Phase 3's variance_floor/covariance_cap machinery in
    this benchmark without triggering the dense-moments scaling warning
    from Phase 3 (a fully dense factor loading would make essentially
    every player pair nonzero; real NFL correlation structure is expected
    to be sparse/block-diagonal by team and game in any case). Not
    validated against real data."""
    ids = tuple(p.id for p in players)
    n = len(players)
    teams = sorted({p.team for p in players})
    sigma = np.eye(n) * 4.0
    for team in teams:
        loading = np.array([1.0 if p.team == team else 0.0 for p in players])
        sigma += 0.3 * np.outer(loading, loading)
    covariance = {tuple(sorted((ids[i], ids[j]))): float(sigma[i, j])
                 for i in range(n) for j in range(i + 1, n) if sigma[i, j] != 0}
    return Moments(
        scoring_version="v1", as_of="2024-01-01T00:00:00Z", player_ids=ids,
        variance=tuple(float(sigma[i, i]) for i in range(n)), covariance=covariance,
        missing_covariance_policy="assume_independent",
    )


def expanded_config(players: tuple[Player, ...], num_lineups: int, time_limit: float = 20.0) -> Config:
    teams = sorted({p.team for p in players})
    moments = expanded_moments(players)
    variance_floor = float(np.median(moments.variance)) * 9 * 0.5
    return Config(
        num_lineups=num_lineups, max_overlap=6, stack=2, bring_back=True,
        strategic_max_teams=min(6, len(teams)), moments=moments,
        variance_floor=variance_floor, time_limit=time_limit,
    )


def compare_solvers_on_expanded_model(
    players: tuple[Player, ...], num_lineups: int, seed: int, repeats: int, time_limit: float = 20.0,
) -> dict:
    config = expanded_config(players, num_lineups, time_limit)
    runners = {"highspy": native_lineups, "scipy_milp": generate_lineups}
    timings: dict[str, list[float]] = {name: [] for name in runners}
    for runner in runners.values():
        runner(players, replace(config, num_lineups=1))
    for repeat in range(repeats):
        names = list(runners)
        if repeat % 2:
            names.reverse()
        for name in names:
            start = time.perf_counter()
            runners[name](players, config)
            timings[name].append(time.perf_counter() - start)
    medians = {name: statistics.median(samples) for name, samples in timings.items()}
    return {"seconds": timings, "median_seconds": medians,
            "fastest_measured": min(medians, key=lambda name: medians[name])}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _log(message: str) -> None:
    """Progress logging to stderr, so a long run is visibly making
    progress rather than appearing to hang (stdout stays clean JSON)."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True, file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teams", type=int, default=8)
    parser.add_argument("--lineups", type=int, nargs="+", default=[1, 20, 50])
    parser.add_argument("--two-at-a-time-lineups", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--pair-time-limit", type=float, default=20.0)
    parser.add_argument("--expanded-model-time-limit", type=float, default=20.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wall-clock-budget", type=float, default=600.0,
                       help="Stop starting new scales once this many seconds have elapsed "
                            "(a bounded-experiment safety valve; already-started scales finish).")
    args = parser.parse_args()
    if args.teams < 4 or args.teams % 2:
        parser.error("Use an even team count >= 4")
    players = synthetic_pool(args.teams, args.seed)
    budget_start = time.perf_counter()

    def within_budget() -> bool:
        return time.perf_counter() - budget_start < args.wall_clock_budget

    report: dict = {
        "python": platform.python_version(), "platform": platform.platform(),
        "highspy": highspy.Highs().version(), "scipy": scipy.__version__,
        "numpy": np.__version__, "players": len(players), "seed": args.seed,
    }

    _log("Starting DST formulation comparison")
    report["dst_formulation"] = {}
    for n in args.lineups:
        if not within_budget():
            _log(f"Wall-clock budget exceeded; skipping DST comparison for n={n}")
            break
        _log(f"  DST comparison: n={n}")
        report["dst_formulation"][str(n)] = compare_dst_formulations(
            players, Config(num_lineups=n, max_overlap=6),
        )

    _log("Starting cardinality-link formulation comparison")
    report["cardinality_formulation"] = compare_cardinality_formulations(seed=args.seed)

    _log("Starting two-at-a-time comparison (capped scale)")
    report["two_at_a_time"] = {}
    for n in args.two_at_a_time_lineups:
        if not within_budget():
            _log(f"Wall-clock budget exceeded; skipping two-at-a-time for n={n}")
            break
        _log(f"  Two-at-a-time: n={n}")
        report["two_at_a_time"][str(n)] = compare_sequential_vs_two_at_a_time(
            players, Config(num_lineups=n, max_overlap=6, time_limit=args.pair_time_limit),
            args.pair_time_limit,
        )

    _log("Starting expanded-model (Phase 1 + Phase 3) solver comparison")
    report["expanded_model_solver_comparison"] = {}
    for n in args.lineups:
        if not within_budget():
            _log(f"Wall-clock budget exceeded; skipping expanded-model comparison for n={n}")
            break
        _log(f"  Expanded-model solver comparison: n={n}")
        report["expanded_model_solver_comparison"][str(n)] = compare_solvers_on_expanded_model(
            players, n, args.seed, args.repeats, args.expanded_model_time_limit,
        )
        _log(f"    done in {report['expanded_model_solver_comparison'][str(n)]['median_seconds']}")

    _log("All experiments complete")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
