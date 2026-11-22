"""Tests for formulation_experiments.py (Phase 5).

Establishes the REQUIRED acceptance property for every alternative
formulation here: "equivalent feasible integer solutions" relative to
multi_entry_optimizer.lineups's existing default, proven either by
exhaustive feasible-region
comparison on a tiny pool or by a provable always-true MILP inequality
(the joint two-lineup model's objective can never be worse than the
sequential one, since sequential is always a feasible point of the joint
problem). Timing numbers are reported/smoke-tested, not hard-asserted,
since relative solver speed is machine-dependent; the full sweep used to
draw an actual conclusion lives in formulation_experiments.main() and is
reported in PAPER_IMPLEMENTATION_PLAN.md.
"""

import itertools
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from multi_entry_optimizer import Config, Player, dst_conflict_rows, load_players
from formulation_experiments import (
    build_cardinality_benchmark_model, build_joint_pair_model,
    build_model_with_dst_formulation, cardinality_link_rows, compare_cardinality_formulations,
    compare_dst_formulations, compare_solvers_on_expanded_model,
    compare_sequential_vs_two_at_a_time, generate_lineups_two_at_a_time, synthetic_pool,
)


# ---------------------------------------------------------------------------
# 1. DST conflict: aggregate vs. pairwise define the identical feasible set
# ---------------------------------------------------------------------------


@pytest.fixture
def two_dst_pool():
    pool = load_players(Path(__file__).resolve().parents[1] / "examples" / "players.csv")
    ids = {"A_QB", "A_RB1", "A_RB2", "A_WR1", "A_WR2", "A_TE",
          "B_WR1", "C_RB1", "C_WR1", "C_TE", "C_DST", "D_DST", "D_WR1"}
    return tuple(p for p in pool if p.id in ids)


def _rows_satisfied(rows, selected: set[int]) -> bool:
    for coefficients, lower, upper in rows:
        value = sum(coefficient for i, coefficient in coefficients.items() if i in selected)
        if not lower - 1e-9 <= value <= upper + 1e-9:
            return False
    return True


def test_dst_conflict_rows_define_identical_feasible_region_exhaustive(two_dst_pool):
    aggregate_rows = dst_conflict_rows(two_dst_pool, "aggregate")
    pairwise_rows = dst_conflict_rows(two_dst_pool, "pairwise")
    assert aggregate_rows and pairwise_rows  # sanity: the fixture actually exercises DST conflicts
    disagreements = 0
    for combo in itertools.combinations(range(len(two_dst_pool)), 9):
        selected = set(combo)
        agg_ok = _rows_satisfied(aggregate_rows, selected)
        pair_ok = _rows_satisfied(pairwise_rows, selected)
        assert agg_ok == pair_ok
        disagreements += agg_ok != pair_ok
    assert disagreements == 0


def test_dst_conflict_rows_rejects_unknown_formulation(two_dst_pool):
    with pytest.raises(ValueError, match="Unknown DST formulation"):
        dst_conflict_rows(two_dst_pool, "not-a-formulation")


@pytest.mark.parametrize("num_lineups,max_overlap", [(1, 6), (5, 6), (5, 3)])
def test_compare_dst_formulations_equivalent_on_realistic_pool(num_lineups, max_overlap):
    players = synthetic_pool(8, seed=3)
    config = Config(num_lineups=num_lineups, max_overlap=max_overlap)
    result = compare_dst_formulations(players, config)
    assert result["equivalent_lineups"]
    assert result["aggregate"]["row_count"] >= 0
    assert result["pairwise"]["row_count"] >= result["aggregate"]["row_count"]  # disaggregation adds rows


def test_build_model_with_dst_formulation_requires_avoid_dst_opponents():
    players = synthetic_pool(4, seed=0)
    with pytest.raises(ValueError, match="avoid_dst_opponents must be True"):
        build_model_with_dst_formulation(players, Config(avoid_dst_opponents=False), "pairwise")


# ---------------------------------------------------------------------------
# 2. Cardinality-indicator linking: aggregate vs. disaggregated (k == M only)
# ---------------------------------------------------------------------------


def test_cardinality_link_rows_disaggregated_requires_k_equals_member_count():
    with pytest.raises(ValueError, match="only a valid"):
        cardinality_link_rows(5, 3, indicator_index=10, member_start=0, formulation="disaggregated")


def test_cardinality_link_rows_rejects_unknown_formulation():
    with pytest.raises(ValueError, match="Unknown formulation"):
        cardinality_link_rows(5, 5, indicator_index=10, member_start=0, formulation="bogus")


def test_cardinality_formulations_define_identical_feasible_region_exhaustive():
    # One group of 3 members (k=3=all) plus a free indicator: the indicator
    # can be 1 only when all 3 members are selected, under EITHER formulation.
    member_count, k = 3, 3
    indicator_index = member_count
    n = member_count + 1
    for formulation in ("aggregate", "disaggregated"):
        rows = cardinality_link_rows(member_count, k, indicator_index, 0, formulation)
        for members in itertools.product((0, 1), repeat=member_count):
            for indicator in (0, 1):
                selected = {i for i, v in enumerate(members) if v == 1}
                if indicator:
                    selected.add(indicator_index)
                feasible = _rows_satisfied(rows, selected)
                all_members_selected = all(members)
                if indicator == 1:
                    assert feasible == all_members_selected, (formulation, members, indicator)
                else:
                    assert feasible  # indicator=0 is never blocked by these rows


def test_compare_cardinality_formulations_equivalent_objective():
    result = compare_cardinality_formulations(group_count=15, member_count=3, seed=2)
    assert result["equivalent_objective"]
    assert result["disaggregated"]["row_count"] > result["aggregate"]["row_count"]


def test_build_cardinality_benchmark_model_structure():
    template, n_variables = build_cardinality_benchmark_model(group_count=5, member_count=4, seed=0)
    assert n_variables == 5 * 4 + 5
    assert template["n_members"] == 20


# ---------------------------------------------------------------------------
# 3. Two-lineups-at-a-time joint MILP
# ---------------------------------------------------------------------------


def test_joint_pair_combined_objective_never_worse_than_sequential():
    # Any feasible sequential pair is also a feasible point of the joint
    # problem (same per-lineup constraints, plus overlap <= max_overlap
    # which the sequential pair already satisfies), so the joint optimum's
    # combined objective can never be LOWER than the sequential pair's.
    players = synthetic_pool(6, seed=5)
    config = Config(num_lineups=2, max_overlap=6, time_limit=30)
    comparison = compare_sequential_vs_two_at_a_time(players, config, pair_time_limit=30)
    assert comparison["joint_total_projection"] >= comparison["sequential_total_projection"] - 1e-6


def test_joint_pair_model_respects_individual_feasibility_and_mutual_overlap():
    players = synthetic_pool(6, seed=7)
    config = Config(num_lineups=2, max_overlap=3, time_limit=30)
    lineups = generate_lineups_two_at_a_time(players, config, pair_time_limit=30)
    assert len(lineups) == 2
    for lineup in lineups:
        lineup.slot_players()  # raises if not a valid classic roster
        counts = Counter(p.position for p in lineup.players)
        assert counts["QB"] == 1 and counts["DST"] == 1
        assert 2 <= counts["RB"] <= 3 and 3 <= counts["WR"] <= 4 and 1 <= counts["TE"] <= 2
        assert sum(p.salary for p in lineup.players) <= 50_000
    overlap = len(lineups[0].ids & lineups[1].ids)
    assert overlap <= config.max_overlap


def test_generate_lineups_two_at_a_time_matches_sequential_lineup_count():
    players = synthetic_pool(6, seed=9)
    for num_lineups in (1, 2, 3, 4, 5):
        config = Config(num_lineups=num_lineups, max_overlap=6, time_limit=30)
        lineups = generate_lineups_two_at_a_time(players, config, pair_time_limit=30)
        assert len(lineups) == num_lineups


def test_generate_lineups_two_at_a_time_respects_overlap_across_all_pairs():
    # Confirms overlap discipline holds not just within a solved pair, but
    # across lineups produced in DIFFERENT pair-solves too (via prior cuts).
    players = synthetic_pool(8, seed=11)
    config = Config(num_lineups=4, max_overlap=4, time_limit=30)
    lineups = generate_lineups_two_at_a_time(players, config, pair_time_limit=30)
    for a, b in itertools.combinations(lineups, 2):
        assert len(a.ids & b.ids) <= config.max_overlap


def test_build_joint_pair_model_symmetry_break_orders_by_projection():
    players = synthetic_pool(6, seed=13)
    config = Config(num_lineups=2, max_overlap=6, time_limit=30)
    lineups = generate_lineups_two_at_a_time(players, config, pair_time_limit=30)
    assert lineups[0].projection >= lineups[1].projection - 1e-6


# ---------------------------------------------------------------------------
# 4. SciPy vs. native HiGHS on an expanded model: smoke test only
# ---------------------------------------------------------------------------


def test_compare_solvers_on_expanded_model_smoke():
    players = synthetic_pool(6, seed=15)
    result = compare_solvers_on_expanded_model(players, num_lineups=2, seed=15, repeats=1)
    assert set(result["seconds"]) == {"highspy", "scipy_milp"}
    assert result["fastest_measured"] in ("highspy", "scipy_milp")
    assert all(seconds > 0 for samples in result["seconds"].values() for seconds in samples)
