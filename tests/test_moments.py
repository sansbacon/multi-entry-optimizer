"""Tests for Phase 3's moment-aware constraints in
multi_entry_optimizer.lineups.

All fixtures here use synthetic, hand-constructed covariance structures
(factor models built for testability), not estimates from real NFL data.
These tests establish mathematical/MILP correctness only; they do not
establish that any variance/covariance target improves real contest
performance (see PAPER_IMPLEMENTATION_PLAN.md Phase 3/4).
"""

import itertools
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from multi_entry_optimizer import (
    Config, Lineup, Moments, OptimizationError, Player, build_model,
    generate_lineups, lineup_covariance, lineup_variance, load_players,
    realized_correlation, validate_moments,
)


# ---------------------------------------------------------------------------
# Factor-model fixture: reuses the Phase 1 13-player exhaustive subset, with
# a hand-constructed, guaranteed-PSD covariance matrix built as
# diag(idiosyncratic variances) + sum of (factor_variance * loading @ loading.T).
# Game 1 (team A vs B): same-team pairs get +phi1, opposing-team pairs -phi1.
# Game 2 (team C vs D): same-team pairs get +phi2, opposing-team pairs -phi2.
# Cross-game pairs get exactly zero covariance (no shared factor).
# ---------------------------------------------------------------------------

IDS = ("A_QB", "A_RB1", "A_RB2", "A_WR1", "A_WR2", "A_TE",
       "B_WR1", "C_RB1", "C_WR1", "C_TE", "C_DST", "D_DST", "D_WR1")
VARIANCE = {"A_QB": 4.0, "A_RB1": 2.0, "A_RB2": 2.2, "A_WR1": 1.6, "A_WR2": 1.7, "A_TE": 1.3,
            "B_WR1": 1.5, "C_RB1": 2.1, "C_WR1": 1.9, "C_TE": 1.4, "C_DST": 1.0, "D_DST": 1.1,
            "D_WR1": 1.5}
GAME1 = {"A_QB": 1.0, "A_RB1": 1.0, "A_RB2": 1.0, "A_WR1": 1.0, "A_WR2": 1.0, "A_TE": 1.0, "B_WR1": -1.0}
GAME2 = {"C_RB1": 1.0, "C_WR1": 1.0, "C_TE": 1.0, "C_DST": 1.0, "D_DST": -1.0, "D_WR1": -1.0}
PHI1, PHI2 = 0.4, 0.4


def dense_sigma(ids):
    f1 = np.array([GAME1.get(i, 0.0) for i in ids])
    f2 = np.array([GAME2.get(i, 0.0) for i in ids])
    return np.diag([VARIANCE[i] for i in ids]) + PHI1 * np.outer(f1, f1) + PHI2 * np.outer(f2, f2)


def build_moments(ids=IDS, missing_covariance_policy="assume_independent"):
    sigma = dense_sigma(ids)
    covariance = {}
    for i, a in enumerate(ids):
        for j in range(i + 1, len(ids)):
            b = ids[j]
            if sigma[i, j] != 0:
                key = tuple(sorted((a, b)))
                covariance[key] = float(sigma[i, j])
    return Moments(
        scoring_version="v1", as_of="2024-09-08T00:00:00Z", player_ids=ids,
        # Note: a factor model's own diagonal includes phi*loading^2, so this
        # is NOT simply VARIANCE[i]; use sigma's actual diagonal to stay
        # internally consistent with the covariance entries built above.
        variance=tuple(float(sigma[i, i]) for i in range(len(ids))), covariance=covariance,
        missing_covariance_policy=missing_covariance_policy,
    )


@pytest.fixture
def pool():
    return load_players(Path(__file__).resolve().parents[1] / "examples" / "players.csv")


@pytest.fixture
def small_pool(pool):
    return tuple(p for p in pool if p.id in IDS)


@pytest.fixture
def moments():
    return build_moments()


# ---------------------------------------------------------------------------
# Moments data-contract validation
# ---------------------------------------------------------------------------


def test_validate_moments_accepts_well_formed_data(small_pool, moments):
    sigma = validate_moments(small_pool, moments)
    assert sigma.shape == (13, 13)
    index = {p.id: i for i, p in enumerate(small_pool)}
    # A factor model's own diagonal is idiosyncratic variance + phi*loading^2.
    assert sigma[index["A_QB"], index["A_QB"]] == pytest.approx(VARIANCE["A_QB"] + PHI1)
    # Same-team (game 1) pair: positive covariance.
    assert sigma[index["A_QB"], index["A_RB1"]] == pytest.approx(PHI1)
    # Opposing-team (game 1) pair: negative covariance.
    assert sigma[index["A_QB"], index["B_WR1"]] == pytest.approx(-PHI1)
    # Cross-game pair: exactly zero (no shared factor).
    assert sigma[index["A_QB"], index["C_RB1"]] == 0.0


def test_validate_moments_rejects_missing_diagonal(small_pool, moments):
    incomplete = replace(moments, player_ids=moments.player_ids[:-1],
                         variance=moments.variance[:-1])
    with pytest.raises(ValueError, match="missing a variance entry"):
        validate_moments(small_pool, incomplete)


def test_validate_moments_rejects_duplicate_player_ids(small_pool, moments):
    duplicated = replace(moments, player_ids=moments.player_ids + (moments.player_ids[0],),
                         variance=moments.variance + (moments.variance[0],))
    with pytest.raises(ValueError, match="duplicate player IDs"):
        validate_moments(small_pool, duplicated)


def test_validate_moments_rejects_misaligned_variance_length(small_pool, moments):
    misaligned = replace(moments, variance=moments.variance[:-1])
    with pytest.raises(ValueError, match="align one-to-one"):
        validate_moments(small_pool, misaligned)


@pytest.mark.parametrize("bad_variance", [float("nan"), float("inf"), -1.0])
def test_validate_moments_rejects_non_finite_or_negative_variance(small_pool, moments, bad_variance):
    variance = list(moments.variance)
    variance[0] = bad_variance
    bad = replace(moments, variance=tuple(variance))
    with pytest.raises(ValueError):
        validate_moments(small_pool, bad)


def test_validate_moments_rejects_non_finite_covariance(small_pool, moments):
    covariance = dict(moments.covariance)
    key = next(iter(covariance))
    covariance[key] = float("nan")
    bad = replace(moments, covariance=covariance)
    with pytest.raises(ValueError, match="Non-finite covariance"):
        validate_moments(small_pool, bad)


def test_validate_moments_rejects_self_pair_covariance(small_pool, moments):
    covariance = dict(moments.covariance)
    covariance[("A_QB", "A_QB")] = 1.0
    bad = replace(moments, covariance=covariance)
    with pytest.raises(ValueError, match="self-pair"):
        validate_moments(small_pool, bad)


def test_validate_moments_rejects_ambiguous_key_ordering(small_pool, moments):
    # "B_WR1" > "A_QB" lexically, so this key is stored in the wrong order.
    covariance = dict(moments.covariance)
    covariance[("B_WR1", "A_QB")] = -0.4
    bad = replace(moments, covariance=covariance)
    with pytest.raises(ValueError, match="sorted .lesser, greater. ID pair"):
        validate_moments(small_pool, bad)


def test_validate_moments_rejects_unknown_covariance_id(small_pool, moments):
    covariance = dict(moments.covariance)
    covariance[("A_QB", "ZZZ_UNKNOWN")] = 0.1
    bad = replace(moments, covariance=covariance)
    with pytest.raises(ValueError, match="unknown player ID"):
        validate_moments(small_pool, bad)


def test_validate_moments_rejects_unsupported_missing_covariance_policy(small_pool, moments):
    bad = replace(moments, missing_covariance_policy="silently_zero")
    with pytest.raises(ValueError, match="Unsupported missing_covariance_policy"):
        validate_moments(small_pool, bad)


def test_validate_moments_rejects_non_psd_matrix(small_pool, moments):
    # A huge positive off-diagonal entry relative to the diagonal variances
    # breaks positive semidefiniteness.
    covariance = dict(moments.covariance)
    covariance[("A_QB", "A_RB1")] = 100.0
    bad = replace(moments, covariance=covariance)
    with pytest.raises(ValueError, match="not positive semidefinite"):
        validate_moments(small_pool, bad)


def test_validate_moments_rejects_invalid_psd_tolerance(small_pool, moments):
    for tolerance in (-1.0, float("nan")):
        bad = replace(moments, psd_tolerance=tolerance)
        with pytest.raises(ValueError, match="psd_tolerance"):
            validate_moments(small_pool, bad)


# ---------------------------------------------------------------------------
# Config-level validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("changes", [
    {"variance_floor": 1.0}, {"covariance_cap": 1.0}, {"correlation_ceiling": 0.5},
])
def test_moment_fields_require_moments_to_be_set(pool, changes):
    with pytest.raises(ValueError, match="requires moments to be set"):
        generate_lineups(pool, replace(Config(), **changes))


def test_correlation_ceiling_requires_variance_floor(pool, moments):
    with pytest.raises(ValueError, match="requires variance_floor"):
        generate_lineups(pool, Config(moments=moments, correlation_ceiling=0.5))


def test_correlation_ceiling_range(pool, moments):
    for ceiling in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            generate_lineups(pool, Config(moments=moments, variance_floor=1.0, correlation_ceiling=ceiling))


def test_covariance_cap_and_correlation_ceiling_are_mutually_exclusive(pool, moments):
    with pytest.raises(ValueError, match="at most one"):
        generate_lineups(pool, Config(moments=moments, variance_floor=1.0,
                                      covariance_cap=1.0, correlation_ceiling=0.5))


@pytest.mark.parametrize("changes", [{"variance_floor": -1.0}, {"variance_floor": float("nan")},
                                     {"covariance_cap": float("nan")}])
def test_moment_field_numeric_validation(pool, moments, changes):
    with pytest.raises(ValueError):
        generate_lineups(pool, replace(Config(moments=moments), **changes))


# ---------------------------------------------------------------------------
# Disabled moment mode preserves current behavior (regression)
# ---------------------------------------------------------------------------


def test_disabled_moment_mode_unchanged(pool):
    data = build_model(pool, Config())
    assert data.sigma is None
    assert np.all(data.integrality == 1)
    baseline = generate_lineups(pool, Config(num_lineups=3, max_overlap=6))
    explicit_default = generate_lineups(pool, Config(num_lineups=3, max_overlap=6, moments=None,
                                                      variance_floor=None, covariance_cap=None,
                                                      correlation_ceiling=None))
    assert [l.ids for l in baseline] == [l.ids for l in explicit_default]


# ---------------------------------------------------------------------------
# Variance floor: exhaustive tiny-pool comparison with direct quadratic eval
# ---------------------------------------------------------------------------


def lineup_quadratic_variance(ids, sigma, id_index):
    idxs = [id_index[i] for i in ids]
    return float(sigma[np.ix_(idxs, idxs)].sum())


def exhaustive_best_with_variance_floor(small_pool, sigma, id_index, variance_floor):
    best = None
    for combo in itertools.combinations(small_pool, 9):
        counts = Counter(p.position for p in combo)
        if (counts["QB"] != 1 or counts["DST"] != 1 or not 2 <= counts["RB"] <= 3
                or not 3 <= counts["WR"] <= 4 or not 1 <= counts["TE"] <= 2
                or sum(p.salary for p in combo) > 50_000):
            continue
        teams, games = {p.team for p in combo}, {p.game for p in combo}
        if len(teams) < 2 or len(games) < 2:
            continue
        dst = next(p for p in combo if p.position == "DST")
        if any(p.team == dst.opponent and p.position != "DST" for p in combo):
            continue
        variance = lineup_quadratic_variance([p.id for p in combo], sigma, id_index)
        if variance < variance_floor:
            continue
        projection = sum(p.projection for p in combo)
        if best is None or projection > best[0]:
            best = (projection, variance, combo)
    assert best is not None, "expected at least one feasible lineup for this floor"
    return best


@pytest.mark.parametrize("variance_floor,expected_projection,expected_variance", [
    (30.0, 150.7, 33.6), (34.0, 146.8, 35.1),
])
def test_variance_floor_matches_exhaustive_quadratic_evaluation(
    small_pool, moments, variance_floor, expected_projection, expected_variance,
):
    sigma = dense_sigma(IDS)
    id_index = {pid: i for i, pid in enumerate(IDS)}
    # Sanity-check the floor actually excludes the globally unconstrained
    # best lineup (153.1, variance 29.0), i.e. the floor truly binds.
    unconstrained_best = exhaustive_best_with_variance_floor(small_pool, sigma, id_index, 0.0)
    assert unconstrained_best[0] == pytest.approx(153.1)
    assert unconstrained_best[1] < variance_floor

    expected = exhaustive_best_with_variance_floor(small_pool, sigma, id_index, variance_floor)
    assert expected[0] == pytest.approx(expected_projection)
    assert expected[1] == pytest.approx(expected_variance, abs=0.05)

    config = Config(moments=moments, variance_floor=variance_floor)
    result, = generate_lineups(small_pool, config)
    assert result.projection == pytest.approx(expected_projection)
    assert lineup_variance(result, moments) >= variance_floor - 1e-6
    assert lineup_variance(result, moments) == pytest.approx(expected_variance, abs=0.05)


def test_variance_floor_infeasible_above_max_achievable_variance(small_pool, moments):
    with pytest.raises(OptimizationError, match="Infeasible"):
        generate_lineups(small_pool, Config(moments=moments, variance_floor=1000.0))


# ---------------------------------------------------------------------------
# lineup_variance / lineup_covariance / realized_correlation: unequal
# variances, positive covariance (same team), negative covariance (opposing
# team), and zero covariance (unrelated game)
# ---------------------------------------------------------------------------


def player_by_id(small_pool, player_id):
    return next(p for p in small_pool if p.id == player_id)


def test_lineup_variance_unequal_variances(small_pool, moments):
    qb_only = Lineup((player_by_id(small_pool, "A_QB"),))
    wr_only = Lineup((player_by_id(small_pool, "B_WR1"),))
    sigma = dense_sigma(IDS)
    id_index = {pid: i for i, pid in enumerate(IDS)}
    # A factor model's own variance is idiosyncratic + phi*loading^2, not
    # simply the raw idiosyncratic VARIANCE value.
    assert lineup_variance(qb_only, moments) == pytest.approx(sigma[id_index["A_QB"], id_index["A_QB"]])
    assert lineup_variance(wr_only, moments) == pytest.approx(sigma[id_index["B_WR1"], id_index["B_WR1"]])
    assert lineup_variance(qb_only, moments) != lineup_variance(wr_only, moments)


def test_lineup_covariance_positive_for_same_team(small_pool, moments):
    a = Lineup((player_by_id(small_pool, "A_QB"),))
    b = Lineup((player_by_id(small_pool, "A_RB1"),))
    assert lineup_covariance(a, b, moments) == pytest.approx(PHI1)


def test_lineup_covariance_negative_for_opposing_team(small_pool, moments):
    a = Lineup((player_by_id(small_pool, "A_QB"),))
    b = Lineup((player_by_id(small_pool, "B_WR1"),))
    assert lineup_covariance(a, b, moments) == pytest.approx(-PHI1)


def test_lineup_covariance_zero_for_unrelated_game(small_pool, moments):
    a = Lineup((player_by_id(small_pool, "A_QB"),))
    b = Lineup((player_by_id(small_pool, "C_RB1"),))
    assert lineup_covariance(a, b, moments) == 0.0


def test_lineup_covariance_disjoint_players_can_be_strongly_positive(small_pool, moments):
    # Acceptance criterion: "two disjoint lineups with positive covariance:
    # zero player overlap does not imply independence." These two lineups
    # share NO players, yet all four are on team A (same factor).
    a = Lineup((player_by_id(small_pool, "A_QB"), player_by_id(small_pool, "A_RB1")))
    b = Lineup((player_by_id(small_pool, "A_RB2"), player_by_id(small_pool, "A_WR1")))
    assert not a.ids & b.ids
    # 2x2 = 4 cross pairs, each contributing +PHI1.
    assert lineup_covariance(a, b, moments) == pytest.approx(4 * PHI1)
    assert lineup_covariance(a, b, moments) > 0


def test_realized_correlation_sign_and_bounds(small_pool, moments):
    a = Lineup((player_by_id(small_pool, "A_QB"),))
    same_team = Lineup((player_by_id(small_pool, "A_RB1"),))
    opposing_team = Lineup((player_by_id(small_pool, "B_WR1"),))
    assert realized_correlation(a, same_team, moments) > 0
    assert realized_correlation(a, opposing_team, moments) < 0
    for lineup_b in (same_team, opposing_team):
        assert -1.0 <= realized_correlation(a, lineup_b, moments) <= 1.0


def test_realized_correlation_requires_positive_variance():
    zero_variance_moments = Moments(
        scoring_version="v1", as_of="2024-01-01T00:00:00Z", player_ids=("x", "y"),
        variance=(0.0, 1.0), covariance={}, missing_covariance_policy="assume_independent",
    )
    zero_var_player = Player("x", "X", "QB", "A", "B", 5000, 10.0)
    other_player = Player("y", "Y", "RB", "A", "B", 5000, 10.0)
    with pytest.raises(ValueError, match="strictly positive modeled variance"):
        realized_correlation(Lineup((zero_var_player,)), Lineup((other_player,)), zero_variance_moments)


# ---------------------------------------------------------------------------
# Equal variances + zero off-diagonals: covariance cuts reduce to scaled
# overlap cuts
# ---------------------------------------------------------------------------


def test_covariance_cap_reduces_to_scaled_overlap_with_equal_variance_no_covariance(pool):
    variance_value = 2.0
    equal_variance_moments = Moments(
        scoring_version="v1", as_of="2024-01-01T00:00:00Z",
        player_ids=tuple(p.id for p in pool), variance=tuple(variance_value for _ in pool),
        covariance={}, missing_covariance_policy="assume_independent",
    )
    overlap_limit = 6
    via_overlap = generate_lineups(pool, Config(num_lineups=4, max_overlap=overlap_limit))
    via_covariance_cap = generate_lineups(pool, Config(
        num_lineups=4, max_overlap=8, moments=equal_variance_moments,
        covariance_cap=variance_value * overlap_limit,
    ))
    assert [l.ids for l in via_overlap] == [l.ids for l in via_covariance_cap]
    assert [l.projection for l in via_overlap] == [l.projection for l in via_covariance_cap]


# ---------------------------------------------------------------------------
# Disjoint-but-correlated MILP demonstration: covariance_cap rejects a
# disjoint lineup that max_overlap alone would permit
# ---------------------------------------------------------------------------


def correlated_versus_independent_pool():
    """20 highly mutually-correlated team-A players (enough depth for two
    fully disjoint 8-player lineups), 10 mutually-independent team-E
    players (lower projections), and 2 DST candidates so two lineups can
    achieve true zero overlap (max_overlap=0) without being forced to
    share a single mandatory DST."""
    players = []
    # Team A: positions distributed so 2 disjoint valid skill groups exist.
    specs_a = (
        ("QB", 2), ("RB", 6), ("WR", 8), ("TE", 4),
    )
    rank = 0
    for position, count in specs_a:
        for i in range(count):
            rank += 1
            # Decreasing projection so the first 8 selected (by solver) are
            # the clear optimum, remaining 12 are still high but lower.
            projection = 30.0 - rank * 0.3
            players.append(Player(f"A_{position}{i}", f"A {position} {i}", position,
                                  "A", "B", 4000, projection))
    specs_e = (("QB", 1), ("RB", 3), ("WR", 4), ("TE", 2))
    for position, count in specs_e:
        for i in range(count):
            players.append(Player(f"E_{position}{i}", f"E {position} {i}", position,
                                  "E", "F", 4000, 8.0))
    players.append(Player("C_DST", "C DST", "DST", "C", "D", 2500, 5.0))
    players.append(Player("D_DST", "D DST", "DST", "D", "C", 2500, 4.9))
    return tuple(players)


def correlated_versus_independent_moments(players):
    """A guaranteed-PSD covariance matrix (diag + phi*loading@loading.T)
    where every team-A player shares one strong common factor and every
    other player is completely independent of it."""
    ids = tuple(p.id for p in players)
    n = len(ids)
    idiosyncratic = np.where(np.array([p.position for p in players]) == "DST", 0.1, 1.0)
    loading = np.array([1.0 if p.team == "A" else 0.0 for p in players])
    phi = 5.0
    sigma = np.diag(idiosyncratic) + phi * np.outer(loading, loading)
    covariance = {
        tuple(sorted((ids[i], ids[j]))): float(sigma[i, j])
        for i in range(n) for j in range(i + 1, n) if sigma[i, j] != 0
    }
    return Moments(
        scoring_version="v1", as_of="2024-01-01T00:00:00Z", player_ids=ids,
        variance=tuple(float(sigma[i, i]) for i in range(n)), covariance=covariance,
        missing_covariance_policy="assume_independent",
    )


def test_covariance_cap_blocks_disjoint_correlated_lineup_that_overlap_allows():
    players = correlated_versus_independent_pool()
    moments = correlated_versus_independent_moments(players)
    position_of = {p.id: p.position for p in players}
    team_of = {p.id: p.team for p in players}
    skill = lambda ids: [pid for pid in ids if position_of[pid] != "DST"]

    without_cap = generate_lineups(players, Config(
        num_lineups=2, max_overlap=0, avoid_dst_opponents=False,
    ))
    assert not without_cap[0].ids & without_cap[1].ids
    # Without a covariance constraint, the solver reuses team A for both
    # lineups (it has more total available projection than team E).
    assert all(team_of[pid] == "A" for pid in skill(without_cap[1].ids))

    with_cap = generate_lineups(players, Config(
        num_lineups=2, max_overlap=0, avoid_dst_opponents=False,
        moments=moments, covariance_cap=10.0,
    ))
    assert not with_cap[0].ids & with_cap[1].ids
    # With the covariance cap active, reusing team A becomes infeasible
    # (cross-covariance between two disjoint 8-player team-A groups is
    # 8*8*5.0 = 320, far above the cap of 10), forcing the second lineup to
    # team E instead, even though max_overlap=0 alone would have allowed
    # team A again.
    assert all(team_of[pid] == "E" for pid in skill(with_cap[1].ids))


def test_correlation_ceiling_also_blocks_the_same_disjoint_correlated_lineup():
    players = correlated_versus_independent_pool()
    moments = correlated_versus_independent_moments(players)
    position_of = {p.id: p.position for p in players}
    team_of = {p.id: p.team for p in players}
    result = generate_lineups(players, Config(
        num_lineups=2, max_overlap=0, avoid_dst_opponents=False,
        moments=moments, variance_floor=0.0, correlation_ceiling=0.05,
    ))
    assert all(team_of[pid] == "E" for pid in result[1].ids if position_of[pid] != "DST")


# ---------------------------------------------------------------------------
# Model-size structural awareness (variable/row count scaling)
# ---------------------------------------------------------------------------


def test_variance_floor_adds_exactly_one_continuous_variable_per_nonzero_pair(small_pool, moments):
    sigma = dense_sigma(IDS)
    nonzero_pairs = int(np.count_nonzero(np.triu(sigma, k=1)))
    data = build_model(small_pool, Config(moments=moments, variance_floor=1.0))
    baseline = build_model(small_pool, Config())
    assert len(data.costs) - len(baseline.costs) == nonzero_pairs
    assert int(np.sum(data.integrality == 0)) == nonzero_pairs
    assert np.all(data.integrality[:len(baseline.costs)] == 1)  # original columns still integer


def test_dense_moments_on_moderately_sized_pool_still_solves():
    # A dense 60-player covariance matrix (1,770 off-diagonal pairs) to
    # exercise the O(n^2) pair growth the plan warns about, without a
    # flaky hard timing assertion.
    rng = np.random.default_rng(0)
    positions = ["QB"] + ["RB"] * 3 + ["WR"] * 5 + ["TE"] * 2 + ["DST"]
    players = []
    for team in range(5):
        for k, position in enumerate(positions):
            salary = int(rng.integers(30, 81)) * 100
            projection = salary / 400 + float(rng.uniform(-3, 7))
            players.append(Player(f"{team}_{k}", f"Synthetic {team} {k}", position,
                                  f"T{team}", f"T{team ^ 1}", salary, projection))
    players = tuple(players)
    n = len(players)
    random_matrix = rng.normal(0, 0.1, size=(n, n))
    sigma = random_matrix @ random_matrix.T + np.eye(n)  # guaranteed PSD
    ids = tuple(p.id for p in players)
    covariance = {
        tuple(sorted((ids[i], ids[j]))): float(sigma[i, j])
        for i in range(n) for j in range(i + 1, n) if sigma[i, j] != 0
    }
    moments = Moments(
        scoring_version="v1", as_of="2024-01-01T00:00:00Z", player_ids=ids,
        variance=tuple(float(sigma[i, i]) for i in range(n)), covariance=covariance,
        missing_covariance_policy="assume_independent",
    )
    config = Config(moments=moments, variance_floor=float(np.median(np.diag(sigma))) * 9,
                    time_limit=120)
    lineup, = generate_lineups(players, config)
    assert lineup_variance(lineup, moments) >= config.variance_floor - 1e-6
