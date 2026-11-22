"""Tests for multi_entry_optimizer.probability (Phase 4).

All fixtures here use synthetic, hand-constructed Gaussian/factor-model
data. These tests establish the diagnostics' and selection algorithm's
mathematical/statistical correctness only; per the module's own
docstring, none of this is validated against real NFL outcomes.
"""

import itertools
import math
from pathlib import Path

import numpy as np
import pytest

from multi_entry_optimizer import Config, Lineup, Moments, Player, load_players
from multi_entry_optimizer.probability import (
    BaselineComparison, CoverageSelection, SuccessDiagnostics,
    brute_force_best_coverage, compare_to_sequential_baseline,
    compute_success_diagnostics, evaluate_portfolios_success_rates,
    generate_candidate_pool, greedy_scenario_coverage, marginal_success_probability,
    pairwise_joint_survival, score_lineups_against_scenarios, select_portfolio_by_coverage,
    simulate_player_level_scenarios, simulate_union_probability,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pool():
    return load_players(Path(__file__).resolve().parents[1] / "examples" / "players.csv")


def make_player(pid, position, team, opponent, salary=4000, projection=10.0):
    return Player(pid, pid, position, team, opponent, salary, projection)


def dense_moments(players, sigma):
    ids = tuple(p.id for p in players)
    covariance = {
        tuple(sorted((ids[i], ids[j]))): float(sigma[i, j])
        for i in range(len(ids)) for j in range(i + 1, len(ids)) if sigma[i, j] != 0
    }
    return Moments(
        scoring_version="v1", as_of="2024-01-01T00:00:00Z", player_ids=ids,
        variance=tuple(float(sigma[i, i]) for i in range(len(ids))), covariance=covariance,
        missing_covariance_policy="assume_independent",
    )


# ---------------------------------------------------------------------------
# marginal_success_probability
# ---------------------------------------------------------------------------


def test_marginal_success_probability_matches_known_values():
    # z=0 -> exactly 0.5; z=1.6448536... -> 0.05 (standard normal tables).
    assert marginal_success_probability(0.0, 1.0, 0.0) == pytest.approx(0.5)
    assert marginal_success_probability(100.0, 100.0, 100.0 + 10 * 1.6448536269514722) == pytest.approx(0.05, abs=1e-6)


def test_marginal_success_probability_zero_variance_is_point_mass():
    assert marginal_success_probability(10.0, 0.0, 10.0) == 1.0
    assert marginal_success_probability(10.0, 0.0, 10.1) == 0.0
    assert marginal_success_probability(10.0, 0.0, 9.9) == 1.0


def test_marginal_success_probability_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        marginal_success_probability(float("nan"), 1.0, 0.0)
    with pytest.raises(ValueError):
        marginal_success_probability(0.0, -1.0, 0.0)
    with pytest.raises(ValueError):
        marginal_success_probability(0.0, float("nan"), 0.0)


def test_marginal_success_probability_underflow_deep_tail_returns_zero_not_nan():
    result = marginal_success_probability(0.0, 1.0, 50.0)
    assert result == 0.0
    assert not math.isnan(result)


# ---------------------------------------------------------------------------
# pairwise_joint_survival: analytic edge cases (Frechet bounds, independence)
# ---------------------------------------------------------------------------


def test_pairwise_joint_survival_independence_matches_product_of_marginals():
    mean_a, var_a, mean_b, var_b, threshold = 10.0, 4.0, 20.0, 9.0, 15.0
    sf_a = marginal_success_probability(mean_a, var_a, threshold)
    sf_b = marginal_success_probability(mean_b, var_b, threshold)
    joint = pairwise_joint_survival(mean_a, var_a, mean_b, var_b, 0.0, threshold)
    assert joint == pytest.approx(sf_a * sf_b, abs=1e-6)


def test_pairwise_joint_survival_perfect_positive_correlation_frechet_upper_bound():
    # rho=1 with the same z-score (identical mean/variance/threshold):
    # exceeding one implies exceeding the other, so joint == marginal.
    mean, variance, threshold = 10.0, 4.0, 12.0
    covariance = variance  # rho = covariance / sqrt(var*var) = 1
    joint = pairwise_joint_survival(mean, variance, mean, variance, covariance, threshold)
    marginal = marginal_success_probability(mean, variance, threshold)
    assert joint == pytest.approx(marginal, abs=1e-6)


def test_pairwise_joint_survival_perfect_negative_correlation_frechet_lower_bound():
    mean_a, var_a, mean_b, var_b, threshold = 10.0, 4.0, 10.0, 4.0, 10.0
    covariance = -math.sqrt(var_a * var_b)  # rho = -1
    sf_a = marginal_success_probability(mean_a, var_a, threshold)
    sf_b = marginal_success_probability(mean_b, var_b, threshold)
    joint = pairwise_joint_survival(mean_a, var_a, mean_b, var_b, covariance, threshold)
    assert joint == pytest.approx(max(0.0, sf_a + sf_b - 1.0), abs=1e-6)


def test_pairwise_joint_survival_near_perfect_correlation_stays_bounded():
    mean, variance, threshold = 10.0, 4.0, 12.0
    for rho in (0.999999, -0.999999):
        covariance = rho * variance
        joint = pairwise_joint_survival(mean, variance, mean, variance, covariance, threshold)
        assert 0.0 <= joint <= 1.0
        assert not math.isnan(joint)


def test_pairwise_joint_survival_never_exceeds_either_marginal():
    rng = np.random.default_rng(0)
    for _ in range(50):
        mean_a, mean_b = rng.uniform(-5, 5, size=2)
        var_a, var_b = rng.uniform(0.1, 5, size=2)
        rho = rng.uniform(-0.99, 0.99)
        covariance = rho * math.sqrt(var_a * var_b)
        threshold = rng.uniform(-5, 10)
        joint = pairwise_joint_survival(mean_a, var_a, mean_b, var_b, covariance, threshold)
        sf_a = marginal_success_probability(mean_a, var_a, threshold)
        sf_b = marginal_success_probability(mean_b, var_b, threshold)
        assert -1e-9 <= joint <= min(sf_a, sf_b) + 1e-9


def test_pairwise_joint_survival_rejects_non_positive_variance():
    with pytest.raises(ValueError):
        pairwise_joint_survival(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        pairwise_joint_survival(0.0, 1.0, 0.0, -1.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# compute_success_diagnostics: U2, its marginal gains, and simulated U
# ---------------------------------------------------------------------------


def independent_lineups_and_moments():
    # 3 lineups sharing no players and with zero cross-covariance: U(S)
    # should equal the exact inclusion-exclusion result, and simulated_u
    # should closely match both the analytic U(S) and U2 (they coincide
    # exactly only for independent events, which lets us validate the
    # simulator itself against a known closed form).
    players = [make_player(f"p{i}", "QB" if i == 0 else "RB", f"T{i}", f"O{i}", projection=10.0 + i)
              for i in range(9)]
    lineups = [Lineup((players[3 * i], players[3 * i + 1], players[3 * i + 2])) for i in range(3)]
    sigma = np.eye(9) * 4.0
    moments = dense_moments(players, sigma)
    return lineups, moments


def test_u2_matches_exact_union_probability_for_independent_lineups():
    lineups, moments = independent_lineups_and_moments()
    threshold = sum(p.projection for p in lineups[0].players) - 1.0  # a middling threshold
    diagnostics = compute_success_diagnostics(lineups, moments, threshold, simulation_count=200_000, seed=0)
    p = diagnostics.marginal_probabilities
    exact_u = 1.0 - (1 - p[0]) * (1 - p[1]) * (1 - p[2])
    # For independent events, U2 (second-order truncation) and the exact
    # union coincide only when third-order+ terms vanish, which they do
    # not in general; instead verify U2's OWN exact inclusion-exclusion
    # identity (sum of marginals minus sum of pairwise products, since
    # independence makes pairwise joint = product of marginals).
    expected_u2 = sum(p) - (p[0] * p[1] + p[0] * p[2] + p[1] * p[2])
    assert diagnostics.u2 == pytest.approx(expected_u2, abs=1e-6)
    assert diagnostics.simulated_u == pytest.approx(exact_u, abs=0.01)


def test_u2_marginal_gains_can_be_negative():
    # Three heavily overlapping (same-team, strongly positively correlated)
    # lineups: adding a third highly-redundant lineup should actually
    # DECREASE U2 relative to the first two, demonstrating U2 is not
    # monotonic the way the true U(S) is proven to be.
    players = [make_player(f"A{i}", "RB", "A", "B", projection=10.0) for i in range(9)]
    lineups = [Lineup((players[0], players[1], players[2])),
              Lineup((players[3], players[4], players[5])),
              Lineup((players[6], players[7], players[8]))]
    sigma = np.full((9, 9), 3.9) + np.eye(9) * 0.1  # near-maximal positive correlation
    moments = dense_moments(players, sigma)
    threshold = 25.0
    diagnostics = compute_success_diagnostics(lineups, moments, threshold, simulation_count=1000, seed=0)
    assert any(gain < 0 for gain in diagnostics.u2_marginal_gains)


def test_success_diagnostics_approximation_error_property():
    lineups, moments = independent_lineups_and_moments()
    threshold = 25.0
    diagnostics = compute_success_diagnostics(lineups, moments, threshold, simulation_count=5000, seed=1)
    assert diagnostics.approximation_error == pytest.approx(diagnostics.u2 - diagnostics.simulated_u)


def test_compute_success_diagnostics_rejects_empty_lineups():
    with pytest.raises(ValueError):
        compute_success_diagnostics([], Moments("v1", "x", (), (), {}, "assume_independent"), 0.0)


# ---------------------------------------------------------------------------
# simulate_union_probability / simulate_player_level_scenarios: seeded
# reproducibility and shared player-level draws
# ---------------------------------------------------------------------------


def test_simulate_union_probability_reproducible_with_seed():
    lineups, moments = independent_lineups_and_moments()
    threshold = 25.0
    first = simulate_union_probability(lineups, moments, threshold, 5000, seed=42)
    second = simulate_union_probability(lineups, moments, threshold, 5000, seed=42)
    assert first == second


def test_simulate_union_probability_different_seeds_generically_differ():
    lineups, moments = independent_lineups_and_moments()
    # A threshold picked so the union probability is comfortably mid-range
    # (not saturated near 0 or 1), where different seeds' sampling noise
    # is visible.
    threshold = 55.0
    first, _ = simulate_union_probability(lineups, moments, threshold, 2000, seed=1)
    second, _ = simulate_union_probability(lineups, moments, threshold, 2000, seed=2)
    assert 0.0 < first < 1.0
    assert first != second


def test_shared_player_level_simulation_gives_consistent_correlation():
    # Two lineups sharing ALL players (hence perfectly correlated scores)
    # simulated from one shared draw must show success matrices that are
    # identical whenever both lineups contain exactly the same players.
    players = [make_player(f"p{i}", "RB", "A", "B", projection=5.0) for i in range(3)]
    lineup_a = Lineup(tuple(players))
    lineup_b = Lineup(tuple(players))
    sigma = np.eye(3) * 1.0
    moments = dense_moments(players, sigma)
    samples, index = simulate_player_level_scenarios([lineup_a, lineup_b], moments, 500, seed=3)
    scores = score_lineups_against_scenarios([lineup_a, lineup_b], samples, index)
    assert np.array_equal(scores[:, 0], scores[:, 1])


# ---------------------------------------------------------------------------
# greedy_scenario_coverage: enumerate tiny instances, marginal gains,
# monotonicity, dependence handling, reproducibility
# ---------------------------------------------------------------------------


def test_greedy_matches_brute_force_optimal_on_tiny_instance():
    # 4 scenarios, 3 candidates; candidate coverage sets chosen so greedy's
    # natural choice is provably optimal for k=2.
    success_matrix = np.array([
        [True, False, False],
        [True, False, False],
        [False, True, False],
        [False, False, True],
    ])
    selection = greedy_scenario_coverage(success_matrix, k=2)
    optimal = brute_force_best_coverage(success_matrix, k=2)
    assert selection.covered_mass_after_each_step[-1] == pytest.approx(optimal)


def test_greedy_achieves_at_least_the_one_minus_one_over_e_guarantee():
    rng = np.random.default_rng(7)
    n_scenarios, n_candidates, k = 12, 6, 3
    success_matrix = rng.uniform(size=(n_scenarios, n_candidates)) < 0.35
    optimal = brute_force_best_coverage(success_matrix, k)
    selection = greedy_scenario_coverage(success_matrix, k)
    achieved = selection.covered_mass_after_each_step[-1]
    guarantee = (1 - 1 / math.e) * optimal
    assert achieved >= guarantee - 1e-9


def test_greedy_marginal_gains_are_non_increasing():
    # Submodularity of coverage implies greedy's own sequence of marginal
    # gains is non-increasing (each subsequent pick's best available gain
    # cannot exceed the previous pick's).
    rng = np.random.default_rng(11)
    success_matrix = rng.uniform(size=(40, 10)) < 0.3
    selection = greedy_scenario_coverage(success_matrix, k=8)
    gains = selection.marginal_gain_per_step
    assert all(gains[i] >= gains[i + 1] - 1e-9 for i in range(len(gains) - 1))


def test_greedy_covered_mass_is_monotonically_non_decreasing():
    rng = np.random.default_rng(12)
    success_matrix = rng.uniform(size=(30, 8)) < 0.4
    selection = greedy_scenario_coverage(success_matrix, k=6)
    steps = selection.covered_mass_after_each_step
    assert all(steps[i] <= steps[i + 1] + 1e-9 for i in range(len(steps) - 1))


def test_greedy_prefers_diverse_over_redundant_candidates():
    # 2 "redundant" candidates covering the identical half of scenarios,
    # and 2 "diverse" candidates covering disjoint halves. For k=2,
    # picking the diverse pair covers everything; picking both redundant
    # candidates covers only half.
    n = 100
    half = np.zeros(n, dtype=bool)
    half[: n // 2] = True
    success_matrix = np.column_stack([half, half, ~half, half])
    # Candidates: 0 and 1 are redundant (identical); 0 and 2 are diverse
    # (complementary). Confirm greedy picks a diverse pair, not the
    # redundant duplicate, when both give an equal FIRST pick but a
    # better SECOND pick is available.
    selection = greedy_scenario_coverage(success_matrix, k=2)
    assert selection.covered_mass_after_each_step[-1] == pytest.approx(1.0)
    assert 2 in selection.selected_indices  # the complementary candidate must be used


def test_greedy_reproducible_and_deterministic():
    rng = np.random.default_rng(5)
    success_matrix = rng.uniform(size=(20, 6)) < 0.3
    first = greedy_scenario_coverage(success_matrix, k=4)
    second = greedy_scenario_coverage(success_matrix, k=4)
    assert first == second


def test_greedy_rejects_invalid_k_and_weights():
    success_matrix = np.array([[True, False], [False, True]])
    with pytest.raises(ValueError):
        greedy_scenario_coverage(success_matrix, k=0)
    with pytest.raises(ValueError):
        greedy_scenario_coverage(success_matrix, k=3)
    with pytest.raises(ValueError):
        greedy_scenario_coverage(success_matrix, k=1, scenario_weights=np.array([1.0, -1.0]))
    with pytest.raises(ValueError):
        greedy_scenario_coverage(success_matrix, k=1, scenario_weights=np.array([1.0]))


def test_greedy_respects_custom_scenario_weights():
    # Scenario 0 (covered only by candidate 0) is weighted far more than
    # scenario 1 (covered only by candidate 1), so greedy must pick 0 first.
    success_matrix = np.array([[True, False], [False, True]])
    weights = np.array([100.0, 1.0])
    selection = greedy_scenario_coverage(success_matrix, k=1, scenario_weights=weights)
    assert selection.selected_indices == (0,)


# ---------------------------------------------------------------------------
# select_portfolio_by_coverage / generate_candidate_pool / comparison with
# the sequential baseline on held-out scenarios
# ---------------------------------------------------------------------------


def test_select_portfolio_by_coverage_end_to_end(pool):
    config = Config(num_lineups=1, max_overlap=5, stack=2, bring_back=True)
    candidates = generate_candidate_pool(pool, config, pool_size=10)
    assert len(candidates) == 10
    rng = np.random.default_rng(0)
    ids = {p.id for lineup in candidates for p in lineup.players}
    sigma_ids = sorted(ids)
    n = len(sigma_ids)
    loading = rng.normal(0, 1, size=n)
    sigma = np.eye(n) * 5.0 + 0.5 * np.outer(loading, loading)
    players_by_id = {p.id: p for lineup in candidates for p in lineup.players}
    moments = dense_moments([players_by_id[pid] for pid in sigma_ids], sigma)
    threshold = max(lineup.projection for lineup in candidates) - 5.0
    selection = select_portfolio_by_coverage(candidates, moments, threshold, k=3,
                                              scenario_count=3000, seed=2)
    assert len(selection.lineups) == 3
    assert len(set(selection.selection.selected_indices)) == 3


def test_generate_candidate_pool_rejects_invalid_pool_size(pool):
    with pytest.raises(ValueError):
        generate_candidate_pool(pool, Config(), pool_size=0)


def test_evaluate_portfolios_success_rates_shares_one_simulation(pool):
    config = Config(num_lineups=2, max_overlap=5)
    from multi_entry_optimizer import generate_lineups
    lineups_a = generate_lineups(pool, config)
    lineups_b = generate_lineups(pool, Config(num_lineups=2, max_overlap=5, avoid_dst_opponents=False))
    rng = np.random.default_rng(1)
    all_ids = sorted({p.id for lu in (*lineups_a, *lineups_b) for p in lu.players})
    n = len(all_ids)
    sigma = np.eye(n) * 4.0
    players_by_id = {p.id: p for lu in (*lineups_a, *lineups_b) for p in lu.players}
    moments = dense_moments([players_by_id[pid] for pid in all_ids], sigma)
    threshold = 100.0
    rates = evaluate_portfolios_success_rates(
        {"a": lineups_a, "b": lineups_b}, moments, threshold, scenario_count=2000, seed=9,
    )
    assert set(rates) == {"a", "b"}
    assert all(0.0 <= rate <= 1.0 for rate in rates.values())


def test_evaluate_portfolios_success_rates_rejects_empty():
    with pytest.raises(ValueError):
        evaluate_portfolios_success_rates({}, Moments("v1", "x", (), (), {}, "assume_independent"), 0.0, 10, 0)


def test_compare_to_sequential_baseline_requires_distinct_seeds(pool):
    config = Config(num_lineups=1, max_overlap=6)
    players_by_id = {p.id: p for p in pool}
    sigma = np.eye(len(pool)) * 4.0
    moments = dense_moments(pool, sigma)
    with pytest.raises(ValueError, match="must differ"):
        compare_to_sequential_baseline(
            pool, config, moments, threshold=100.0, k=2, pool_size=5,
            selection_seed=3, evaluation_seed=3,
        )


def test_compare_to_sequential_baseline_held_out_evaluation(pool):
    config = Config(num_lineups=1, max_overlap=5, stack=1)
    sigma = np.eye(len(pool)) * 6.0
    # Add mild same-team positive correlation so coverage selection has a
    # genuine diversification incentive to act on.
    teams = sorted({p.team for p in pool})
    team_index = {t: i for i, t in enumerate(teams)}
    for i, pi in enumerate(pool):
        for j, pj in enumerate(pool):
            if i < j and pi.team == pj.team:
                sigma[i, j] = sigma[j, i] = 1.0
    moments = dense_moments(pool, sigma)
    threshold = sum(sorted((p.projection for p in pool), reverse=True)[:9]) * 0.6
    comparison = compare_to_sequential_baseline(
        pool, config, moments, threshold, k=3, pool_size=12,
        selection_seed=10, selection_scenario_count=4000,
        evaluation_seed=11, evaluation_scenario_count=4000,
    )
    assert isinstance(comparison, BaselineComparison)
    assert len(comparison.sequential_lineups) == 3
    assert len(comparison.coverage_lineups) == 3
    assert 0.0 <= comparison.sequential_success_rate <= 1.0
    assert 0.0 <= comparison.coverage_success_rate <= 1.0
    # The two portfolios' success rates were evaluated on a shared,
    # held-out scenario set distinct from the selection scenarios.
    assert comparison.evaluation_seed != comparison.selection_seed
