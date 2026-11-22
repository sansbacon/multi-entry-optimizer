"""Tests for multi_entry_optimizer.evaluation (Phase 2).

All fixtures here are synthetic/fabricated. Per PAPER_IMPLEMENTATION_PLAN.md,
these tests establish the evaluator's arithmetic correctness only; they
cannot and do not establish that any strategy performs well in real NFL
contests, since no real historical NFL contest/projection data is used.
"""

import json
import math

import numpy as np
import pytest

from multi_entry_optimizer.evaluation import (
    Contest, CorrelationEstimate, ExperimentRecord, PayoutTier, PlayerMeta,
    SlateEvaluation, SlateRecord, affordable_entries, align_player_values,
    assert_chronological_no_leakage, bucket_by_game_count, chronological_split,
    cross_sectional_dispersion, estimate_group_correlations, expand_score_histogram,
    fit_ols, fit_ridge, hash_payload, load_experiment_records, mae_rmse, predict_blend,
    rank_and_payout, record_experiment, replay_additive, replay_fixed_capacity,
    residualize, run_contest_replay, score_field_entries, score_lineup,
    select_policy_by_bucket, summarize_across_slates, summarize_replay_trials,
    validate_payouts, validate_scoring_version,
)


# A simple top-heavy payout table: 1st pays 100, 2nd-3rd pay 20 each, 4th-10th pay 5 each.
PAYOUTS = (
    PayoutTier(1, 1, 100.0), PayoutTier(2, 3, 20.0), PayoutTier(4, 10, 5.0),
)


def contest(**overrides):
    base = dict(
        contest_id="c1", slate_id="s1", site="draftkings", scoring_version="v1",
        lock_time="2024-09-08T17:00:00Z", games=(("A", "B"),), entry_fee=10.0,
        payouts=PAYOUTS, field_capacity=20, max_entries_per_user=150,
    )
    base.update(overrides)
    return Contest(**base)


# ---------------------------------------------------------------------------
# Payout validation and ranking/tie-splitting
# ---------------------------------------------------------------------------


def test_validate_payouts_rejects_bad_tiers():
    with pytest.raises(ValueError):
        validate_payouts(())
    with pytest.raises(ValueError, match="Overlapping"):
        validate_payouts((PayoutTier(1, 2, 10.0), PayoutTier(2, 3, 5.0)))
    with pytest.raises(ValueError):
        validate_payouts((PayoutTier(2, 1, 10.0),))
    with pytest.raises(ValueError):
        validate_payouts((PayoutTier(1, 1, -1.0),))


def test_rank_and_payout_no_ties():
    entries = [("a", 100.0), ("b", 90.0), ("c", 50.0)]
    ranked = rank_and_payout(entries, PAYOUTS)
    assert ranked["a"] == (1, 1, 100.0)
    assert ranked["b"] == (2, 1, 20.0)
    assert ranked["c"] == (3, 1, 20.0)


def test_rank_and_payout_ties_within_one_tier():
    # Two entries tied for 2nd/3rd (both pay 20) split evenly: 20 each, no change.
    entries = [("a", 100.0), ("b", 50.0), ("c", 50.0)]
    ranked = rank_and_payout(entries, PAYOUTS)
    assert ranked["b"] == (2, 2, 20.0)
    assert ranked["c"] == (2, 2, 20.0)


def test_rank_and_payout_ties_crossing_payout_bands():
    # Three-way tie for 1st occupies ranks 1,2,3: prizes 100+20+20=140, split
    # three ways. This is the "ties crossing payout bands" acceptance case.
    entries = [("a", 100.0), ("b", 100.0), ("c", 100.0), ("d", 10.0)]
    ranked = rank_and_payout(entries, PAYOUTS)
    assert ranked["a"] == (1, 3, pytest.approx(140.0 / 3))
    assert ranked["b"] == (1, 3, pytest.approx(140.0 / 3))
    assert ranked["c"] == (1, 3, pytest.approx(140.0 / 3))
    # d is now rank 4, in the 5.0 tier.
    assert ranked["d"] == (4, 1, 5.0)


def test_rank_and_payout_tie_beyond_paid_ranks_gets_zero():
    # Ties entirely outside the paid ranks split a prize of zero.
    entries = [(f"paid-{i}", 1000.0 - i) for i in range(10)] + [("x", 1.0), ("y", 1.0)]
    ranked = rank_and_payout(entries, PAYOUTS)
    assert ranked["x"] == (11, 2, 0.0)
    assert ranked["y"] == (11, 2, 0.0)


def test_rank_and_payout_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="Duplicate"):
        rank_and_payout([("a", 1.0), ("a", 2.0)], PAYOUTS)


# ---------------------------------------------------------------------------
# Strict-matching scoring
# ---------------------------------------------------------------------------


def test_score_lineup_strict_matching():
    assert score_lineup(["p1", "p2"], {"p1": 10.0, "p2": 5.5}) == pytest.approx(15.5)
    with pytest.raises(ValueError, match="Missing realized points"):
        score_lineup(["p1", "p3"], {"p1": 10.0})


def test_score_field_entries_and_histogram():
    assert score_field_entries([["p1", "p2"], ["p1"]], {"p1": 1.0, "p2": 2.0}) == (3.0, 1.0)
    assert sorted(expand_score_histogram({100.0: 2, 50.0: 1})) == [50.0, 100.0, 100.0]
    with pytest.raises(ValueError):
        expand_score_histogram({100.0: -1})


def test_validate_scoring_version_mismatch():
    with pytest.raises(ValueError, match="Scoring version mismatch"):
        validate_scoring_version(contest(), "v2")
    validate_scoring_version(contest(), "v1")  # no raise


# ---------------------------------------------------------------------------
# Replay: additive, fixed-capacity, "multiple own entries", "no profitable entries"
# ---------------------------------------------------------------------------


def test_replay_additive_basic_fees_and_profit():
    field = [80.0, 70.0, 60.0, 50.0, 40.0]
    generated = [100.0, 30.0]  # one entry wins outright, one scores poorly
    result = replay_additive(field, generated, entry_fee=10.0, payouts=PAYOUTS)
    assert result.entry_count == 2
    # 100 is rank 1 (prize 100); 30 is rank 7, within the 4-10 tier (prize 5).
    assert result.ranks == (1, 7)
    assert result.total_fees == pytest.approx(20.0)
    assert result.gross_winnings == pytest.approx(105.0)
    assert result.net_profit == pytest.approx(85.0)
    assert result.roi == pytest.approx(85.0 / 20.0)
    assert result.best_rank == 1
    assert result.best_entry_creation_index == 0


def test_replay_additive_multiple_own_entries_different_tiers():
    # Three of our own entries land in three different payout tiers at once.
    field = [1.0] * 10
    generated = [100.0, 25.0, 6.0]  # ranks 1, 2, 3 respectively
    result = replay_additive(field, generated, entry_fee=5.0, payouts=PAYOUTS)
    assert result.ranks == (1, 2, 3)
    assert result.prizes == (100.0, 20.0, 20.0)
    assert result.gross_winnings == pytest.approx(140.0)
    assert result.total_fees == pytest.approx(15.0)
    assert result.net_profit == pytest.approx(125.0)


def test_replay_additive_no_profitable_entries():
    field = [100.0] * 15
    generated = [1.0, 2.0, 3.0]  # finishes outside all paid ranks
    result = replay_additive(field, generated, entry_fee=10.0, payouts=PAYOUTS)
    assert result.gross_winnings == pytest.approx(0.0)
    assert result.net_profit == pytest.approx(-30.0)
    assert result.roi == pytest.approx(-1.0)


def test_replay_additive_rejects_empty_generated_and_negative_fee():
    with pytest.raises(ValueError):
        replay_additive([1.0], [], entry_fee=1.0, payouts=PAYOUTS)
    with pytest.raises(ValueError):
        replay_additive([1.0], [2.0], entry_fee=-1.0, payouts=PAYOUTS)


def test_replay_fixed_capacity_reproducible_with_seed():
    field = list(np.linspace(1, 100, 50))
    generated = [95.0, 50.0]
    first = replay_fixed_capacity(field, generated, 10.0, PAYOUTS, field_capacity=20, seed=7, trials=5)
    second = replay_fixed_capacity(field, generated, 10.0, PAYOUTS, field_capacity=20, seed=7, trials=5)
    assert first == second
    third = replay_fixed_capacity(field, generated, 10.0, PAYOUTS, field_capacity=20, seed=8, trials=5)
    assert first != third


def test_replay_fixed_capacity_enforces_contest_capacity():
    field = [50.0] * 5  # not enough field entries for the requested capacity
    with pytest.raises(ValueError, match="Not enough recorded field_scores"):
        replay_fixed_capacity(field, [10.0, 20.0], 1.0, PAYOUTS, field_capacity=20, seed=1, trials=1)
    with pytest.raises(ValueError, match="cannot exceed field_capacity"):
        replay_fixed_capacity([1.0] * 30, [1.0] * 25, 1.0, PAYOUTS, field_capacity=20, seed=1, trials=1)


def test_run_contest_replay_additive_and_fixed_capacity():
    realized = {"p1": 20.0, "p2": 15.0, "p3": 5.0, "p4": 1.0}
    field = [10.0] * 25
    result, = run_contest_replay(
        contest=contest(), generated_entries=[["p1", "p2"]], realized_points=realized,
        realized_scoring_version="v1", field_scores=field, mode="additive",
    )
    assert result.scores == (35.0,)
    with pytest.raises(ValueError, match="requires an explicit seed"):
        run_contest_replay(
            contest=contest(), generated_entries=[["p1"]], realized_points=realized,
            realized_scoring_version="v1", field_scores=field, mode="fixed_capacity",
        )
    results = run_contest_replay(
        contest=contest(), generated_entries=[["p1"]], realized_points=realized,
        realized_scoring_version="v1", field_scores=field, mode="fixed_capacity",
        trials=3, seed=1,
    )
    assert len(results) == 3


def test_run_contest_replay_missing_ids_and_scoring_version_and_entry_limit():
    realized = {"p1": 20.0}
    with pytest.raises(ValueError, match="Missing realized points"):
        run_contest_replay(
            contest=contest(), generated_entries=[["p1", "missing"]], realized_points=realized,
            realized_scoring_version="v1", field_scores=[1.0] * 25,
        )
    with pytest.raises(ValueError, match="Scoring version mismatch"):
        run_contest_replay(
            contest=contest(), generated_entries=[["p1"]], realized_points=realized,
            realized_scoring_version="v2", field_scores=[1.0] * 25,
        )
    with pytest.raises(ValueError, match="exceeds"):
        run_contest_replay(
            contest=contest(max_entries_per_user=1), generated_entries=[["p1"], ["p1"]],
            realized_points=realized, realized_scoring_version="v1", field_scores=[1.0] * 25,
        )


# ---------------------------------------------------------------------------
# Metrics aggregation: trials and cross-slate
# ---------------------------------------------------------------------------


def test_summarize_replay_trials_and_top_k_and_success_threshold():
    field = [50.0] * 30
    generated = [90.0, 20.0]
    results = replay_fixed_capacity(field, generated, 10.0, PAYOUTS, field_capacity=20, seed=3, trials=20)
    summary = summarize_replay_trials(results, top_k=(1, 5), success_thresholds=(80.0,))
    assert summary.trial_count == 20
    assert 0.0 <= summary.top_k_frequency[1] <= 1.0
    assert summary.any_success_frequency[80.0] == 1.0  # 90.0 always clears the threshold
    assert len(summary.roi_quantiles) == 5
    assert summary.mean_creation_order_spearman is not None


def test_summarize_replay_trials_rejects_empty():
    with pytest.raises(ValueError):
        summarize_replay_trials([])


def test_summarize_across_slates_pools_multiple_slates():
    field = [50.0] * 30
    results_a = replay_fixed_capacity(field, [90.0], 10.0, PAYOUTS, field_capacity=20, seed=1, trials=5)
    results_b = replay_fixed_capacity(field, [40.0], 10.0, PAYOUTS, field_capacity=20, seed=2, trials=5)
    pooled = summarize_across_slates([
        SlateEvaluation("s1", results_a), SlateEvaluation("s2", results_b),
    ])
    assert pooled.trial_count == 10
    with pytest.raises(ValueError):
        summarize_across_slates([])


def test_cross_sectional_dispersion():
    stats_ = cross_sectional_dispersion([10.0, 20.0, 30.0])
    assert stats_["mean"] == pytest.approx(20.0)
    assert stats_["min"] == 10.0 and stats_["max"] == 30.0
    assert cross_sectional_dispersion([5.0])["std"] == 0.0
    with pytest.raises(ValueError):
        cross_sectional_dispersion([])


def test_affordable_entries():
    assert affordable_entries(budget=105.0, entry_fee=10.0) == 10
    with pytest.raises(ValueError):
        affordable_entries(budget=-1.0, entry_fee=10.0)
    with pytest.raises(ValueError):
        affordable_entries(budget=100.0, entry_fee=0.0)


# ---------------------------------------------------------------------------
# Chronological split and leakage detection
# ---------------------------------------------------------------------------


def slates(n, start_day=1):
    return [SlateRecord(f"slate-{i}", f"2024-09-{start_day + i:02d}T17:00:00Z") for i in range(n)]


def test_chronological_split_basic():
    train, val, test = chronological_split(slates(10), train_fraction=0.6, validation_fraction=0.2)
    assert len(train) == 6 and len(val) == 2 and len(test) == 2
    assert_chronological_no_leakage(slates(10), train, val, test)


def test_chronological_split_no_validation():
    train, val, test = chronological_split(slates(5), train_fraction=0.6, validation_fraction=0.0)
    assert val == ()
    assert len(train) + len(test) == 5
    assert_chronological_no_leakage(slates(5), train, val, test)


def test_chronological_split_rejects_bad_fractions_and_too_few_slates():
    with pytest.raises(ValueError):
        chronological_split(slates(10), train_fraction=1.0, validation_fraction=0.0)
    with pytest.raises(ValueError):
        chronological_split(slates(10), train_fraction=0.9, validation_fraction=0.2)
    with pytest.raises(ValueError):
        chronological_split(slates(2), train_fraction=0.5, validation_fraction=0.0)
    with pytest.raises(ValueError, match="Duplicate"):
        chronological_split(slates(3) + [SlateRecord("slate-0", "2024-09-10T00:00:00Z")], 0.5, 0.0)


def test_assert_chronological_no_leakage_rejects_overlap_and_future_leakage():
    data = slates(6)
    with pytest.raises(ValueError, match="overlap"):
        assert_chronological_no_leakage(data, ("slate-0", "slate-1"), ("slate-1",), ("slate-2",))
    with pytest.raises(ValueError, match="Unknown slate"):
        assert_chronological_no_leakage(data, ("slate-0",), ("slate-1",), ("not-a-real-slate",))
    # Reversed order: "test" slate is earlier in time than the "train" slate.
    reversed_data = [SlateRecord("train-1", "2024-09-10T00:00:00Z"), SlateRecord("test-1", "2024-09-01T00:00:00Z")]
    with pytest.raises(ValueError, match="chronological leakage"):
        assert_chronological_no_leakage(reversed_data, ("train-1",), (), ("test-1",))


# ---------------------------------------------------------------------------
# Overlap/strategy policy selection
# ---------------------------------------------------------------------------


def test_bucket_by_game_count():
    assert bucket_by_game_count(2) == "small"
    assert bucket_by_game_count(5) == "medium"
    assert bucket_by_game_count(8) == "large"
    assert bucket_by_game_count(13) == "xlarge"
    with pytest.raises(ValueError):
        bucket_by_game_count(0)


def test_select_policy_by_bucket_picks_best_candidate():
    good = replay_additive([1.0] * 10, [100.0], 10.0, PAYOUTS)
    bad = replay_additive([1.0] * 10, [1.0], 10.0, PAYOUTS)
    policy = select_policy_by_bucket({
        "small": {4: [bad], 7: [good]},
        "large": {4: [good], 7: [bad]},
    })
    assert policy == {"small": 7, "large": 4}


def test_select_policy_by_bucket_rejects_empty_candidates():
    with pytest.raises(ValueError):
        select_policy_by_bucket({"small": {}})


# ---------------------------------------------------------------------------
# Projection blending (OLS / ridge)
# ---------------------------------------------------------------------------


def test_align_player_values_strict_intersection():
    sources = {"a": {"p1": 10.0, "p2": 20.0}, "b": {"p1": 12.0, "p3": 5.0}}
    realized = {"p1": 15.0, "p2": 18.0}
    ids, features, targets = align_player_values(sources, realized)
    assert ids == ("p1",)  # only p1 is in both sources and realized
    assert features.shape == (1, 2)
    assert targets.shape == (1,)
    with pytest.raises(ValueError):
        align_player_values({"a": {"p9": 1.0}}, {"p1": 1.0})


def test_fit_ols_recovers_linear_relationship():
    rng = np.random.default_rng(0)
    features = rng.uniform(0, 10, size=(50, 1))
    targets = 2.0 * features[:, 0] + 3.0 + rng.normal(0, 1e-9, size=50)
    coefficients = fit_ols(features, targets)
    assert coefficients[0] == pytest.approx(2.0, abs=1e-6)
    assert coefficients[1] == pytest.approx(3.0, abs=1e-6)
    predicted = predict_blend(features, coefficients)
    mae, rmse = mae_rmse(predicted, targets)
    assert mae < 1e-6 and rmse < 1e-6


def test_fit_ridge_shrinks_toward_zero_relative_to_ols():
    rng = np.random.default_rng(1)
    features = rng.uniform(0, 10, size=(60, 2))
    targets = 2.0 * features[:, 0] - 1.0 * features[:, 1] + 5.0 + rng.normal(0, 0.1, size=60)
    ols_coefficients = fit_ols(features, targets)
    ridge_coefficients = fit_ridge(features, targets, alpha=50.0)
    assert np.linalg.norm(ridge_coefficients[:-1]) < np.linalg.norm(ols_coefficients[:-1])
    with pytest.raises(ValueError):
        fit_ridge(features, targets, alpha=-1.0)


def test_fit_rejects_insufficient_samples():
    features = np.array([[1.0, 2.0]])
    targets = np.array([3.0])
    with pytest.raises(ValueError):
        fit_ols(features, targets)
    with pytest.raises(ValueError):
        fit_ridge(features, targets, alpha=1.0)


def test_mae_rmse_validation():
    with pytest.raises(ValueError):
        mae_rmse([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        mae_rmse([], [])


# ---------------------------------------------------------------------------
# Pairwise correlation analysis
# ---------------------------------------------------------------------------


def test_residualize_strict_matching():
    residuals = residualize({"p1": 10.0, "p2": 5.0}, {"p1": 12.0, "p2": 4.0})
    assert residuals == {"p1": 2.0, "p2": -1.0}
    with pytest.raises(ValueError, match="Missing projections"):
        residualize({"p1": 10.0}, {"p1": 12.0, "p2": 4.0})


def test_estimate_group_correlations_groups_by_game_position_and_relation():
    # Two different games, each contributing exactly one pair per group key,
    # so grouping (by game, sorted position pair, and same/opposing team) can
    # be checked without n=2 pooling complicating the correlation value.
    players = (
        PlayerMeta("qb_a", "A", "QB", ("A", "B")), PlayerMeta("wr_a", "A", "WR", ("A", "B")),
        PlayerMeta("dst_b", "B", "DST", ("A", "B")),
        PlayerMeta("qb_c", "C", "QB", ("C", "D")), PlayerMeta("rb_c", "C", "RB", ("C", "D")),
    )
    residuals = {"qb_a": 1.0, "wr_a": 2.0, "dst_b": -1.0, "qb_c": 0.5, "rb_c": -0.5}
    estimates = estimate_group_correlations(players, residuals, min_sample=1)
    by_group = {e.group: e for e in estimates}
    assert set(by_group) == {
        "QB-WR same_team", "DST-QB opposing_team", "DST-WR opposing_team", "QB-RB same_team",
    }
    for estimate in estimates:
        assert estimate.sample_count == 1
        # A single pair has no variance to correlate; the n<2 guard applies.
        assert math.isnan(estimate.raw_correlation)
        assert math.isnan(estimate.shrunk_correlation)
        assert estimate.confidence_interval is None


def test_estimate_group_correlations_shrinkage_and_confidence_interval():
    rng = np.random.default_rng(3)
    players = []
    residuals = {}
    for game_index in range(30):
        qb_id, wr_id = f"qb_{game_index}", f"wr_{game_index}"
        players.append(PlayerMeta(qb_id, "A", "QB", (f"A{game_index}", f"B{game_index}")))
        players.append(PlayerMeta(wr_id, "A", "WR", (f"A{game_index}", f"B{game_index}")))
        base = rng.normal(0, 1)
        residuals[qb_id] = base + rng.normal(0, 0.5)
        residuals[wr_id] = base + rng.normal(0, 0.5)
    estimates = estimate_group_correlations(tuple(players), residuals, min_sample=5,
                                            shrinkage_prior_strength=10.0)
    estimate, = [e for e in estimates if e.group == "QB-WR same_team"]
    assert estimate.sample_count == 30
    assert abs(estimate.shrunk_correlation) < abs(estimate.raw_correlation)
    assert estimate.confidence_interval is not None
    lower, upper = estimate.confidence_interval
    assert lower < estimate.raw_correlation < upper


def test_estimate_group_correlations_insufficient_sample_has_no_confidence_interval():
    players = (
        PlayerMeta("qb_a", "A", "QB", ("A", "B")), PlayerMeta("wr_a", "A", "WR", ("A", "B")),
    )
    residuals = {"qb_a": 1.0, "wr_a": 2.0}
    estimates = estimate_group_correlations(players, residuals, min_sample=5)
    assert estimates[0].confidence_interval is None
    assert math.isnan(estimates[0].raw_correlation)  # a single pair has no variance to correlate


def test_estimate_group_correlations_requires_all_residuals():
    players = (PlayerMeta("qb_a", "A", "QB", ("A", "B")),)
    with pytest.raises(ValueError, match="Missing residuals"):
        estimate_group_correlations(players, {})


# ---------------------------------------------------------------------------
# Experiment logging and reproducibility
# ---------------------------------------------------------------------------


def test_hash_payload_deterministic_and_order_independent():
    assert hash_payload({"a": 1, "b": 2}) == hash_payload({"b": 2, "a": 1})
    assert hash_payload({"a": 1}) != hash_payload({"a": 2})


def test_record_and_load_experiment_records(tmp_path):
    path = tmp_path / "experiments.jsonl"
    record = ExperimentRecord(
        experiment_id="exp-1", config_hash=hash_payload({"stack": 2}),
        data_hash=hash_payload({"slate": "s1"}), created_at="2024-09-08T00:00:00Z",
        training_cutoff="2024-09-01T00:00:00Z", random_seed=42,
        solver_versions={"scipy": "1.18.0"}, solve_statuses=("optimal",),
    )
    record_experiment(path, record)
    record_experiment(path, record)
    loaded = load_experiment_records(path)
    assert loaded == (record, record)
    assert load_experiment_records(tmp_path / "missing.jsonl") == ()
    # Confirm it is a genuine append-only JSON-lines file, not a full rewrite.
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["experiment_id"] == "exp-1"
