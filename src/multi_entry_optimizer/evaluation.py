"""Historical NFL DFS contest replay, calibration, and correlation tooling.

Part of the `multi_entry_optimizer` package; import as
`multi_entry_optimizer.evaluation`. Previously the top-level
`historical_evaluation.py`; see CHANGELOG.md for the rename.

Implements Phase 2 of PAPER_IMPLEMENTATION_PLAN.md: an evaluator kept
separate from lineup generation (multi_entry_optimizer.lineups) and from
the solver-performance benchmark (benchmark.py). This module answers
"how would a set of generated
entries have performed in a real historical contest?", not "which lineup
should I generate?".

Scope and limits (read before using):
  * All functions here use strict player-ID and scoring-version matching.
    A missing realized score, projection, or ID never silently becomes zero;
    every such mismatch raises ValueError.
  * "Fixed capacity" replay reproduces the paper's methodology of randomly
    deleting historical field entries to make room for generated entries,
    repeated across seeded trials to report variation, not a single number.
  * Tuning helpers (chronological_split, select_policy_by_bucket) only
    perform splitting/selection; the caller is responsible for actually
    generating lineups and running replay for each candidate setting on
    each slate using the functions in this module.
  * No NFL-specific numeric defaults (overlap policies, blend weights,
    correlation priors) are hardcoded as "recommended" here, and nothing
    in multi_entry_optimizer.lineups's Config defaults is changed by this
    module. Promoting
    any tuned default requires running this tooling against a sufficiently
    broad, held-out sample of REAL historical NFL contest and projection
    data, which is not supplied in this workspace. The test suite for this
    module uses only synthetic fixtures, which can prove the evaluator's
    arithmetic is correct but cannot establish that any strategy performs
    well in real NFL contests.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import stats

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PayoutTier:
    """Pays `prize` to each occupied rank position in [min_rank, max_rank]."""

    min_rank: int
    max_rank: int
    prize: float


@dataclass(frozen=True)
class Contest:
    contest_id: str
    slate_id: str
    site: str
    scoring_version: str
    lock_time: str
    games: tuple[tuple[str, str], ...]
    entry_fee: float
    payouts: tuple[PayoutTier, ...]
    field_capacity: int
    max_entries_per_user: int


@dataclass(frozen=True)
class SlateRecord:
    """Minimal slate identity/time used for chronological splitting."""

    slate_id: str
    lock_time: str


@dataclass(frozen=True)
class PlayerMeta:
    id: str
    team: str
    position: str
    game: tuple[str, str]


def validate_payouts(payouts: Sequence[PayoutTier]) -> None:
    if not payouts:
        raise ValueError("payouts must be non-empty")
    ordered = sorted(payouts, key=lambda tier: tier.min_rank)
    previous_max = 0
    for tier in ordered:
        if tier.min_rank < 1 or tier.max_rank < tier.min_rank:
            raise ValueError(f"Invalid payout tier: {tier}")
        if tier.min_rank <= previous_max:
            raise ValueError(f"Overlapping payout tiers at rank {tier.min_rank}")
        if tier.prize < 0:
            raise ValueError("Prize must be non-negative")
        previous_max = tier.max_rank


def validate_scoring_version(contest: Contest, realized_scoring_version: str) -> None:
    if contest.scoring_version != realized_scoring_version:
        raise ValueError(
            f"Scoring version mismatch: contest {contest.contest_id!r} requires "
            f"{contest.scoring_version!r}, but realized points use {realized_scoring_version!r}"
        )


# ---------------------------------------------------------------------------
# Strict-matching scoring
# ---------------------------------------------------------------------------


def score_lineup(player_ids: Sequence[str], realized_points: Mapping[str, float]) -> float:
    """Sums realized points for a lineup; raises on any missing player ID."""
    missing = sorted(set(player_ids) - set(realized_points))
    if missing:
        raise ValueError(f"Missing realized points for player IDs: {missing}")
    return float(sum(realized_points[pid] for pid in player_ids))


def score_field_entries(
    entries: Sequence[Sequence[str]], realized_points: Mapping[str, float]
) -> tuple[float, ...]:
    """Scores historical opponent lineups when full rosters are available."""
    return tuple(score_lineup(ids, realized_points) for ids in entries)


def expand_score_histogram(histogram: Mapping[float, int]) -> tuple[float, ...]:
    """Expands a {score: count} histogram into a flat score list.

    Use this when only aggregate field score counts are available, not full
    opponent rosters.
    """
    if any(count < 0 for count in histogram.values()):
        raise ValueError("Histogram counts must be non-negative")
    scores: list[float] = []
    for score, count in histogram.items():
        scores.extend([float(score)] * int(count))
    return tuple(scores)


# ---------------------------------------------------------------------------
# Ranking and payouts, with tie handling
# ---------------------------------------------------------------------------


def _prize_for_rank(payouts: Sequence[PayoutTier], rank: int) -> float:
    for tier in payouts:
        if tier.min_rank <= rank <= tier.max_rank:
            return tier.prize
    return 0.0


def rank_and_payout(
    entries: Sequence[tuple[str, float]], payouts: Sequence[PayoutTier]
) -> dict[str, tuple[int, int, float]]:
    """Ranks entries by score descending and splits tied payouts.

    Returns {entry_id: (best_rank, tie_group_size, prize_per_entry)}.
    Ties share the best (lowest-numbered) occupied rank and evenly split the
    SUM of prizes for every rank position the tied group occupies, including
    when that group spans more than one payout tier.
    """
    validate_payouts(payouts)
    ids = [entry_id for entry_id, _ in entries]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate entry IDs in entries")
    ordered = sorted(entries, key=lambda item: item[1], reverse=True)
    results: dict[str, tuple[int, int, float]] = {}
    index = 0
    rank = 1
    total = len(ordered)
    while index < total:
        group_end = index
        while group_end < total and ordered[group_end][1] == ordered[index][1]:
            group_end += 1
        group = ordered[index:group_end]
        group_size = group_end - index
        total_prize = sum(_prize_for_rank(payouts, rank + offset) for offset in range(group_size))
        per_entry_prize = total_prize / group_size
        for entry_id, _ in group:
            results[entry_id] = (rank, group_size, per_entry_prize)
        rank += group_size
        index = group_end
    return results


# ---------------------------------------------------------------------------
# Replay: additive counterfactual and fixed-capacity (seeded trials)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayResult:
    entry_count: int
    scores: tuple[float, ...]
    ranks: tuple[int, ...]
    prizes: tuple[float, ...]
    total_fees: float
    gross_winnings: float
    net_profit: float
    roi: float
    best_rank: int
    median_rank: float
    creation_order_spearman: float | None
    best_entry_creation_index: int


def _summarize_generated(
    ranked: Mapping[str, tuple[int, int, float]],
    generated_scores: Sequence[float],
    entry_fee: float,
) -> ReplayResult:
    ranks = tuple(ranked[f"generated-{i}"][0] for i in range(len(generated_scores)))
    prizes = tuple(ranked[f"generated-{i}"][2] for i in range(len(generated_scores)))
    gross_winnings = float(sum(prizes))
    total_fees = entry_fee * len(generated_scores)
    net_profit = gross_winnings - total_fees
    roi = net_profit / total_fees if total_fees > 0 else float("nan")
    best_rank = min(ranks)
    median_rank = float(np.median(np.asarray(ranks, dtype=float)))
    spearman = None
    if len(ranks) >= 2:
        correlation, _ = stats.spearmanr(range(len(ranks)), ranks)
        spearman = float(correlation) if math.isfinite(correlation) else None
    best_entry_creation_index = min(range(len(ranks)), key=lambda i: ranks[i])
    return ReplayResult(
        entry_count=len(generated_scores), scores=tuple(generated_scores), ranks=ranks,
        prizes=prizes, total_fees=total_fees, gross_winnings=gross_winnings,
        net_profit=net_profit, roi=roi, best_rank=best_rank, median_rank=median_rank,
        creation_order_spearman=spearman, best_entry_creation_index=best_entry_creation_index,
    )


def replay_additive(
    field_scores: Sequence[float], generated_scores: Sequence[float],
    entry_fee: float, payouts: Sequence[PayoutTier],
) -> ReplayResult:
    """Adds generated entries alongside the full historical field as-is.

    This ignores any field-capacity limit; use replay_fixed_capacity to
    reproduce a sold-out, capacity-constrained contest instead.
    """
    if not generated_scores:
        raise ValueError("generated_scores must be non-empty")
    if entry_fee < 0:
        raise ValueError("entry_fee must be non-negative")
    entries = [(f"field-{i}", score) for i, score in enumerate(field_scores)]
    entries += [(f"generated-{i}", score) for i, score in enumerate(generated_scores)]
    ranked = rank_and_payout(entries, payouts)
    return _summarize_generated(ranked, generated_scores, entry_fee)


def replay_fixed_capacity(
    field_scores: Sequence[float], generated_scores: Sequence[float],
    entry_fee: float, payouts: Sequence[PayoutTier], field_capacity: int,
    seed: int, trials: int,
) -> tuple[ReplayResult, ...]:
    """Replays a capacity-constrained contest across seeded random trials.

    Each trial randomly keeps `field_capacity - len(generated_scores)` of the
    recorded field_scores (reproducing the paper's seeded random deletion of
    field entries to make room for the generated entries), then runs
    replay_additive on the kept field plus the generated entries. The same
    seed always reproduces the same sequence of trials.
    """
    if trials < 1:
        raise ValueError("trials must be at least 1")
    if len(generated_scores) > field_capacity:
        raise ValueError("generated_scores cannot exceed field_capacity")
    kept_count = field_capacity - len(generated_scores)
    if len(field_scores) < kept_count:
        raise ValueError(
            f"Not enough recorded field_scores ({len(field_scores)}) to keep "
            f"{kept_count} after making room for {len(generated_scores)} generated entries"
        )
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(trials):
        kept_indices = rng.choice(len(field_scores), size=kept_count, replace=False)
        kept_scores = [field_scores[i] for i in kept_indices]
        results.append(replay_additive(kept_scores, generated_scores, entry_fee, payouts))
    return tuple(results)


# ---------------------------------------------------------------------------
# Top-level convenience wrapper
# ---------------------------------------------------------------------------


def run_contest_replay(
    *, contest: Contest, generated_entries: Sequence[Sequence[str]],
    realized_points: Mapping[str, float], realized_scoring_version: str,
    field_scores: Sequence[float], mode: str = "additive",
    trials: int = 1, seed: int | None = None,
) -> tuple[ReplayResult, ...]:
    """Ties Contest metadata to the lower-level replay functions above."""
    validate_scoring_version(contest, realized_scoring_version)
    validate_payouts(contest.payouts)
    if contest.entry_fee < 0:
        raise ValueError("entry_fee must be non-negative")
    if len(generated_entries) > contest.max_entries_per_user:
        raise ValueError(
            f"generated_entries ({len(generated_entries)}) exceeds "
            f"max_entries_per_user ({contest.max_entries_per_user})"
        )
    generated_scores = tuple(score_lineup(ids, realized_points) for ids in generated_entries)
    if mode == "additive":
        return (replay_additive(field_scores, generated_scores, contest.entry_fee, contest.payouts),)
    if mode == "fixed_capacity":
        if seed is None:
            raise ValueError("fixed_capacity replay requires an explicit seed for reproducibility")
        return replay_fixed_capacity(
            field_scores, generated_scores, contest.entry_fee, contest.payouts,
            contest.field_capacity, seed, trials,
        )
    raise ValueError(f"Unknown mode: {mode}. Use 'additive' or 'fixed_capacity'")


# ---------------------------------------------------------------------------
# Metrics aggregation, across trials and across slates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialSummary:
    trial_count: int
    mean_roi: float
    median_roi: float
    roi_quantiles: tuple[float, ...]
    mean_net_profit: float
    median_net_profit: float
    best_rank_min: int
    best_rank_median: float
    top_k_frequency: Mapping[int, float]
    any_success_frequency: Mapping[float, float]
    mean_creation_order_spearman: float | None


_QUANTILE_LEVELS = (0.10, 0.25, 0.50, 0.75, 0.90)


def summarize_replay_trials(
    results: Sequence[ReplayResult], top_k: Sequence[int] = (1, 10, 100),
    success_thresholds: Sequence[float] = (),
) -> TrialSummary:
    """Aggregates ReplayResults, whether they come from repeated trials of
    one slate or pooled across many slates (see summarize_across_slates)."""
    if not results:
        raise ValueError("results must be non-empty")
    rois = np.array([r.roi for r in results], dtype=float)
    profits = np.array([r.net_profit for r in results], dtype=float)
    best_ranks = np.array([r.best_rank for r in results], dtype=float)
    finite_rois = rois[np.isfinite(rois)]
    roi_quantiles = (
        tuple(float(q) for q in np.quantile(finite_rois, _QUANTILE_LEVELS))
        if finite_rois.size else tuple(float("nan") for _ in _QUANTILE_LEVELS)
    )
    top_k_frequency = {
        k: float(np.mean([any(rank <= k for rank in r.ranks) for r in results])) for k in top_k
    }
    any_success_frequency = {
        threshold: float(np.mean([any(score >= threshold for score in r.scores) for r in results]))
        for threshold in success_thresholds
    }
    spearmans = [r.creation_order_spearman for r in results if r.creation_order_spearman is not None]
    mean_spearman = float(np.mean(spearmans)) if spearmans else None
    return TrialSummary(
        trial_count=len(results),
        mean_roi=float(np.mean(finite_rois)) if finite_rois.size else float("nan"),
        median_roi=float(np.median(finite_rois)) if finite_rois.size else float("nan"),
        roi_quantiles=roi_quantiles, mean_net_profit=float(np.mean(profits)),
        median_net_profit=float(np.median(profits)), best_rank_min=int(best_ranks.min()),
        best_rank_median=float(np.median(best_ranks)), top_k_frequency=top_k_frequency,
        any_success_frequency=any_success_frequency, mean_creation_order_spearman=mean_spearman,
    )


@dataclass(frozen=True)
class SlateEvaluation:
    slate_id: str
    results: tuple[ReplayResult, ...]


def summarize_across_slates(
    evaluations: Sequence[SlateEvaluation], top_k: Sequence[int] = (1, 10, 100),
    success_thresholds: Sequence[float] = (),
) -> TrialSummary:
    """Pools results across multiple slates to report cross-slate uncertainty,
    distinct from the within-slate trial uncertainty in summarize_replay_trials
    applied to a single slate's trials."""
    if not evaluations:
        raise ValueError("evaluations must be non-empty")
    pooled = tuple(result for evaluation in evaluations for result in evaluation.results)
    return summarize_replay_trials(pooled, top_k=top_k, success_thresholds=success_thresholds)


def cross_sectional_dispersion(field_scores: Sequence[float]) -> dict[str, float]:
    """Dispersion of REALIZED field scores in one contest (a population
    statistic). This is NOT a model's predicted lineup variance; Phase 2 does
    not compute predictive variance (see Phase 3's moment-aware MILP)."""
    values = np.asarray(field_scores, dtype=float)
    if values.size == 0:
        raise ValueError("field_scores must be non-empty")
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()), "max": float(values.max()),
    }


def affordable_entries(budget: float, entry_fee: float) -> int:
    """Entries affordable under a fixed total budget, for comparing
    strategies at equal total cost rather than equal entry count."""
    if budget < 0 or entry_fee <= 0:
        raise ValueError("budget must be non-negative and entry_fee must be positive")
    return int(budget // entry_fee)


# ---------------------------------------------------------------------------
# Chronological train/validation/test splitting
# ---------------------------------------------------------------------------


def chronological_split(
    slates: Sequence[SlateRecord], train_fraction: float, validation_fraction: float,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Splits slates by lock_time into (train, validation, test) slate IDs,
    never dividing a slate's data across folds."""
    if len({s.slate_id for s in slates}) != len(slates):
        raise ValueError("Duplicate slate_id in slates")
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError(
            "Require 0 < train_fraction < 1, 0 <= validation_fraction, and "
            "train_fraction + validation_fraction < 1 (to leave a non-empty test split)"
        )
    ordered = sorted(slates, key=lambda s: s.lock_time)
    n = len(ordered)
    if n < 3:
        raise ValueError("Need at least 3 slates to form non-empty train/validation/test splits")
    n_train = max(1, min(round(n * train_fraction), n - 2))
    remaining = n - n_train
    n_val = min(round(n * validation_fraction), remaining - 1) if validation_fraction > 0 else 0
    n_val = max(0, n_val)
    n_test = n - n_train - n_val
    if n_test < 1:
        raise ValueError("Not enough slates to form a non-empty test split")
    train_ids = tuple(s.slate_id for s in ordered[:n_train])
    val_ids = tuple(s.slate_id for s in ordered[n_train:n_train + n_val])
    test_ids = tuple(s.slate_id for s in ordered[n_train + n_val:])
    return train_ids, val_ids, test_ids


def assert_chronological_no_leakage(
    slates: Sequence[SlateRecord], train_ids: Sequence[str],
    val_ids: Sequence[str], test_ids: Sequence[str],
) -> None:
    """Rejects overlapping slate IDs and any split boundary where a later
    fold's earliest slate is not strictly after the earlier fold's latest."""
    by_id = {s.slate_id: s for s in slates}
    all_ids = [*train_ids, *val_ids, *test_ids]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Slate IDs overlap across train/validation/test splits")
    unknown = set(all_ids) - set(by_id)
    if unknown:
        raise ValueError(f"Unknown slate IDs in splits: {sorted(unknown)}")

    def latest(ids: Sequence[str]) -> str | None:
        return max((by_id[i].lock_time for i in ids), default=None)

    def earliest(ids: Sequence[str]) -> str | None:
        return min((by_id[i].lock_time for i in ids), default=None)

    if val_ids:
        train_end, val_start = latest(train_ids), earliest(val_ids)
        if train_end is not None and val_start is not None and train_end > val_start:
            raise ValueError("Training slates are not strictly before validation slates (chronological leakage)")
    if test_ids:
        boundary_ids = val_ids if val_ids else train_ids
        boundary_end, test_start = latest(boundary_ids), earliest(test_ids)
        if boundary_end is not None and test_start is not None and boundary_end > test_start:
            raise ValueError("Training/validation slates are not strictly before test slates (chronological leakage)")


# ---------------------------------------------------------------------------
# Overlap/strategy policy selection (selection only; caller runs replay)
# ---------------------------------------------------------------------------


def bucket_by_game_count(game_count: int) -> str:
    """A simple, inspectable slate-size bucketing used only as an input to
    select_policy_by_bucket; NHL overlap recommendations (4-7) are NOT
    reused here as NFL defaults."""
    if game_count < 1:
        raise ValueError("game_count must be positive")
    if game_count <= 3:
        return "small"
    if game_count <= 6:
        return "medium"
    if game_count <= 9:
        return "large"
    return "xlarge"


def select_policy_by_bucket(
    bucket_candidate_results: Mapping[str, Mapping[int, Sequence[ReplayResult]]],
    metric: Callable[[Sequence[ReplayResult]], float] | None = None,
) -> dict[str, int]:
    """Given, for each slate-size bucket, each candidate setting's training
    ReplayResults, selects the candidate value maximizing `metric` per
    bucket. This performs selection only: the caller must have already
    generated lineups and run replay for every candidate on every training/
    validation slate using the functions above."""
    if metric is None:
        metric = lambda results: float(np.mean([r.net_profit for r in results]))
    policy: dict[str, int] = {}
    for bucket, candidates in bucket_candidate_results.items():
        if not candidates:
            raise ValueError(f"No candidate values supplied for bucket {bucket!r}")
        policy[bucket] = max(candidates, key=lambda value: metric(candidates[value]))
    return policy


# ---------------------------------------------------------------------------
# Projection blending (OLS / ridge via NumPy only)
# ---------------------------------------------------------------------------


def align_player_values(
    sources: Mapping[str, Mapping[str, float]], realized: Mapping[str, float],
) -> tuple[tuple[str, ...], NDArray[np.float64], NDArray[np.float64]]:
    """Returns (player_ids, feature_matrix, target_vector) using only player
    IDs present in every source and in realized (a strict intersection)."""
    if not sources:
        raise ValueError("sources must be non-empty")
    source_names = tuple(sources)
    common_ids = set(realized)
    for name in source_names:
        common_ids &= set(sources[name])
    if not common_ids:
        raise ValueError("No player IDs are present in every source and in realized")
    ordered_ids = tuple(sorted(common_ids))
    features = np.array(
        [[sources[name][pid] for name in source_names] for pid in ordered_ids], dtype=float
    )
    targets = np.array([realized[pid] for pid in ordered_ids], dtype=float)
    return ordered_ids, features, targets


def _design_matrix(features: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.column_stack([features, np.ones(len(features))])


def fit_ols(features: NDArray[np.float64], targets: NDArray[np.float64]) -> NDArray[np.float64]:
    """Ordinary least squares with an intercept; last coefficient is the
    intercept. Requires more samples than parameters."""
    design = _design_matrix(features)
    if design.shape[0] <= design.shape[1]:
        raise ValueError("Need more samples than parameters to fit OLS")
    coefficients, *_ = np.linalg.lstsq(design, targets, rcond=None)
    return coefficients


def fit_ridge(
    features: NDArray[np.float64], targets: NDArray[np.float64], alpha: float,
) -> NDArray[np.float64]:
    """Ridge regression with an intercept (not penalized); last coefficient
    is the intercept."""
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    design = _design_matrix(features)
    if design.shape[0] <= design.shape[1]:
        raise ValueError("Need more samples than parameters to fit ridge regression")
    penalty = alpha * np.eye(design.shape[1])
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return coefficients


def predict_blend(features: NDArray[np.float64], coefficients: NDArray[np.float64]) -> NDArray[np.float64]:
    return _design_matrix(features) @ coefficients


def mae_rmse(predicted: Sequence[float], actual: Sequence[float]) -> tuple[float, float]:
    """Projection accuracy only; evaluate downstream contest outcomes with
    the replay functions above as a SEPARATE evaluation, not a substitute."""
    predicted_arr = np.asarray(predicted, dtype=float)
    actual_arr = np.asarray(actual, dtype=float)
    if predicted_arr.shape != actual_arr.shape:
        raise ValueError("predicted and actual must have the same length")
    if predicted_arr.size == 0:
        raise ValueError("predicted/actual must be non-empty")
    errors = predicted_arr - actual_arr
    return float(np.mean(np.abs(errors))), float(np.sqrt(np.mean(errors ** 2)))


# ---------------------------------------------------------------------------
# Pairwise correlation analysis, with shrinkage and confidence intervals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrelationEstimate:
    group: str
    raw_correlation: float
    shrunk_correlation: float
    sample_count: int
    confidence_interval: tuple[float, float] | None


def residualize(projections: Mapping[str, float], realized: Mapping[str, float]) -> dict[str, float]:
    """Residual = realized - projection, used to reduce confounding from
    shared pregame/game-level expectations before estimating correlations."""
    missing = sorted(set(realized) - set(projections))
    if missing:
        raise ValueError(f"Missing projections for player IDs: {missing}")
    return {pid: realized[pid] - projections[pid] for pid in realized}


def estimate_group_correlations(
    players: Sequence[PlayerMeta], residuals: Mapping[str, float],
    min_sample: int = 5, shrinkage_prior_strength: float = 10.0,
) -> tuple[CorrelationEstimate, ...]:
    """Pearson correlation of residuals for same-game player pairs, grouped
    by sorted position pair and same/opposing team, with a sample-size
    shrinkage estimate and a Fisher z confidence interval (only reported
    when sample_count >= min_sample)."""
    if shrinkage_prior_strength < 0:
        raise ValueError("shrinkage_prior_strength must be non-negative")
    missing = sorted({p.id for p in players} - set(residuals))
    if missing:
        raise ValueError(f"Missing residuals for player IDs: {missing}")
    by_game: dict[tuple[str, str], list[PlayerMeta]] = {}
    for player in players:
        by_game.setdefault(player.game, []).append(player)
    pairs: dict[str, list[tuple[float, float]]] = {}
    for group in by_game.values():
        for a, b in itertools.combinations(group, 2):
            relation = "same_team" if a.team == b.team else "opposing_team"
            key = f"{'-'.join(sorted((a.position, b.position)))} {relation}"
            pairs.setdefault(key, []).append((residuals[a.id], residuals[b.id]))
    estimates = []
    for key, values in sorted(pairs.items()):
        n = len(values)
        xs = np.array([v[0] for v in values], dtype=float)
        ys = np.array([v[1] for v in values], dtype=float)
        if n < 2 or np.std(xs) == 0 or np.std(ys) == 0:
            raw = float("nan")
        else:
            raw = float(np.corrcoef(xs, ys)[0, 1])
        shrunk = raw * n / (n + shrinkage_prior_strength) if math.isfinite(raw) else float("nan")
        confidence_interval = None
        if n >= min_sample and math.isfinite(raw) and abs(raw) < 1:
            z = math.atanh(raw)
            standard_error = 1 / math.sqrt(n - 3) if n > 3 else float("inf")
            if math.isfinite(standard_error):
                confidence_interval = (math.tanh(z - 1.96 * standard_error), math.tanh(z + 1.96 * standard_error))
        estimates.append(CorrelationEstimate(
            group=key, raw_correlation=raw, shrunk_correlation=shrunk,
            sample_count=n, confidence_interval=confidence_interval,
        ))
    return tuple(estimates)


# ---------------------------------------------------------------------------
# Reproducible experiment logging
# ---------------------------------------------------------------------------


def hash_payload(payload: object) -> str:
    """Deterministic hash of a JSON-serializable payload (sorted keys)."""
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    config_hash: str
    data_hash: str
    created_at: str
    training_cutoff: str | None
    random_seed: int | None
    solver_versions: Mapping[str, str]
    solve_statuses: tuple[str, ...]


def record_experiment(path: Path, record: ExperimentRecord) -> None:
    """Appends one JSON line per experiment; never overwrites prior records."""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def load_experiment_records(path: Path) -> tuple[ExperimentRecord, ...]:
    if not path.exists():
        return ()
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            payload["solver_versions"] = dict(payload["solver_versions"])
            payload["solve_statuses"] = tuple(payload["solve_statuses"])
            records.append(ExperimentRecord(**payload))
    return tuple(records)
