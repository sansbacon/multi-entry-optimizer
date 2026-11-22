"""Phase 4: probability-of-success diagnostics and optional greedy
scenario-coverage portfolio selection, built on top of Phase 1-3
(multi_entry_optimizer.lineups)'s lineups and Moments.

Part of the `multi_entry_optimizer` package; import as
`multi_entry_optimizer.probability`. Previously the top-level
`probability_diagnostics.py`; see CHANGELOG.md for the rename.

Scope and limits (read before using):
  * "Success" always means a declared, caller-supplied score threshold
    `t` is met or exceeded. This module never infers `t` from a
    held-out contest's actual winning score; that would leak the answer
    being evaluated. Source `t` from training data or state it directly.
  * Gaussian diagnostics use standard scipy tail/CDF functions on
    standardized (z-score) coordinates, not a transcription of the
    paper's extracted closed-form bound (Eq. 2.4): that extraction has
    apparent scale/normalization ambiguities that were not independently
    re-derivable with confidence (see PAPER_IMPLEMENTATION_PLAN.md).
  * `U2(S)`, the second-order inclusion-exclusion surrogate, is NOT a
    calibrated probability and is NOT guaranteed monotonic in S: it can
    be negative, and adding one more lineup can decrease it. It is
    reported raw, alongside a Monte Carlo estimate of the true union
    probability `U(S)`, so its approximation error is visible, never
    silently clipped or relabeled as exact.
  * The optional greedy scenario-coverage selection is exact maximum
    coverage on the FINITE candidate/scenario set actually used. Its
    classical (1 - 1/e) guarantee is relative to the best subset of that
    same candidate set under that same simulated scenario distribution,
    not the full feasible lineup universe or the true unknown NFL score
    distribution. A fixed threshold is a modeling proxy for "winning,"
    not literal tournament victory (which depends on the whole field's
    scores and a tie convention); this module does not model the field.
  * All simulation here is synthetic (drawn from a caller-supplied
    Moments covariance), never fit to or validated against real NFL
    outcomes in this workspace. Nothing here should be read as evidence
    that any portfolio wins real money.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from typing import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from .lineups import (
    Config, Lineup, Moments, Player, generate_lineups, lineup_covariance,
    lineup_variance, validate_moments,
)


# ---------------------------------------------------------------------------
# Gaussian success-probability diagnostics
# ---------------------------------------------------------------------------


def marginal_success_probability(mean: float, variance: float, threshold: float) -> float:
    """P(X >= threshold) for X ~ Normal(mean, variance), via scipy's
    standard normal survival function on the z-score. Zero variance is a
    point mass, handled explicitly rather than dividing by zero."""
    if not math.isfinite(mean) or not math.isfinite(threshold):
        raise ValueError("mean and threshold must be finite")
    if not math.isfinite(variance) or variance < 0:
        raise ValueError("variance must be finite and non-negative")
    if variance == 0:
        return 1.0 if mean >= threshold else 0.0
    z = (threshold - mean) / math.sqrt(variance)
    return float(stats.norm.sf(z))


def pairwise_joint_survival(
    mean_a: float, variance_a: float, mean_b: float, variance_b: float,
    covariance: float, threshold: float,
) -> float:
    """P(X_a >= threshold AND X_b >= threshold) for jointly Gaussian
    X_a, X_b, via the standard orthant-probability identity

        P(A>t, B>t) = sf(za) + sf(zb) - 1 + Phi2(za, zb; rho)

    computed on standardized z-score coordinates for numerical stability
    (rather than transcribing the paper's extracted Eq. 2.4 bound)."""
    if not all(math.isfinite(v) for v in (mean_a, mean_b, covariance, threshold)):
        raise ValueError("mean_a, mean_b, covariance, and threshold must be finite")
    if not math.isfinite(variance_a) or not math.isfinite(variance_b):
        raise ValueError("variance_a and variance_b must be finite")
    if variance_a <= 0 or variance_b <= 0:
        raise ValueError("variance_a and variance_b must be strictly positive")
    correlation = covariance / math.sqrt(variance_a * variance_b)
    # Guard tiny floating-point excursions outside the mathematically valid
    # [-1, 1] range (can arise from a covariance matrix validated only
    # within a PSD tolerance), not a silent cover-up of a real error.
    correlation = max(-1.0, min(1.0, correlation))
    za = (threshold - mean_a) / math.sqrt(variance_a)
    zb = (threshold - mean_b) / math.sqrt(variance_b)
    sf_a = float(stats.norm.sf(za))
    sf_b = float(stats.norm.sf(zb))
    if correlation >= 1.0 - 1e-12:
        # Perfectly (or numerically indistinguishably) correlated: exceeding
        # the larger z-score implies exceeding the smaller one (Frechet
        # upper bound), avoiding scipy's bivariate CDF becoming singular.
        joint_survival = min(sf_a, sf_b)
    elif correlation <= -1.0 + 1e-12:
        # Frechet lower bound.
        joint_survival = max(0.0, sf_a + sf_b - 1.0)
    else:
        joint_cdf = float(stats.multivariate_normal(
            mean=[0.0, 0.0], cov=[[1.0, correlation], [correlation, 1.0]],
        ).cdf([za, zb]))
        joint_survival = sf_a + sf_b - 1.0 + joint_cdf
    # A true probability cannot be negative or exceed either marginal;
    # clip only a tiny numerical tolerance, and only here (a genuine
    # probability), never for U2 below (which is allowed to be negative).
    if joint_survival < -1e-9 or joint_survival > min(sf_a, sf_b) + 1e-9:
        raise ValueError(
            f"Computed joint survival {joint_survival!r} is outside valid bounds "
            f"[0, min(sf_a, sf_b)={min(sf_a, sf_b)!r}]; this indicates a numerical "
            "integration problem, not a modeling choice to paper over."
        )
    return float(min(max(joint_survival, 0.0), min(sf_a, sf_b)))


@dataclass(frozen=True)
class SuccessDiagnostics:
    threshold: float
    marginal_probabilities: tuple[float, ...]
    pairwise_joint_probabilities: Mapping[tuple[int, int], float]
    u2: float
    u2_marginal_gains: tuple[float, ...]
    simulated_u: float
    simulated_u_standard_error: float
    simulation_count: int
    seed: int

    @property
    def approximation_error(self) -> float:
        """u2 - simulated_u. Can be positive or negative; u2 is not a
        bound in either direction once S has more than ~2 elements."""
        return self.u2 - self.simulated_u


def _union_player_table(lineups: Sequence[Lineup]) -> tuple[tuple[str, ...], tuple[Player, ...]]:
    union_player_map: dict[str, Player] = {}
    for lineup in lineups:
        for player in lineup.players:
            union_player_map.setdefault(player.id, player)
    ordered_ids = tuple(sorted(union_player_map))
    return ordered_ids, tuple(union_player_map[pid] for pid in ordered_ids)


def simulate_player_level_scenarios(
    lineups: Sequence[Lineup], moments: Moments, scenario_count: int, seed: int,
) -> tuple[NDArray[np.float64], dict[str, int]]:
    """Simulates scenario_count correlated PLAYER-level score draws ONCE,
    from the multivariate Gaussian implied by `moments` over the union of
    players appearing in any of `lineups`. Every lineup's score in every
    scenario is then a simple column sum over this SAME draw, which is
    what makes lineup-level success correctly correlated: simulating
    pairwise lineup correlations directly need not even combine to a
    valid joint distribution for 3+ lineups, but linear combinations of a
    single multivariate Gaussian draw always do.

    Returns (samples, player_index) where samples has shape
    (scenario_count, n_union_players) and player_index maps player ID to
    its column.
    """
    if scenario_count < 1:
        raise ValueError("scenario_count must be positive")
    ordered_ids, ordered_players = _union_player_table(lineups)
    sigma = validate_moments(ordered_players, moments)
    mean_vector = np.array([p.projection for p in ordered_players], dtype=np.float64)
    rng = np.random.default_rng(seed)
    # method="eigh" tolerates the "within tolerance" PSD guarantee from
    # validate_moments (exact Cholesky can reject a matrix that is PSD
    # only up to floating-point/solver tolerance).
    samples = rng.multivariate_normal(
        mean_vector, sigma, size=scenario_count, method="eigh", check_valid="ignore",
    )
    player_index = {pid: i for i, pid in enumerate(ordered_ids)}
    return samples, player_index


def score_lineups_against_scenarios(
    lineups: Sequence[Lineup], samples: NDArray[np.float64], player_index: Mapping[str, int],
) -> NDArray[np.float64]:
    """Sums the shared per-scenario player draws for each lineup, returning
    a (scenario_count, len(lineups)) score matrix."""
    scores = np.zeros((samples.shape[0], len(lineups)), dtype=np.float64)
    for k, lineup in enumerate(lineups):
        missing = [p.id for p in lineup.players if p.id not in player_index]
        if missing:
            raise ValueError(f"Lineup players missing from simulated scenarios: {missing}")
        columns = [player_index[p.id] for p in lineup.players]
        scores[:, k] = samples[:, columns].sum(axis=1)
    return scores


def simulate_union_probability(
    lineups: Sequence[Lineup], moments: Moments, threshold: float,
    scenario_count: int, seed: int,
) -> tuple[float, float]:
    """Monte Carlo estimate of P(at least one lineup's score >= threshold),
    with its binomial-proportion standard error. Returns (p_hat, se)."""
    samples, player_index = simulate_player_level_scenarios(lineups, moments, scenario_count, seed)
    scores = score_lineups_against_scenarios(lineups, samples, player_index)
    any_success = (scores >= threshold).any(axis=1)
    p_hat = float(any_success.mean())
    standard_error = math.sqrt(p_hat * (1.0 - p_hat) / scenario_count)
    return p_hat, standard_error


def compute_success_diagnostics(
    lineups: Sequence[Lineup], moments: Moments, threshold: float,
    simulation_count: int = 20_000, seed: int = 0,
) -> SuccessDiagnostics:
    """Computes marginal/pairwise Gaussian success probabilities, the raw
    (unclipped) U2 surrogate and its per-lineup marginal gains in the
    given order, and a Monte Carlo estimate of the true union probability
    for comparison. See the module docstring for what each quantity does
    and does not mean."""
    if not lineups:
        raise ValueError("lineups must be non-empty")
    means = [lineup.projection for lineup in lineups]
    variances = [lineup_variance(lineup, moments) for lineup in lineups]
    n = len(lineups)
    marginals = [marginal_success_probability(means[i], variances[i], threshold) for i in range(n)]
    pairwise: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            covariance = lineup_covariance(lineups[i], lineups[j], moments)
            pairwise[(i, j)] = pairwise_joint_survival(
                means[i], variances[i], means[j], variances[j], covariance, threshold,
            )
    u2 = sum(marginals) - sum(pairwise.values())
    gains = []
    running_marginal_sum = 0.0
    running_pair_sum = 0.0
    previous_u2 = 0.0
    for i in range(n):
        running_marginal_sum += marginals[i]
        for j in range(i):
            running_pair_sum += pairwise[(j, i)]
        current_u2 = running_marginal_sum - running_pair_sum
        gains.append(current_u2 - previous_u2)
        previous_u2 = current_u2
    simulated_u, standard_error = simulate_union_probability(
        lineups, moments, threshold, simulation_count, seed,
    )
    return SuccessDiagnostics(
        threshold=threshold, marginal_probabilities=tuple(marginals),
        pairwise_joint_probabilities=pairwise, u2=u2, u2_marginal_gains=tuple(gains),
        simulated_u=simulated_u, simulated_u_standard_error=standard_error,
        simulation_count=simulation_count, seed=seed,
    )


# ---------------------------------------------------------------------------
# Optional candidate-pool portfolio selection: exact greedy max coverage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageSelection:
    selected_indices: tuple[int, ...]
    covered_mass_after_each_step: tuple[float, ...]
    marginal_gain_per_step: tuple[float, ...]
    scenario_count: int


def greedy_scenario_coverage(
    success_matrix: NDArray[np.bool_], k: int,
    scenario_weights: NDArray[np.float64] | None = None,
) -> CoverageSelection:
    """Exact greedy maximum coverage: each step selects the remaining
    candidate (column) covering the most additional weighted scenario
    (row) mass not yet covered by any previously selected candidate.

    This is the classical NP-hard maximum-coverage problem; greedy has the
    standard (1 - 1/e) approximation guarantee relative to the true optimal
    k-subset of THESE SAME candidates and scenarios (Nemhauser et al.
    1978), not a guarantee about any other candidate set or the true
    distribution these scenarios were sampled from.
    """
    if success_matrix.ndim != 2:
        raise ValueError("success_matrix must be 2-dimensional (scenarios x candidates)")
    n_scenarios, n_candidates = success_matrix.shape
    if n_scenarios == 0 or n_candidates == 0:
        raise ValueError("success_matrix must have at least one scenario and one candidate")
    if scenario_weights is None:
        weights = np.full(n_scenarios, 1.0 / n_scenarios)
    else:
        weights = np.asarray(scenario_weights, dtype=np.float64)
        if weights.shape != (n_scenarios,):
            raise ValueError("scenario_weights must align one-to-one with success_matrix rows")
        if np.any(weights < 0) or not np.all(np.isfinite(weights)) or weights.sum() <= 0:
            raise ValueError("scenario_weights must be finite, non-negative, and sum to a positive value")
        weights = weights / weights.sum()
    if not 1 <= k <= n_candidates:
        raise ValueError(f"k must be between 1 and the number of candidates ({n_candidates})")

    covered = np.zeros(n_scenarios, dtype=bool)
    remaining = set(range(n_candidates))
    selected: list[int] = []
    covered_mass_steps: list[float] = []
    marginal_gains: list[float] = []
    for _ in range(k):
        best_candidate, best_gain = None, -1.0
        for candidate in remaining:
            newly_covered = success_matrix[:, candidate] & ~covered
            gain = float(weights[newly_covered].sum())
            if gain > best_gain:
                best_gain, best_candidate = gain, candidate
        assert best_candidate is not None
        selected.append(best_candidate)
        remaining.discard(best_candidate)
        covered = covered | success_matrix[:, best_candidate]
        covered_mass_steps.append(float(weights[covered].sum()))
        marginal_gains.append(best_gain)
    return CoverageSelection(
        selected_indices=tuple(selected), covered_mass_after_each_step=tuple(covered_mass_steps),
        marginal_gain_per_step=tuple(marginal_gains), scenario_count=n_scenarios,
    )


def brute_force_best_coverage(
    success_matrix: NDArray[np.bool_], k: int, scenario_weights: NDArray[np.float64] | None = None,
) -> float:
    """Exhaustively finds the true optimal covered mass for a size-k subset
    of candidates. Exponential in the number of candidates; intended only
    for small test instances that validate the greedy approximation."""
    n_scenarios, n_candidates = success_matrix.shape
    weights = (np.full(n_scenarios, 1.0 / n_scenarios) if scenario_weights is None
              else np.asarray(scenario_weights, dtype=np.float64) / np.sum(scenario_weights))
    best = 0.0
    for combo in itertools.combinations(range(n_candidates), k):
        covered = np.any(success_matrix[:, combo], axis=1)
        best = max(best, float(weights[covered].sum()))
    return best


def generate_candidate_pool(
    players: Sequence[Player], config: Config, pool_size: int,
) -> tuple[Lineup, ...]:
    """Generates a candidate pool larger than the eventual portfolio size
    k, reusing the existing sequential MILP (lineups.generate_lineups,
    Phases 1-3) rather than a separate optimizer. The candidate pool's
    overlap/stacking settings come entirely from `config`."""
    if pool_size < 1:
        raise ValueError("pool_size must be positive")
    return generate_lineups(players, replace(config, num_lineups=pool_size))


@dataclass(frozen=True)
class PortfolioSelection:
    lineups: tuple[Lineup, ...]
    selection: CoverageSelection
    scenario_count: int
    seed: int


def select_portfolio_by_coverage(
    candidates: Sequence[Lineup], moments: Moments, threshold: float, k: int,
    scenario_count: int = 20_000, seed: int = 0,
) -> PortfolioSelection:
    """Simulates correlated player-level scenarios ONCE, scores every
    candidate against the SAME scenarios, then greedily selects k of them
    by uncovered-success scenario mass."""
    samples, player_index = simulate_player_level_scenarios(candidates, moments, scenario_count, seed)
    scores = score_lineups_against_scenarios(candidates, samples, player_index)
    success_matrix = scores >= threshold
    selection = greedy_scenario_coverage(success_matrix, k)
    return PortfolioSelection(
        lineups=tuple(candidates[i] for i in selection.selected_indices),
        selection=selection, scenario_count=scenario_count, seed=seed,
    )


def evaluate_portfolios_success_rates(
    portfolios: Mapping[str, Sequence[Lineup]], moments: Moments, threshold: float,
    scenario_count: int, seed: int,
) -> dict[str, float]:
    """Evaluates every named portfolio's estimated union success
    probability against the SAME simulated scenarios (one shared
    simulation over the union of all players across all portfolios), for
    a fair, paired comparison. Use a seed distinct from any seed used to
    SELECT a portfolio (e.g. select_portfolio_by_coverage's seed), so the
    comparison is a genuine held-out evaluation, not graded on the same
    draws the selection was optimized against."""
    if not portfolios:
        raise ValueError("portfolios must be non-empty")
    all_lineups = [lineup for lineups in portfolios.values() for lineup in lineups]
    if not all_lineups:
        raise ValueError("every portfolio must contain at least one lineup")
    samples, player_index = simulate_player_level_scenarios(all_lineups, moments, scenario_count, seed)
    results = {}
    for name, lineups in portfolios.items():
        scores = score_lineups_against_scenarios(lineups, samples, player_index)
        results[name] = float((scores >= threshold).any(axis=1).mean())
    return results


@dataclass(frozen=True)
class BaselineComparison:
    sequential_lineups: tuple[Lineup, ...]
    coverage_lineups: tuple[Lineup, ...]
    sequential_success_rate: float
    coverage_success_rate: float
    evaluation_scenario_count: int
    evaluation_seed: int
    selection_scenario_count: int
    selection_seed: int


def compare_to_sequential_baseline(
    players: Sequence[Player], config: Config, moments: Moments, threshold: float,
    k: int, pool_size: int, selection_seed: int = 0, selection_scenario_count: int = 20_000,
    evaluation_seed: int = 1, evaluation_scenario_count: int = 20_000,
) -> BaselineComparison:
    """Compares the plain sequential top-k lineups (the existing
    Phase 1-3 default behavior) against a coverage-selected k-subset of a
    larger candidate pool, evaluated on held-out scenarios distinct from
    the ones used for coverage selection."""
    if evaluation_seed == selection_seed:
        raise ValueError(
            "evaluation_seed must differ from selection_seed for a genuine "
            "held-out comparison, not a re-evaluation on the selection draws"
        )
    sequential_lineups = generate_lineups(players, replace(config, num_lineups=k))
    candidate_pool = generate_candidate_pool(players, config, pool_size)
    selection = select_portfolio_by_coverage(
        candidate_pool, moments, threshold, k, selection_scenario_count, selection_seed,
    )
    rates = evaluate_portfolios_success_rates(
        {"sequential": sequential_lineups, "coverage": selection.lineups},
        moments, threshold, evaluation_scenario_count, evaluation_seed,
    )
    return BaselineComparison(
        sequential_lineups=sequential_lineups, coverage_lineups=selection.lineups,
        sequential_success_rate=rates["sequential"], coverage_success_rate=rates["coverage"],
        evaluation_scenario_count=evaluation_scenario_count, evaluation_seed=evaluation_seed,
        selection_scenario_count=selection_scenario_count, selection_seed=selection_seed,
    )
