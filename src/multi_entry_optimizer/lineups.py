"""Sequential, projection-maximizing NFL classic DFS lineups via SciPy's
MILP solver (HiGHS backend).

Part of the `multi_entry_optimizer` package (top-level API: see
`multi_entry_optimizer/__init__.py`). This module was previously the
top-level `nfl_dfs.py`; see CHANGELOG.md for the rename.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix


POSITIONS = ("QB", "RB", "WR", "TE", "DST")
SLOTS = ("QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST")


@dataclass(frozen=True)
class Player:
    id: str
    name: str
    position: str
    team: str
    opponent: str
    salary: int
    projection: float

    @property
    def game(self) -> tuple[str, str]:
        first, second = sorted((self.team, self.opponent))
        return (first, second)


@dataclass(frozen=True)
class Rules:
    salary_cap: int
    min_teams: int
    min_games: int
    max_per_team: int


RULES = {
    "draftkings": Rules(50_000, 2, 2, 9),
    "fanduel": Rules(60_000, 3, 1, 4),
}


# Phase 3: optional moment-aware (variance/covariance) data contract. This is
# entirely opt-in; nothing here is used unless Config.moments is set.
SUPPORTED_MISSING_COVARIANCE_POLICIES = frozenset({"assume_independent"})


@dataclass(frozen=True)
class Moments:
    """Player-ID-aligned variance/covariance dataset, kept separate from the
    core Player/CSV contract. Off-diagonal pairs not present in `covariance`
    are filled per `missing_covariance_policy` (currently only
    "assume_independent", meaning covariance 0); the field exists so that
    choice is always explicit and recorded, never an implicit default."""

    scoring_version: str
    as_of: str
    player_ids: tuple[str, ...]
    variance: tuple[float, ...]
    covariance: Mapping[tuple[str, str], float]
    missing_covariance_policy: str
    psd_tolerance: float = 1e-6


def validate_moments(players: Sequence[Player], moments: Moments) -> NDArray[np.float64]:
    """Validates a Moments dataset against a specific player pool and
    returns the dense covariance submatrix aligned to `players`' order.

    Requires complete diagonal coverage for every supplied player, finite
    values, nonnegative variances, unambiguous (sorted, non-self) covariance
    keys, and a positive-semidefinite result within `psd_tolerance`.
    """
    if moments.missing_covariance_policy not in SUPPORTED_MISSING_COVARIANCE_POLICIES:
        raise ValueError(
            f"Unsupported missing_covariance_policy: {moments.missing_covariance_policy!r}. "
            f"Supported: {sorted(SUPPORTED_MISSING_COVARIANCE_POLICIES)}"
        )
    if len(moments.variance) != len(moments.player_ids):
        raise ValueError("Moments.variance must align one-to-one with Moments.player_ids")
    if len(set(moments.player_ids)) != len(moments.player_ids):
        raise ValueError("Moments.player_ids contains duplicate player IDs")
    if not math.isfinite(moments.psd_tolerance) or moments.psd_tolerance < 0:
        raise ValueError("Moments.psd_tolerance must be finite and non-negative")
    moment_index = {pid: i for i, pid in enumerate(moments.player_ids)}
    missing = sorted({p.id for p in players} - set(moment_index))
    if missing:
        raise ValueError(f"Moments dataset is missing a variance entry for player IDs: {missing}")
    for pid, variance in zip(moments.player_ids, moments.variance):
        if not math.isfinite(variance):
            raise ValueError(f"Non-finite variance for player ID {pid!r}")
        if variance < 0:
            raise ValueError(f"Negative variance for player ID {pid!r}")
    for (a, b), value in moments.covariance.items():
        if a == b:
            raise ValueError(f"Covariance entry ({a!r}, {b!r}) is a self-pair; use variance for diagonal terms")
        if a > b:
            raise ValueError(
                f"Covariance keys must be a sorted (lesser, greater) ID pair to avoid "
                f"ambiguous duplicate entries; got ({a!r}, {b!r})"
            )
        if a not in moment_index or b not in moment_index:
            raise ValueError(f"Covariance entry references an unknown player ID: ({a!r}, {b!r})")
        if not math.isfinite(value):
            raise ValueError(f"Non-finite covariance for pair ({a!r}, {b!r})")

    n = len(players)
    local_index = {p.id: i for i, p in enumerate(players)}
    sigma = np.zeros((n, n), dtype=np.float64)
    for p in players:
        sigma[local_index[p.id], local_index[p.id]] = moments.variance[moment_index[p.id]]
    for (a, b), value in moments.covariance.items():
        if a in local_index and b in local_index:
            i, j = local_index[a], local_index[b]
            sigma[i, j] = value
            sigma[j, i] = value
    eigenvalues = np.linalg.eigvalsh(sigma)
    if eigenvalues.min() < -moments.psd_tolerance:
        raise ValueError(
            f"Covariance submatrix for the supplied player pool is not positive "
            f"semidefinite within tolerance {moments.psd_tolerance}: "
            f"minimum eigenvalue {eigenvalues.min():.6g}"
        )
    return sigma


@dataclass(frozen=True)
class Config:
    site: str = "draftkings"
    num_lineups: int = 1
    max_overlap: int = 7
    stack: int = 0
    bring_back: bool = False
    avoid_dst_opponents: bool = True
    min_salary: int = 0
    locks: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()
    time_limit: float = 60.0
    # Strategic team concentration, independent of each site's own minimum
    # team requirement (Rules.min_teams) and maximum-players-per-team rule
    # (Rules.max_per_team). None disables each bound.
    strategic_min_teams: int | None = None
    strategic_max_teams: int | None = None
    strategic_team_scope: str = "all"
    # Secondary, QB-independent skill-position stack drawn from a team in a
    # different game than the selected QB's game. 0 disables it.
    secondary_stack: int = 0
    secondary_stack_positions: tuple[str, ...] = ("RB", "WR", "TE")
    # Restricts which positions may occupy the single FLEX roster slot.
    flex_positions: tuple[str, ...] = ("RB", "WR", "TE")
    # Phase 3: optional moment-aware constraints. All default to disabled;
    # setting variance_floor/covariance_cap/correlation_ceiling requires
    # moments to be set (see validate_config).
    moments: Moments | None = None
    variance_floor: float | None = None
    covariance_cap: float | None = None
    correlation_ceiling: float | None = None


# Named Phase 1 strategy presets. Each maps to the stacking-related Config
# fields it overrides; names are NFL-specific and do not mirror the Julia
# hockey "Type 1-5" formulations, since NFL has no line/power-play structure.
STRATEGIES: dict[str, dict[str, object]] = {
    "baseline": {"stack": 0, "bring_back": False, "secondary_stack": 0,
                 "avoid_dst_opponents": False},
    "dst-avoidance": {"stack": 0, "bring_back": False, "secondary_stack": 0,
                      "avoid_dst_opponents": True},
    "qb-plus-one": {"stack": 1, "bring_back": False, "secondary_stack": 0,
                    "avoid_dst_opponents": True},
    "qb-plus-two": {"stack": 2, "bring_back": False, "secondary_stack": 0,
                    "avoid_dst_opponents": True},
    "qb-stack-bring-back": {"stack": 2, "bring_back": True, "secondary_stack": 0,
                            "avoid_dst_opponents": True},
    "secondary-stack": {"stack": 2, "bring_back": True, "secondary_stack": 2,
                        "avoid_dst_opponents": True},
}


def apply_strategy(config: Config, name: str) -> Config:
    """Return config with a named Phase 1 strategy's stacking fields applied."""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Choices: {sorted(STRATEGIES)}")
    return replace(config, **STRATEGIES[name])


@dataclass(frozen=True)
class Lineup:
    players: tuple[Player, ...]

    @property
    def salary(self) -> int:
        return sum(p.salary for p in self.players)

    @property
    def projection(self) -> float:
        return sum(p.projection for p in self.players)

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(p.id for p in self.players)

    def slot_players(self) -> tuple[Player, ...]:
        groups = {pos: [p for p in self.players if p.position == pos] for pos in POSITIONS}
        base: list[Player] = []
        for pos, count in (("QB", 1), ("RB", 2), ("WR", 3), ("TE", 1)):
            base.extend(groups[pos][:count])
            del groups[pos][:count]
        flex = [p for pos in ("RB", "WR", "TE") for p in groups[pos]]
        if len(base) != 7 or len(flex) != 1 or len(groups["DST"]) != 1:
            raise ValueError("Lineup cannot be assigned to classic roster slots")
        return tuple(base + flex + groups["DST"])


class OptimizationError(RuntimeError):
    """No complete, proven-optimal portfolio could be generated."""


def validate_players(players: Sequence[Player]) -> None:
    if not players:
        raise ValueError("Player pool is empty")
    seen: set[str] = set()
    opponents: dict[str, str] = {}
    for p in players:
        if not all(isinstance(v, str) and v.strip() == v and v for v in
                   (p.id, p.name, p.team, p.opponent)):
            raise ValueError("Player IDs, names, teams, and opponents must be nonempty trimmed strings")
        if p.id in seen:
            raise ValueError(f"Duplicate player ID: {p.id}")
        seen.add(p.id)
        if p.position not in POSITIONS:
            raise ValueError(f"Unsupported position for {p.id}: {p.position}")
        if isinstance(p.salary, bool) or not isinstance(p.salary, int) or p.salary <= 0:
            raise ValueError(f"Salary must be a positive integer for {p.id}")
        if not math.isfinite(p.projection):
            raise ValueError(f"Projection must be finite for {p.id}")
        if p.team == p.opponent:
            raise ValueError(f"Team cannot play itself: {p.id}")
        for team, opponent in ((p.team, p.opponent), (p.opponent, p.team)):
            if team in opponents and opponents[team] != opponent:
                raise ValueError(f"Inconsistent opponents for {team}; use a single slate")
            opponents[team] = opponent


def load_players(path: Path) -> tuple[Player, ...]:
    required = {"id", "name", "position", "team", "opponent", "salary", "projection"}
    players: list[Player] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"CSV requires columns: {', '.join(sorted(required))}")
        for line, row in enumerate(reader, 2):
            try:
                players.append(Player(
                    id=row["id"].strip(), name=row["name"].strip(),
                    position=row["position"].strip().upper(),
                    team=row["team"].strip().upper(), opponent=row["opponent"].strip().upper(),
                    salary=int(row["salary"]), projection=float(row["projection"]),
                ))
            except (ValueError, AttributeError) as exc:
                raise ValueError(f"Invalid player CSV row {line}: {exc}") from exc
    validate_players(players)
    return tuple(players)


def validate_config(players: Sequence[Player], config: Config) -> Rules:
    if config.site not in RULES:
        raise ValueError(f"Unknown site: {config.site}")
    for name, value, lower, upper in (
        ("num_lineups", config.num_lineups, 1, math.inf),
        ("max_overlap", config.max_overlap, 0, 8),
        ("stack", config.stack, 0, 2),
        ("min_salary", config.min_salary, 0, RULES[config.site].salary_cap),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
            raise ValueError(f"{name} must be an integer between {lower} and {upper}")
    if not math.isfinite(config.time_limit) or config.time_limit <= 0:
        raise ValueError("time_limit must be finite and positive")
    if config.bring_back and not config.stack:
        raise ValueError("bring_back requires a QB stack")
    ids = {p.id for p in players}
    unknown = (set(config.locks) | set(config.excludes)) - ids
    if unknown:
        raise ValueError(f"Unknown lock/exclude IDs: {sorted(unknown)}")
    if set(config.locks) & set(config.excludes):
        raise ValueError("A player cannot be both locked and excluded")
    teams_count = len({p.team for p in players})
    for name, value, lower in (("strategic_min_teams", config.strategic_min_teams, 0),
                               ("strategic_max_teams", config.strategic_max_teams, 1)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= teams_count:
            raise ValueError(f"{name} must be an integer between {lower} and {teams_count} "
                             f"(the number of teams in the pool), or None")
    if (config.strategic_min_teams is not None and config.strategic_max_teams is not None
            and config.strategic_min_teams > config.strategic_max_teams):
        raise ValueError("strategic_min_teams cannot exceed strategic_max_teams")
    if config.strategic_team_scope not in ("all", "offense"):
        raise ValueError("strategic_team_scope must be 'all' or 'offense'")
    if (isinstance(config.secondary_stack, bool) or not isinstance(config.secondary_stack, int)
            or not 0 <= config.secondary_stack <= 2):
        raise ValueError("secondary_stack must be an integer between 0 and 2")
    skill_positions = {"RB", "WR", "TE"}
    for name, value in (("secondary_stack_positions", config.secondary_stack_positions),
                        ("flex_positions", config.flex_positions)):
        values = set(value)
        if not values or not values <= skill_positions:
            raise ValueError(f"{name} must be a non-empty subset of RB, WR, TE")
    if config.variance_floor is not None:
        if config.moments is None:
            raise ValueError("variance_floor requires moments to be set")
        if not math.isfinite(config.variance_floor) or config.variance_floor < 0:
            raise ValueError("variance_floor must be finite and non-negative")
    if config.covariance_cap is not None:
        if config.moments is None:
            raise ValueError("covariance_cap requires moments to be set")
        if not math.isfinite(config.covariance_cap):
            raise ValueError("covariance_cap must be finite")
    if config.correlation_ceiling is not None:
        if config.moments is None:
            raise ValueError("correlation_ceiling requires moments to be set")
        if config.variance_floor is None:
            raise ValueError("correlation_ceiling requires variance_floor to be set")
        if not 0 <= config.correlation_ceiling <= 1:
            raise ValueError("correlation_ceiling must be between 0 and 1")
    if config.covariance_cap is not None and config.correlation_ceiling is not None:
        raise ValueError("Set at most one of covariance_cap or correlation_ceiling")
    return RULES[config.site]


@dataclass
class ModelData:
    costs: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    integrality: NDArray[np.float64]
    rows: list[dict[int, float]]
    row_lower: list[float]
    row_upper: list[float]
    # Phase 3: dense covariance matrix aligned to the player pool's order,
    # set only when Config.moments is provided; used for covariance-cap
    # cuts added incrementally in generate_lineups.
    sigma: NDArray[np.float64] | None = None

    def add(self, coefficients: Mapping[int, float], lower: float, upper: float) -> None:
        self.rows.append(dict(coefficients))
        self.row_lower.append(lower)
        self.row_upper.append(upper)

    def sparse_rows(self) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float64]]:
        starts = [0]
        indices: list[int] = []
        values: list[float] = []
        for row in self.rows:
            indices.extend(row)
            values.extend(row.values())
            starts.append(len(indices))
        return (np.asarray(starts, dtype=np.int32), np.asarray(indices, dtype=np.int32),
                np.asarray(values, dtype=np.float64))


def dst_conflict_rows(
    players: Sequence[Player], formulation: str = "aggregate",
) -> list[tuple[dict[int, float], float, float]]:
    """Rows forbidding a selected DST from facing its own offense, as
    either the default aggregate ("big-M") form or a pairwise-disaggregated
    form. Both forms define the IDENTICAL feasible region for binary
    variables; they differ only in solver performance. See
    PAPER_IMPLEMENTATION_PLAN.md Phase 5 for the measured comparison
    (this function exists so formulation_experiments.py can build the
    pairwise alternative without duplicating the rest of build_model)."""
    if formulation not in ("aggregate", "pairwise"):
        raise ValueError(f"Unknown DST formulation: {formulation!r}")
    rows: list[tuple[dict[int, float], float, float]] = []
    for i, p in enumerate(players):
        if p.position != "DST":
            continue
        opponents = [j for j, other in enumerate(players)
                    if other.team == p.opponent and other.position != "DST"]
        if formulation == "aggregate":
            rows.append(({**{j: 1 for j in opponents}, i: 8}, -math.inf, 8))
        else:
            rows.extend(({i: 1, j: 1}, -math.inf, 1) for j in opponents)
    return rows


def build_model(players: Sequence[Player], config: Config) -> ModelData:
    validate_players(players)
    rules = validate_config(players, config)
    n = len(players)
    teams = sorted({p.team for p in players})
    games = sorted({p.game for p in players})

    sigma: NDArray[np.float64] | None = None
    if config.moments is not None:
        sigma = validate_moments(players, config.moments)

    size = n + len(teams) + len(games)
    strategic_index: dict[str, int] | None = None
    if config.strategic_min_teams is not None or config.strategic_max_teams is not None:
        strategic_index = {team: size + k for k, team in enumerate(teams)}
        size += len(teams)
    secondary_index: dict[str, int] | None = None
    if config.secondary_stack:
        secondary_index = {team: size + k for k, team in enumerate(teams)}
        size += len(teams)
    variance_pairs: list[tuple[int, int]] = []
    z_index: dict[tuple[int, int], int] = {}
    if config.variance_floor is not None:
        # validate_config guarantees moments is set whenever variance_floor
        # is set, so sigma was computed above.
        assert sigma is not None
        variance_pairs = [(j, k) for j in range(n) for k in range(j + 1, n) if sigma[j, k] != 0]
        z_index = {pair: size + i for i, pair in enumerate(variance_pairs)}
        size += len(variance_pairs)

    integrality = np.ones(size)
    for z in z_index.values():
        integrality[z] = 0  # continuous; linked exactly to binary x[j]*x[k] below
    data = ModelData(np.zeros(size), np.zeros(size), np.ones(size), integrality, [], [], [], sigma=sigma)
    data.costs[:n] = [-p.projection for p in players]
    for i, p in enumerate(players):
        if p.id in config.locks:
            data.lower[i] = 1
        if p.id in config.excludes:
            data.upper[i] = 0
    data.add({i: 1 for i in range(n)}, 9, 9)
    flex_eligible = set(config.flex_positions)
    for pos, lower, upper in (("QB", 1, 1), ("RB", 2, 3), ("WR", 3, 4),
                              ("TE", 1, 2), ("DST", 1, 1)):
        if pos in ("RB", "WR", "TE") and pos not in flex_eligible:
            # Fixing the upper bound to the base count removes this position's
            # eligibility for the single FLEX slot without needing per-slot
            # player assignment variables.
            upper = lower
        data.add({i: 1 for i, p in enumerate(players) if p.position == pos}, lower, upper)
    data.add({i: p.salary for i, p in enumerate(players)}, config.min_salary, rules.salary_cap)
    for k, team in enumerate(teams):
        members = {i: 1 for i, p in enumerate(players) if p.team == team}
        data.add(members, 0, rules.max_per_team)
        # A team indicator may be one only when at least one member is selected.
        data.add({**members, n + k: -1}, 0, math.inf)
    data.add({n + k: 1 for k in range(len(teams))}, rules.min_teams, math.inf)
    offset = n + len(teams)
    for k, game in enumerate(games):
        data.add({**{i: 1 for i, p in enumerate(players) if p.game == game},
                  offset + k: -1}, 0, math.inf)
    data.add({offset + k: 1 for k in range(len(games))}, rules.min_games, math.inf)

    if strategic_index is not None:
        offense_only = config.strategic_team_scope == "offense"
        for team in teams:
            idx = strategic_index[team]
            members = {i: 1 for i, p in enumerate(players)
                       if p.team == team and not (offense_only and p.position == "DST")}
            count = len(members)
            # Two-way link so idx is one if and only if the team is used.
            # A maximum/exact team count requires this direction too, unlike
            # the one-way site team indicators above, which only ever
            # enforce a minimum.
            data.add({**members, idx: -count}, -math.inf, 0)
            data.add({**members, idx: -1}, 0, math.inf)
        if config.strategic_min_teams is not None:
            data.add({idx: 1 for idx in strategic_index.values()},
                     config.strategic_min_teams, math.inf)
        if config.strategic_max_teams is not None:
            data.add({idx: 1 for idx in strategic_index.values()},
                     -math.inf, config.strategic_max_teams)

    if secondary_index is not None:
        eligible_positions = set(config.secondary_stack_positions)
        for team in teams:
            idx = secondary_index[team]
            members = {i: 1 for i, p in enumerate(players)
                       if p.team == team and p.position in eligible_positions}
            # idx may be one only when the team actually has enough selected
            # members; merely counting the team's existence is insufficient.
            data.add({**members, idx: -config.secondary_stack}, 0, math.inf)
        data.add({idx: 1 for idx in secondary_index.values()}, 1, math.inf)
        for i, p in enumerate(players):
            if p.position == "QB":
                # The secondary group cannot come from the selected QB's own
                # game: both of its teams are excluded whenever that QB
                # is chosen.
                for team in (p.team, p.opponent):
                    data.add({secondary_index[team]: 1, i: 1}, -math.inf, 1)

    if z_index:
        # z_index is only populated when config.variance_floor is not None
        # (and therefore sigma is not None; see above).
        assert sigma is not None and config.variance_floor is not None
        # z[j,k] linearizes x[j]*x[k] for the variance floor below: z<=x[j],
        # z<=x[k], and z>=x[j]+x[k]-1 force z to equal the product exactly
        # for binary x[j],x[k], even though z itself is only continuous.
        variance_row: dict[int, float] = {
            i: float(sigma[i, i]) for i in range(n) if sigma[i, i] != 0
        }
        for (j, k), z in z_index.items():
            data.add({z: 1, j: -1}, -math.inf, 0)
            data.add({z: 1, k: -1}, -math.inf, 0)
            data.add({z: 1, j: -1, k: -1}, -1, math.inf)
            variance_row[z] = variance_row.get(z, 0.0) + 2.0 * float(sigma[j, k])
        data.add(variance_row, config.variance_floor, math.inf)

    for i, p in enumerate(players):
        if p.position == "QB" and config.stack:
            receivers = {j: 1 for j, other in enumerate(players)
                         if other.team == p.team and other.position in ("WR", "TE")}
            data.add({**receivers, i: -config.stack}, 0, math.inf)
            if config.bring_back:
                opponents = {j: 1 for j, other in enumerate(players)
                             if other.team == p.opponent and other.position in ("RB", "WR", "TE")}
                data.add({**opponents, i: -1}, 0, math.inf)
    if config.avoid_dst_opponents:
        for coefficients, lower, upper in dst_conflict_rows(players, "aggregate"):
            data.add(coefficients, lower, upper)
    return data


def generate_lineups(players: Sequence[Player], config: Config = Config()) -> tuple[Lineup, ...]:
    players = tuple(players)
    data = build_model(players, config)
    lineups: list[Lineup] = []
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
            raise OptimizationError(
                f"Lineup {number}/{config.num_lineups}: {result.message}. "
                f"Only {len(lineups)} lineups solved; no partial output written. "
                "Relax constraints, request fewer lineups, or increase the time limit."
            )
        selected = [i for i in range(len(players)) if result.x[i] > 0.5]
        if len(selected) != 9:
            raise OptimizationError("Solver returned an invalid roster size")
        lineup = Lineup(tuple(players[i] for i in selected))
        lineup.slot_players()
        lineups.append(lineup)
        if number < config.num_lineups:
            data.add({i: 1 for i in selected}, -math.inf, config.max_overlap)
            if data.sigma is not None and (
                config.covariance_cap is not None or config.correlation_ceiling is not None
            ):
                x_vector = np.zeros(len(players))
                x_vector[selected] = 1.0
                cross = data.sigma @ x_vector
                if config.covariance_cap is not None:
                    cap = config.covariance_cap
                else:
                    # validate_config guarantees correlation_ceiling requires
                    # both moments and variance_floor to be set.
                    assert config.correlation_ceiling is not None and config.variance_floor is not None
                    # Conservative covariance bound implied by a desired
                    # correlation ceiling; see PAPER_IMPLEMENTATION_PLAN.md
                    # Phase 3 for why this is conservative, not exact.
                    previous_variance = max(0.0, float(x_vector @ data.sigma @ x_vector))
                    cap = config.correlation_ceiling * math.sqrt(config.variance_floor * previous_variance)
                coefficients = {i: float(cross[i]) for i in range(len(players)) if cross[i] != 0}
                if coefficients:
                    data.add(coefficients, -math.inf, cap)
    return tuple(lineups)


def lineup_variance(lineup: Lineup, moments: Moments) -> float:
    """Modeled variance of a lineup's total score (x.T @ Sigma @ x), computed
    as the sum of the lineup's own covariance submatrix since x is all-ones
    over exactly these players."""
    sigma = validate_moments(lineup.players, moments)
    return float(sigma.sum())


def lineup_covariance(lineup_a: Lineup, lineup_b: Lineup, moments: Moments) -> float:
    """Modeled covariance between two lineups' total scores, correctly
    accounting for any players they share."""
    combined = {p.id: p for p in (*lineup_a.players, *lineup_b.players)}
    union_players = tuple(combined.values())
    sigma = validate_moments(union_players, moments)
    index = {p.id: i for i, p in enumerate(union_players)}
    a_indices = [index[p.id] for p in lineup_a.players]
    b_indices = [index[p.id] for p in lineup_b.players]
    return float(sigma[np.ix_(a_indices, b_indices)].sum())


def realized_correlation(lineup_a: Lineup, lineup_b: Lineup, moments: Moments) -> float:
    """Modeled Pearson correlation between two lineups' total scores. Report
    this as a diagnostic after solving; it is not an input to the MILP."""
    variance_a = lineup_variance(lineup_a, moments)
    variance_b = lineup_variance(lineup_b, moments)
    if variance_a <= 0 or variance_b <= 0:
        raise ValueError("Both lineups must have strictly positive modeled variance")
    covariance = lineup_covariance(lineup_a, lineup_b, moments)
    return covariance / math.sqrt(variance_a * variance_b)



def write_lineups(path: Path, lineups: Sequence[Lineup]) -> None:
    """Replace output only after every lineup is generated and serialized."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8",
                                         dir=path.parent, delete=False, suffix=".csv") as stream:
            temporary = Path(stream.name)
            writer = csv.writer(stream)
            writer.writerow((*SLOTS, "salary", "projection"))
            for lineup in lineups:
                writer.writerow((*(p.id for p in lineup.slot_players()),
                                 lineup.salary, f"{lineup.projection:.6f}"))
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("players", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--site", choices=RULES, default="draftkings")
    parser.add_argument("--lineups", type=int, default=1)
    parser.add_argument("--max-overlap", type=int, default=7)
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default=None,
                       help="Apply a named strategy; cannot be combined with "
                            "--stack, --bring-back, --allow-dst-opponents, or "
                            "--secondary-stack.")
    parser.add_argument("--stack", type=int, choices=(0, 1, 2), default=None)
    parser.add_argument("--bring-back", action="store_true", default=None)
    parser.add_argument("--allow-dst-opponents", action="store_true", default=None)
    parser.add_argument("--secondary-stack", type=int, choices=(0, 1, 2), default=None)
    parser.add_argument("--secondary-stack-positions", nargs="+",
                       choices=("RB", "WR", "TE"), default=None)
    parser.add_argument("--strategic-min-teams", type=int, default=None)
    parser.add_argument("--strategic-max-teams", type=int, default=None)
    parser.add_argument("--strategic-team-scope", choices=("all", "offense"), default="all")
    parser.add_argument("--flex-positions", nargs="+", choices=("RB", "WR", "TE"), default=None)
    parser.add_argument("--min-salary", type=int, default=0)
    parser.add_argument("--lock", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--time-limit", type=float, default=60)
    args = parser.parse_args(argv)
    if args.players.resolve() == args.output.resolve():
        parser.error("Input and output must be different files")
    manual_stacking = (args.stack, args.bring_back, args.allow_dst_opponents, args.secondary_stack)
    if args.strategy is not None and any(value is not None for value in manual_stacking):
        parser.error("--strategy cannot be combined with --stack, --bring-back, "
                     "--allow-dst-opponents, or --secondary-stack")
    config = Config(
        site=args.site, num_lineups=args.lineups, max_overlap=args.max_overlap,
        min_salary=args.min_salary, locks=tuple(args.lock), excludes=tuple(args.exclude),
        time_limit=args.time_limit, strategic_min_teams=args.strategic_min_teams,
        strategic_max_teams=args.strategic_max_teams,
        strategic_team_scope=args.strategic_team_scope,
        secondary_stack_positions=(tuple(args.secondary_stack_positions)
                                   if args.secondary_stack_positions else ("RB", "WR", "TE")),
        flex_positions=tuple(args.flex_positions) if args.flex_positions else ("RB", "WR", "TE"),
    )
    if args.strategy is not None:
        config = apply_strategy(config, args.strategy)
    else:
        config = replace(
            config, stack=args.stack or 0, bring_back=bool(args.bring_back),
            avoid_dst_opponents=not bool(args.allow_dst_opponents),
            secondary_stack=args.secondary_stack or 0,
        )
    try:
        lineups = generate_lineups(load_players(args.players), config)
        write_lineups(args.output, lineups)
    except (OSError, ValueError, OptimizationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {len(lineups)} optimal lineups to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
