"""Reproducible end-to-end native HiGHS versus scipy.optimize.milp benchmark."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import asdict

import highspy
import numpy as np
import scipy

from multi_entry_optimizer import Config, Lineup, Player, build_model, generate_lineups


def synthetic_pool(teams: int, seed: int) -> tuple[Player, ...]:
    rng = np.random.default_rng(seed)
    positions = ["QB"] + ["RB"] * 3 + ["WR"] * 5 + ["TE"] * 2 + ["DST"]
    players = []
    for team in range(teams):
        for k, pos in enumerate(positions):
            salary = int(rng.integers(30, 81)) * 100
            projection = salary / 400 + float(rng.uniform(-3, 7))
            if pos == "DST":
                salary = int(rng.integers(25, 40)) * 100
                projection = float(rng.uniform(4, 10))
            players.append(Player(f"{team}_{k}", f"Synthetic {team} {k}", pos,
                                  f"T{team}", f"T{team ^ 1}", salary, projection))
    return tuple(players)


def native_lineups(players: tuple[Player, ...], config: Config) -> tuple[Lineup, ...]:
    data = build_model(players, config)
    solver = highspy.Highs()

    def check(status: highspy.HighsStatus) -> None:
        if status != highspy.HighsStatus.kOk:
            raise RuntimeError(f"Native HiGHS API failed: {status}")

    for name, value in (("output_flag", False), ("time_limit", config.time_limit),
                        ("mip_rel_gap", 0.0), ("mip_abs_gap", 0.0), ("threads", 1),
                        ("random_seed", 0)):
        check(solver.setOptionValue(name, value))
    size = len(data.costs)
    check(solver.addCols(size, data.costs, data.lower, data.upper, 0,
                         np.zeros(size + 1, dtype=np.int32),
                         np.array([], dtype=np.int32), np.array([], dtype=np.float64)))
    check(solver.changeColsIntegrality(
        size, np.arange(size, dtype=np.int32),
        np.full(size, highspy.HighsVarType.kInteger, dtype=np.uint8)))
    starts, indices, coefficients = data.sparse_rows()
    check(solver.addRows(len(data.rows), np.asarray(data.row_lower),
                         np.asarray(data.row_upper), len(indices), starts, indices, coefficients))
    results = []
    for number in range(config.num_lineups):
        run_status = solver.run()
        if solver.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f"Native lineup {number + 1}: "
                               f"{solver.modelStatusToString(solver.getModelStatus())}")
        check(run_status)
        values = solver.getSolution().col_value
        selected = [i for i in range(len(players)) if values[i] > 0.5]
        results.append(Lineup(tuple(players[i] for i in selected)))
        if number + 1 < config.num_lineups:
            check(solver.addRow(-np.inf, config.max_overlap, len(selected),
                                np.asarray(selected, dtype=np.int32), np.ones(len(selected))))
    return tuple(results)


def verify(players: tuple[Player, ...], lineups: tuple[Lineup, ...], config: Config) -> None:
    data = build_model(players, config)
    teams = sorted({p.team for p in players})
    games = sorted({p.game for p in players})
    for lineup in lineups:
        lineup.slot_players()
        selected = [i for i, p in enumerate(players) if p.id in lineup.ids]
        values = np.asarray(
            [int(p.id in lineup.ids) for p in players]
            + [int(any(p.team == team for p in lineup.players)) for team in teams]
            + [int(any(p.game == game for p in lineup.players)) for game in games],
            dtype=float,
        )
        for row, lower, upper in zip(data.rows, data.row_lower, data.row_upper):
            value = sum(coefficient * values[i] for i, coefficient in row.items())
            if not lower - 1e-6 <= value <= upper + 1e-6:
                raise RuntimeError("Benchmark generated an infeasible lineup")
        data.add({i: 1 for i in selected}, -np.inf, config.max_overlap)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teams", type=int, default=16)
    parser.add_argument("--lineups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.teams < 4 or args.teams % 2 or args.repeats < 1:
        parser.error("Use an even team count >=4 and repeats >=1")
    players = synthetic_pool(args.teams, args.seed)
    config = Config(num_lineups=args.lineups, stack=2, bring_back=True, time_limit=120)
    runners = {"highspy": native_lineups, "scipy_milp": generate_lineups}
    timings: dict[str, list[float]] = {name: [] for name in runners}
    objectives: dict[str, list[float]] = {}
    # Warm both paths, then alternate timed order to reduce startup/order bias.
    for runner in runners.values():
        runner(players, Config(stack=2, bring_back=True))
    for repeat in range(args.repeats):
        names = list(runners)
        if repeat % 2:
            names.reverse()
        for name in names:
            start = time.perf_counter()
            lineups = runners[name](players, config)
            timings[name].append(time.perf_counter() - start)
            verify(players, lineups, config)
            objectives[name] = [lineup.projection for lineup in lineups]
    # Tied earlier optima can produce different later greedy paths; compare only
    # the first optimum, whose feasible region is identical for both interfaces.
    if abs(objectives["highspy"][0] - objectives["scipy_milp"][0]) > 1e-5:
        raise RuntimeError("First-lineup optimal objectives differ")
    medians = {name: statistics.median(samples) for name, samples in timings.items()}
    print(json.dumps({
        "python": platform.python_version(), "platform": platform.platform(),
        "highspy": highspy.Highs().version(), "scipy": scipy.__version__,
        "numpy": np.__version__, "players": len(players), "seed": args.seed,
        "config": asdict(config), "seconds": timings, "median_seconds": medians,
        "fastest_measured": min(medians, key=lambda name: medians[name]),
        "scipy_over_highspy_ratio": medians["scipy_milp"] / medians["highspy"],
        "first_optimal_projection": objectives["highspy"][0],
        "all_lineups_feasible": True,
    }, indent=2))


if __name__ == "__main__":
    main()
