import csv
import itertools
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from multi_entry_optimizer import (
    Config, Lineup, OptimizationError, Player, RULES, STRATEGIES, SLOTS,
    apply_strategy, generate_lineups, load_players, main, validate_players,
    write_lineups,
)


@pytest.fixture
def pool():
    return load_players(Path(__file__).resolve().parents[1] / "examples" / "players.csv")


def legal(lineup, config):
    players = lineup.players
    counts = Counter(p.position for p in players)
    rules = RULES[config.site]
    if not (
        len(lineup.ids) == len(players) == 9
        and counts["QB"] == counts["DST"] == 1
        and 2 <= counts["RB"] <= 3 and 3 <= counts["WR"] <= 4
        and 1 <= counts["TE"] <= 2
        and config.min_salary <= lineup.salary <= rules.salary_cap
        and len({p.team for p in players}) >= rules.min_teams
        and len({p.game for p in players}) >= rules.min_games
        and max(Counter(p.team for p in players).values()) <= rules.max_per_team
        and set(config.locks) <= lineup.ids
        and not set(config.excludes) & lineup.ids
    ):
        return False
    qb = next(p for p in players if p.position == "QB")
    dst = next(p for p in players if p.position == "DST")
    if config.avoid_dst_opponents and any(
        p.team == dst.opponent and p.position != "DST" for p in players
    ):
        return False
    if config.stack and sum(
        p.team == qb.team and p.position in ("WR", "TE") for p in players
    ) < config.stack:
        return False
    if config.bring_back and not any(
        p.team == qb.opponent and p.position in ("RB", "WR", "TE") for p in players
    ):
        return False
    scope = players if config.strategic_team_scope == "all" else [
        p for p in players if p.position != "DST"
    ]
    strategic_teams = len({p.team for p in scope})
    if config.strategic_min_teams is not None and strategic_teams < config.strategic_min_teams:
        return False
    if config.strategic_max_teams is not None and strategic_teams > config.strategic_max_teams:
        return False
    if config.secondary_stack:
        eligible = set(config.secondary_stack_positions)
        qb_game = {qb.team, qb.opponent}
        group_counts = Counter(p.team for p in players if p.position in eligible)
        if not any(team not in qb_game and count >= config.secondary_stack
                  for team, count in group_counts.items()):
            return False
    base_needed = {"RB": 2, "WR": 3, "TE": 1}
    extra = {pos: counts[pos] - base_needed[pos] for pos in base_needed}
    if sum(extra.values()) != 1 or any(value < 0 for value in extra.values()):
        return False
    flex_position = next(pos for pos, value in extra.items() if value == 1)
    if flex_position not in set(config.flex_positions):
        return False
    return True


@pytest.mark.parametrize("site", RULES)
@pytest.mark.parametrize("stack,bring_back", [(0, False), (1, False), (2, True)])
def test_portfolios(pool, site, stack, bring_back):
    config = Config(site=site, num_lineups=5, max_overlap=6, stack=stack,
                    bring_back=bring_back, min_salary=40_000)
    lineups = generate_lineups(pool, config)
    assert len(lineups) == 5
    for lineup in lineups:
        assert legal(lineup, config)
        slots = lineup.slot_players()
        assert tuple(p.position for p in slots[:7]) == ("QB", "RB", "RB", "WR", "WR", "WR", "TE")
        assert slots[7].position in ("RB", "WR", "TE") and slots[8].position == "DST"
    for a, b in itertools.combinations(lineups, 2):
        assert len(a.ids & b.ids) <= 6
    assert all(a.projection >= b.projection - 1e-6 for a, b in zip(lineups, lineups[1:]))


@pytest.mark.parametrize("stack,bring_back", [(0, False), (1, False), (2, True)])
def test_matches_exhaustive_search(pool, stack, bring_back):
    ids = {"A_QB", "A_RB1", "A_RB2", "A_WR1", "A_WR2", "A_TE",
           "B_WR1", "C_RB1", "C_WR1", "C_TE", "C_DST", "D_DST", "D_WR1"}
    small = tuple(p for p in pool if p.id in ids)
    config = Config(num_lineups=2, max_overlap=7, stack=stack, bring_back=bring_back)
    candidates = [Lineup(p) for p in itertools.combinations(small, 9)]
    candidates = [lineup for lineup in candidates if legal(lineup, config)]
    assert candidates
    result = generate_lineups(small, config)
    assert result[0].projection == pytest.approx(max(p.projection for p in candidates))
    eligible = [p for p in candidates if len(p.ids & result[0].ids) <= config.max_overlap]
    assert result[1].projection == pytest.approx(max(p.projection for p in eligible))


@pytest.fixture
def small_pool(pool):
    # 1 QB (team A only), 3 RB, 5 WR, 2 TE, 2 DST candidates across all 4
    # teams/2 games. Small enough (13 players) for exhaustive 9-player
    # enumeration while still exercising every Phase 1 feature.
    ids = {"A_QB", "A_RB1", "A_RB2", "A_WR1", "A_WR2", "A_TE",
           "B_WR1", "C_RB1", "C_WR1", "C_TE", "C_DST", "D_DST", "D_WR1"}
    return tuple(p for p in pool if p.id in ids)


def exhaustive_best(small_pool, config):
    candidates = [Lineup(p) for p in itertools.combinations(small_pool, 9)
                 if legal(Lineup(p), config)]
    assert candidates, "expected at least one feasible lineup for this scenario"
    return max(c.projection for c in candidates)


@pytest.mark.parametrize("max_teams,expected", [(3, 154.6), (2, 150.7)])
def test_strategic_max_teams_exhaustive(small_pool, max_teams, expected):
    # Without any team-count bound, the unconstrained optimum uses 4 teams
    # (154.9); each bound below must exclude it in favor of a worse lineup
    # that respects the cap, proving the bound actually binds.
    config = Config(avoid_dst_opponents=False, strategic_max_teams=max_teams)
    unrestricted = exhaustive_best(small_pool, replace(config, strategic_max_teams=None))
    assert unrestricted == pytest.approx(154.9)
    expected_best = exhaustive_best(small_pool, config)
    assert expected_best == pytest.approx(expected)
    result, = generate_lineups(small_pool, config)
    assert legal(result, config)
    assert result.projection == pytest.approx(expected_best)
    assert len({p.team for p in result.players}) <= max_teams


@pytest.mark.parametrize("scope", ["all", "offense"])
def test_strategic_min_teams_constructive(pool, scope):
    # A fully locked 9-player lineup spans 4 teams in "all" scope (via its
    # DST) but only 3 in "offense" scope (the DST's team has no other
    # offensive player). This proves scope choice, not just the bound,
    # since the locks leave no freedom to pick a different lineup.
    locked = ("B_QB", "B_RB2", "A_RB2", "C_RB2", "A_WR3", "B_WR3", "C_WR3", "B_TE", "D_DST")
    config = Config(avoid_dst_opponents=False, locks=locked,
                    strategic_team_scope=scope, strategic_min_teams=4)
    if scope == "offense":
        with pytest.raises(OptimizationError, match="Infeasible"):
            generate_lineups(pool, config)
    else:
        result, = generate_lineups(pool, config)
        assert result.ids == set(locked)
        assert legal(result, config)


def test_strategic_team_scope_distinguishes_dst_only_team(pool):
    # Same locked lineup: strategic_max_teams=3 is feasible only once the
    # DST-only team is excluded from the counted scope.
    locked = ("B_QB", "B_RB2", "A_RB2", "C_RB2", "A_WR3", "B_WR3", "C_WR3", "B_TE", "D_DST")
    base = Config(avoid_dst_opponents=False, locks=locked, strategic_max_teams=3)
    with pytest.raises(OptimizationError, match="Infeasible"):
        generate_lineups(pool, replace(base, strategic_team_scope="all"))
    result, = generate_lineups(pool, replace(base, strategic_team_scope="offense"))
    assert result.ids == set(locked)
    assert legal(result, replace(base, strategic_team_scope="offense"))


@pytest.mark.parametrize("secondary_stack,expected", [(0, 153.1), (2, 151.4)])
def test_secondary_stack_exhaustive(small_pool, secondary_stack, expected):
    # The lone QB candidate is always on team A (game A-B), so a correct
    # secondary group must come from team C or D (never A or B). Team D has
    # only one eligible player, so the group must be on team C, and must
    # actually include 2 of its eligible members (not merely "use" team C,
    # which the unconstrained optimum already does via C_DST).
    config = Config(secondary_stack=secondary_stack)
    expected_best = exhaustive_best(small_pool, config)
    assert expected_best == pytest.approx(expected)
    result, = generate_lineups(small_pool, config)
    assert legal(result, config)
    assert result.projection == pytest.approx(expected_best)
    if secondary_stack:
        qb = next(p for p in result.players if p.position == "QB")
        qb_game = {qb.team, qb.opponent}
        group_counts = Counter(p.team for p in result.players
                               if p.position in ("RB", "WR", "TE"))
        assert any(team not in qb_game and count >= secondary_stack
                  for team, count in group_counts.items())


def test_secondary_stack_requires_different_game(small_pool):
    # Confirms the satisfying group cannot be team D (only one eligible
    # member) or the QB's own game (A/B, structurally excluded), leaving C
    # as the only team that can actually satisfy the requirement.
    result, = generate_lineups(small_pool, Config(secondary_stack=2))
    qb = next(p for p in result.players if p.position == "QB")
    qb_game = {qb.team, qb.opponent}
    group_counts = Counter(p.team for p in result.players if p.position in ("RB", "WR", "TE"))
    satisfying_teams = [team for team, count in group_counts.items()
                        if count >= 2 and team not in qb_game]
    assert satisfying_teams == ["C"]


@pytest.mark.parametrize("flex_positions,expected", [
    (("RB", "WR", "TE"), 153.1), (("WR", "TE"), 152.4), (("TE",), 151.4),
])
def test_flex_positions_exhaustive(small_pool, flex_positions, expected):
    config = Config(flex_positions=flex_positions)
    expected_best = exhaustive_best(small_pool, config)
    assert expected_best == pytest.approx(expected)
    result, = generate_lineups(small_pool, config)
    assert legal(result, config)
    assert result.projection == pytest.approx(expected_best)
    assert result.slot_players()[7].position in flex_positions


def test_strategies_apply_distinct_stacking_fields(pool):
    for name in STRATEGIES:
        config = apply_strategy(Config(num_lineups=2, max_overlap=6), name)
        lineups = generate_lineups(pool, config)
        assert len(lineups) == 2
        for lineup in lineups:
            assert legal(lineup, config)


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError, match="Unknown strategy"):
        apply_strategy(Config(), "not-a-real-strategy")


def test_strategy_cli_flag(pool, tmp_path):
    source = Path(__file__).resolve().parents[1] / "examples" / "players.csv"
    output = tmp_path / "lineups.csv"
    assert main([str(source), str(output), "--lineups", "2",
                "--strategy", "qb-stack-bring-back"]) == 0
    mapping = {p.id: p for p in pool}
    with output.open(newline="") as stream:
        rows = list(csv.reader(stream))[1:]
    config = apply_strategy(Config(), "qb-stack-bring-back")
    for row in rows:
        lineup = Lineup(tuple(mapping[id] for id in row[:9]))
        assert legal(lineup, config)


def test_strategy_cli_conflicts_with_manual_flags(tmp_path):
    source = Path(__file__).resolve().parents[1] / "examples" / "players.csv"
    output = tmp_path / "lineups.csv"
    with pytest.raises(SystemExit):
        main([str(source), str(output), "--strategy", "baseline", "--stack", "1"])


def test_no_new_options_changes_default_behavior(pool):
    # Confirms the Phase 1 additions are fully opt-in: identical results to
    # explicitly setting every new field to its default/no-op value.
    baseline = generate_lineups(pool, Config(num_lineups=3, max_overlap=6))
    extended = generate_lineups(pool, Config(
        num_lineups=3, max_overlap=6, strategic_min_teams=None,
        strategic_max_teams=None, strategic_team_scope="all", secondary_stack=0,
        flex_positions=("RB", "WR", "TE"),
    ))
    assert [l.ids for l in baseline] == [l.ids for l in extended]
    assert [l.projection for l in baseline] == [l.projection for l in extended]


def test_infeasible_secondary_stack_explains(small_pool):
    # Team C's eligible members are all excluded, leaving D (only 1 eligible
    # member) and the QB's own game A/B (structurally disallowed).
    config = Config(secondary_stack=2, excludes=("C_RB1", "C_WR1", "C_TE"))
    with pytest.raises(OptimizationError, match="Infeasible"):
        generate_lineups(small_pool, config)


def test_infeasible_strategic_min_teams_explains(pool):
    # Team D's players remain in the pool (so the min bound passes config
    # validation) but are all excluded from selection, so only 3 selectable
    # teams remain to satisfy a minimum of 4.
    config = Config(strategic_min_teams=4,
                    excludes=tuple(p.id for p in pool if p.team == "D"))
    with pytest.raises(OptimizationError, match="Infeasible"):
        generate_lineups(pool, config)


def test_infeasible_strategic_max_teams_explains(pool):
    # Locking one player from each of the 4 teams forces 4 distinct teams,
    # directly conflicting with a maximum of 2.
    config = Config(strategic_max_teams=2, locks=("A_QB", "B_RB1", "C_WR1", "D_TE"))
    with pytest.raises(OptimizationError, match="Infeasible"):
        generate_lineups(pool, config)


@pytest.mark.parametrize("changes", [
    {"strategic_min_teams": -1}, {"strategic_min_teams": 1.5},
    {"strategic_min_teams": 5}, {"strategic_max_teams": 0},
    {"strategic_max_teams": 5}, {"strategic_min_teams": 3, "strategic_max_teams": 2},
    {"strategic_team_scope": "offensive"}, {"secondary_stack": 3},
    {"secondary_stack": -1}, {"secondary_stack_positions": ()},
    {"secondary_stack_positions": ("QB",)}, {"flex_positions": ()},
    {"flex_positions": ("DST",)},
])
def test_invalid_phase1_config(pool, changes):
    with pytest.raises(ValueError):
        generate_lineups(pool, replace(Config(), **changes))
    config = Config(num_lineups=3, locks=("A_QB", "B_WR1"), excludes=("C_RB1",))
    assert all(legal(lineup, config) for lineup in generate_lineups(pool, config))


def test_minimum_salary_is_enforced(pool):
    config = Config(min_salary=49_900)
    lineup, = generate_lineups(pool, config)
    assert legal(lineup, config)
    assert lineup.salary >= 49_900


def test_draftkings_rejects_single_game(pool):
    single_game = tuple(p for p in pool if p.team in ("A", "B"))
    with pytest.raises(OptimizationError, match="infeasible"):
        generate_lineups(single_game, Config(avoid_dst_opponents=False))


@pytest.mark.parametrize("position", ["RB", "WR", "TE"])
def test_each_flex_position(pool, position):
    needed = {"RB": 2, "WR": 3, "TE": 1}
    # Fix base skill players and one extra of the requested position.
    locked = []
    for pos, count in needed.items():
        candidates = [p.id for p in pool if p.position == pos]
        locked.extend(candidates[:count + int(pos == position)])
    config = Config(locks=tuple(locked), avoid_dst_opponents=False)
    lineup, = generate_lineups(pool, config)
    assert legal(lineup, config)
    assert lineup.slot_players()[7].position == position


def test_single_lineup_and_zero_overlap(pool):
    assert len(generate_lineups(pool)) == 1
    config = Config(num_lineups=2, max_overlap=0, avoid_dst_opponents=False)
    first, second = generate_lineups(pool, config)
    assert legal(first, config) and legal(second, config)
    assert not first.ids & second.ids


def test_no_partial_output(pool, tmp_path, capsys):
    source = Path(__file__).resolve().parents[1] / "examples" / "players.csv"
    output = tmp_path / "existing.csv"
    output.write_text("preserve me", encoding="utf-8")
    # First lineup succeeds, then a locked QB prevents zero-overlap lineup two.
    assert main([str(source), str(output), "--lineups", "2", "--max-overlap", "0",
                 "--lock", "A_QB"]) == 1
    assert "Lineup 2/2" in capsys.readouterr().err
    assert output.read_text(encoding="utf-8") == "preserve me"
    assert list(tmp_path.iterdir()) == [output]


def test_infeasible(pool):
    with pytest.raises(OptimizationError, match="Infeasible"):
        generate_lineups(pool, Config(locks=("A_QB", "B_QB")))


@pytest.mark.parametrize("changes", [
    {"num_lineups": 0}, {"num_lineups": 1.5}, {"max_overlap": 9},
    {"max_overlap": -1}, {"stack": 3}, {"bring_back": True},
    {"time_limit": 0}, {"time_limit": float("nan")}, {"min_salary": 50_001},
    {"locks": ("missing",)}, {"locks": ("A_QB",), "excludes": ("A_QB",)},
    {"site": "unknown"},
])
def test_invalid_config(pool, changes):
    with pytest.raises(ValueError):
        generate_lineups(pool, replace(Config(), **changes))


@pytest.mark.parametrize("changes", [
    {"projection": float("nan")}, {"projection": float("inf")}, {"salary": 0},
    {"salary": 1.5}, {"position": "K"}, {"name": ""}, {"opponent": "A"},
    {"team": " A"},
])
def test_invalid_player(pool, changes):
    with pytest.raises(ValueError):
        validate_players((replace(pool[0], **changes), *pool[1:]))


def test_duplicate_and_inconsistent_teams(pool):
    with pytest.raises(ValueError, match="Duplicate"):
        validate_players((*pool, pool[0]))
    with pytest.raises(ValueError, match="Inconsistent"):
        validate_players((replace(pool[0], opponent="C"), *pool[1:]))


def test_negative_projections_are_not_clamped(pool):
    negative = tuple(replace(p, projection=-p.projection) for p in pool)
    lineup, = generate_lineups(negative)
    assert legal(lineup, Config())
    assert lineup.projection < 0


def test_csv_and_cli(pool, tmp_path):
    source = Path(__file__).resolve().parents[1] / "examples" / "players.csv"
    output = tmp_path / "lineups.csv"
    assert main([str(source), str(output), "--lineups", "3", "--stack", "2",
                 "--bring-back"]) == 0
    with output.open(newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == [*SLOTS, "salary", "projection"]
    assert len(rows) == 4
    mapping = {p.id: p for p in pool}
    for row in rows[1:]:
        lineup = Lineup(tuple(mapping[id] for id in row[:9]))
        assert legal(lineup, Config(stack=2, bring_back=True))
        assert int(row[9]) == lineup.salary
        assert float(row[10]) == pytest.approx(lineup.projection)


@pytest.mark.parametrize("content", [
    "", "name,position\nX,QB\n",
    "id,name,position,team,opponent,salary,projection\nx,X,QB,A,B,bad,2\n",
    "id,name,position,team,opponent,salary,projection\nx,X\n",
])
def test_bad_csv(tmp_path, content):
    source = tmp_path / "bad.csv"
    source.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_players(source)


def test_atomic_write_failure(pool, tmp_path):
    output = tmp_path / "lineups.csv"
    output.write_text("unchanged", encoding="utf-8")
    with pytest.raises(ValueError):
        write_lineups(output, [Lineup(pool[:1])])
    assert output.read_text(encoding="utf-8") == "unchanged"
    assert list(tmp_path.iterdir()) == [output]


def test_time_limit_is_explicit(pool):
    with pytest.raises(OptimizationError, match="Time limit"):
        generate_lineups(pool, Config(time_limit=1e-9))
