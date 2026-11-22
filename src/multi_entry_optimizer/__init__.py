"""multi_entry_optimizer: open-source mixed-integer NFL DFS lineup generation.

The top-level package exposes the core lineup-generation API (players,
site rules, configuration, lineup construction, strategy presets, and the
optional moment-aware variance/covariance MILP extensions from Phase 3).

Related but separable tooling lives in dedicated submodules rather than
the top-level namespace, since they serve different workflows:

* `multi_entry_optimizer.evaluation` -- historical contest replay,
  calibration, and correlation estimation (Phase 2).
* `multi_entry_optimizer.probability` -- Gaussian/Monte Carlo
  success-probability diagnostics and greedy portfolio selection (Phase 4).

Not affiliated with, endorsed by, or sponsored by the National Football
League (NFL) or any fantasy sports operator.
"""

from .lineups import (
    Config,
    Lineup,
    ModelData,
    Moments,
    OptimizationError,
    Player,
    POSITIONS,
    RULES,
    Rules,
    SLOTS,
    STRATEGIES,
    apply_strategy,
    build_model,
    dst_conflict_rows,
    generate_lineups,
    load_players,
    lineup_covariance,
    lineup_variance,
    main,
    realized_correlation,
    validate_config,
    validate_moments,
    validate_players,
    write_lineups,
)

__all__ = [
    "Config",
    "Lineup",
    "ModelData",
    "Moments",
    "OptimizationError",
    "Player",
    "POSITIONS",
    "RULES",
    "Rules",
    "SLOTS",
    "STRATEGIES",
    "apply_strategy",
    "build_model",
    "dst_conflict_rows",
    "generate_lineups",
    "load_players",
    "lineup_covariance",
    "lineup_variance",
    "main",
    "realized_correlation",
    "validate_config",
    "validate_moments",
    "validate_players",
    "write_lineups",
]

__version__ = "0.1.0"
