# Contributing

Thanks for your interest in `multi-entry-optimizer`. This is a small,
personal open-source project maintained on a best-effort basis.

## Development setup

Requires Python 3.11+.

```powershell
git clone https://github.com/sansbacon/multi-entry-optimizer.git
cd multi-entry-optimizer
python -m pip install -e ".[dev]"
```

## Running tests

```powershell
python -m pytest tests\ -q
```

All 206+ tests should pass. Please add focused tests for any behavior
change (a few cases that would fail without the fix/feature, not
exhaustive coverage of every input), preferably in the existing test
file for the module you're touching
(`tests/test_lineups.py`, `tests/test_evaluation.py`,
`tests/test_moments.py`, `tests/test_probability.py`, or
`tests/test_formulation_experiments.py`).

## Project layout

- `src/multi_entry_optimizer/` -- the installable package
  (`lineups.py`, `evaluation.py`, `probability.py`).
- `benchmark.py`, `formulation_experiments.py` -- dev-only solver/
  formulation experiments, intentionally **not** part of the installed
  package (see `pyproject.toml`).
- `tests/` -- pytest suite.
- `examples/players.csv` -- fictional example data for the CLI/README.
- `PAPER_IMPLEMENTATION_PLAN.md` -- the paper-to-code gap analysis and
  phase-by-phase implementation record; read this before adding a new
  feature that relates to the underlying academic paper.
- `CHANGELOG.md` -- release history.

## Before opening a pull request

1. Run the full test suite (above) and make sure it passes.
2. If you touch `src/multi_entry_optimizer/lineups.py`'s MILP
   construction, prefer validating new constraints against exhaustive
   enumeration on a small synthetic pool (see existing fixtures in
   `tests/test_lineups.py`/`tests/test_moments.py`) rather than trusting
   the solver's output alone.
3. If you claim a performance improvement, measure it (see
   `benchmark.py`/`formulation_experiments.py` for the existing
   measurement pattern) rather than asserting it from a tighter LP
   relaxation alone — this project found more than one case where a
   "stronger" formulation was measurably slower in practice.
4. Update `CHANGELOG.md` and relevant documentation
   (`README.md`/`PAPER_IMPLEMENTATION_PLAN.md`) for any user-visible
   change.
5. No real historical NFL contest or projection data is used anywhere in
   this project's tests; please don't represent any synthetic-fixture
   result as validated against real contest outcomes.

## Reporting issues

Use the GitHub issue tracker. Include a minimal reproduction (a small
player CSV and the CLI/API call) for any bug report.
