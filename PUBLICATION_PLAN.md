# Cleanup and publication plan: picking-winners-nfl

**Status: EXECUTED.** You reviewed this plan and answered the open
questions (see "Confirmed decisions" immediately below); Phases A-F were
then carried out. The rest of this document is the original plan as
drafted before execution, kept as the historical record of the
reasoning; where it conflicts with the summary below, the summary
reflects what actually happened.

## Confirmed decisions and what was executed

| Decision | Your answer | What was done |
| --- | --- | --- |
| Package/import name | `multi-entry-optimizer` / `import multi_entry_optimizer` | Renamed everywhere (PyPI name confirmed unclaimed) |
| License | MIT | `LICENSE` added, `pyproject.toml` updated |
| Author | sansbacon / sansbacon@gmail.com | Added to `pyproject.toml` and `LICENSE`; GitHub URLs assumed `github.com/sansbacon/multi-entry-optimizer` -- **verify/correct this username** before pushing |
| Repo host | GitHub, public | `git init` done locally; **the actual GitHub remote was NOT created** (no `gh` CLI available, no credentials provided) -- see "Remaining manual steps" below |
| PDF/documents/Julia file | Omit the manuscript (PDF + `documents/`), keep `picking_winners.jl` with credit | Both gitignored (not committed); found and documented the original repo's actual MPL-2.0 license and the paper's free arXiv listing -- see `NOTICE.md` |
| `src/` layout rename | Approved, import name `multi_entry_optimizer` | Done: `nfl_dfs.py`/`historical_evaluation.py`/`probability_diagnostics.py` → `src/multi_entry_optimizer/{lineups,evaluation,probability}.py` + `__init__.py` + `__main__.py` |
| CI scope | Full | `.github/workflows/test.yml` (3.11-3.13 matrix + build) and `publish.yml` (tag-triggered, Trusted Publishing) added |
| README | Full rewrite | Done: quickstart-first; phase narrative moved to `CHANGELOG.md` |

**Additional work done beyond the original plan's scope, discovered
during execution:**
- Fixed the 2 pre-existing type diagnostics (`Player.game`, `ModelData.add`).
- Verified the original Julia repository's actual license (Mozilla
  Public License 2.0) and the paper's free arXiv listing
  (<https://arxiv.org/abs/1604.01455>) instead of leaving these as
  open/unverified questions -- see `NOTICE.md` for the full provenance
  write-up, including the honest nuance that this Python code began as a
  conversion of the MPL-2.0 Julia file (not an independent reimplementation).
- Added `CHANGELOG.md`, `CONTRIBUTING.md`, and `NOTICE.md` (not just a
  LICENSE file).
- Added `src/multi_entry_optimizer/__main__.py` so `python -m
  multi_entry_optimizer` genuinely works, as documented in the README
  (caught by an actual clean-venv smoke test, not assumed).
- Built the sdist/wheel, installed the wheel into a disposable virtual
  environment with no repo checkout present, and confirmed the CLI
  (`python -m multi_entry_optimizer`) and all three import paths
  (`multi_entry_optimizer`, `.evaluation`, `.probability`) work; verified
  the wheel contains only the 4 intended package files (no tests,
  examples, or excluded reference material leaked in).
- Validated every Markdown cross-reference link and header anchor across
  `README.md`/`PAPER_IMPLEMENTATION_PLAN.md`/`CHANGELOG.md` programmatically
  (not by eyeballing), fixing 2 real anchor mismatches this caught.
- `git init`'d, staged, and committed (2 commits on `main`); verified via
  `git status --short` that nothing excluded (PDF, `documents/`, `.vscode/`,
  build artifacts) was accidentally staged.
- Full 206-test suite re-run and passing after every structural change.

## Remaining manual steps (not possible for me to complete)

1. **Create the actual GitHub repository** (`sansbacon/multi-entry-optimizer`,
   public) and push: `git remote add origin
   https://github.com/sansbacon/multi-entry-optimizer.git && git push -u
   origin main`. No `gh` CLI is installed in this environment and I have
   no GitHub credentials, so this step needs you (or install `gh` and
   run `gh repo create`).
2. **Verify the GitHub username** `sansbacon` is actually correct --
   `pyproject.toml`'s `[project.urls]` and `NOTICE.md`/`README.md`
   assume it from your email handle; update if wrong.
3. **PyPI release** (Phase F in the original plan below): create a PyPI
   account if needed, register this repo+`publish.yml` as a Trusted
   Publisher on PyPI (one-time, done on pypi.org after the GitHub repo
   exists), then push a `v0.1.0` tag to trigger the publish workflow --
   or do a manual `twine upload` first release. I built and smoke-tested
   the wheel locally, but did not and cannot publish to the real PyPI
   index (no account/credentials available to me).
4. **Confirm the `multi-entry-optimizer` PyPI name is still unclaimed**
   at the moment you actually publish (it was unclaimed when checked
   during this session).

---


## How to read this plan

I asked you a batch of clarifying questions (package name, license,
repo host, author metadata, dev-tool packaging, CI, README approach)
before writing this, but you were unavailable to answer. Per your
instruction to produce a plan for review, I proceeded using explicit,
clearly-labeled **assumptions** instead of blocking. Every assumption is
called out in the table below and reiterated inline wherever it drives a
concrete recommendation. Treat this entire document as a draft: accept,
reject, or redirect any item when you review it.

## Assumptions made (no response available)

| Decision | Assumption used | Why | Confidence |
| --- | --- | --- | --- |
| PyPI package name | `picking-winners-nfl` (unchanged) | Verified unclaimed on PyPI as of this review (`pypi.org/simple/picking-winners-nfl/` returns 404). Re-verify at publish time; names can be claimed between now and then. | Medium — you may want a different public name |
| License | MIT | Matches HiGHS's own license; simplest, most common choice for a small permissive Python library; compatible with BSD-licensed SciPy/NumPy | Medium — Apache-2.0/BSD-3-Clause are equally reasonable |
| Author/maintainer metadata | Left as an explicit placeholder | I don't have your name/email/preferred public identity to put in `pyproject.toml` or a LICENSE copyright line | N/A — you must supply this before first publish |
| Repo host | GitHub | Most common; enables GitHub Actions CI and PyPI Trusted Publishing without stored secrets | Medium |
| Package dev tools (`benchmark.py`, `formulation_experiments.py`) | Excluded from the installable PyPI package, kept in the source repo | They are solver-comparison/experiment scripts for contributors, not part of the public API end users need | High — matches `pyproject.toml`'s existing `py-modules` list, which already excludes them |
| CI scope | Full: tests on push/PR, automated publish on version tags | You get a repeatable, low-effort release process; avoids manual upload mistakes | Medium — "tests only, manual publish" is a reasonable lighter alternative |
| README approach | Full rewrite: standard package structure (install/quickstart first); move the Phase 1-5 development narrative to a separate doc | The current README is excellent **internal** documentation of a development session, but is not what a first-time PyPI visitor expects to read before `pip install` | Medium — "light cleanup only" is a valid lower-effort alternative if you're attached to the current structure |

If any of these don't match your intent, tell me and I'll revise the
plan before any execution begins.

## Current-state summary (facts, not assumptions)

- **Packages/modules shipped today** (`pyproject.toml`
  `[tool.setuptools] py-modules`): `nfl_dfs`, `historical_evaluation`,
  `probability_diagnostics`. One CLI entry point: `nfl-dfs = "nfl_dfs:main"`.
- **Not shipped, repo-only**: `benchmark.py`, `formulation_experiments.py`
  (dev/experiment tools; `formulation_experiments.py` itself imports
  `benchmark.py`).
- **Tests**: 206 passing, across `tests/test_nfl_dfs.py`,
  `tests/test_historical_evaluation.py`, `tests/test_moments.py`,
  `tests/test_probability_diagnostics.py`,
  `tests/test_formulation_experiments.py`.
- **Dependencies**: runtime `scipy>=1.11,<2`, `numpy>=1.24,<3`; dev extra
  `pytest>=8,<10`, `highspy>=1.10,<2`. `requires-python = ">=3.11"`.
- **No LICENSE file exists.**
- **Not a git repository** (`.git` absent) — no remote, no commit
  history, no GitHub/GitLab URL available for packaging metadata.
- **No CI configuration** (no `.github/` directory).
- **No CHANGELOG, CONTRIBUTING, or CODE_OF_CONDUCT files.**
- **`pyproject.toml` is missing**: license declaration, author/maintainer
  info, classifiers, keywords, project URLs (homepage/repo/issues),
  and an explicit `readme` declaration.
- **Reference materials present in the working directory** (see the
  dedicated compliance section below for why these need a decision
  before anything is pushed publicly):
  - `picking_winners.jl` — a **third-party** Julia reference
    implementation (the paper's own authors' GitHub code, used here only
    as a conversion baseline), not something you wrote.
  - `Picking-Winners.pdf` — the full academic manuscript PDF.
  - `documents/*.md` (36 files) — substantial extracted/parsed text from
    that same manuscript.
  - `PAPER_IMPLEMENTATION_PLAN.md` — your own analysis/plan; fine to
    publish, but it currently links to and quotes short passages from
    `documents/*.md` and the PDF.
- **Secrets scan**: a targeted grep for common credential patterns
  (API keys, passwords, tokens, private key headers) found **no
  matches**. No evidence of secrets in the repo. (This is a lightweight
  check, not a substitute for a dedicated secret-scanning tool in CI —
  see Phase E.)
- Pylance currently reports **3 pre-existing, non-blocking diagnostics**
  in `nfl_dfs.py`: an environment-specific `scipy.optimize`/`scipy.sparse`
  import-resolution issue (not a real bug — the package imports and runs
  correctly), a `Player.game` property return-type mismatch
  (`tuple[str, str]` declared vs. `tuple[str, ...]` inferred), and a
  `ModelData.add` parameter-variance issue (`dict[int, int]` passed where
  `dict[int, float]` is declared — `dict` is invariant in its value type;
  the fix is to type the parameter as `Mapping[int, float]`). None affect
  runtime behavior, but a published package should ship with a clean type
  checker, so these are in the cleanup phase below.

## Important: a compliance decision to make before anything is pushed publicly

This needs your explicit judgment, not just my assumption, so I'm not
silently resolving it:

**`Picking-Winners.pdf` and `documents/*.md` contain substantial
extracted text from the academic manuscript this project is based on.**
That manuscript is a copyrighted, INFORMS-submitted work by its authors
(Hunter, Vielma, and Zaman) — not something either of us authored or
hold distribution rights to. Redistributing the full PDF and extensive
parsed excerpts of it in a public GitHub repository (and, if included,
inside a PyPI package) is a meaningfully different act than citing or
linking to the paper, and carries real copyright/redistribution risk
that I can't clear on your behalf.

Separately, **`picking_winners.jl` is third-party code**: per its own
header comment and the paper's references, it is the paper authors'
own GitHub reference implementation, used here only as a conversion
baseline — not your original work. Its original repository's license
terms (permitting or restricting redistribution) were not verified as
part of this plan.

**Recommendation** (not yet applied to any file):
1. Do not include `Picking-Winners.pdf` or `documents/*.md` in the public
   GitHub repository or in the PyPI package. Keep them locally only.
2. Replace the PDF/`documents/` links in `PAPER_IMPLEMENTATION_PLAN.md`
   and `README.md` with a citation and a link to wherever the paper is
   legitimately hosted (its authors' site, SSRN, arXiv, or the publisher),
   so the analysis remains traceable without redistributing the text.
3. Before publishing `picking_winners.jl` publicly, confirm the original
   `Fantasy-Hockey-IP-Code` repository's license permits redistribution,
   and keep/add clear attribution to the original authors either way.
4. If you'd rather keep the PDF/`documents/` folder in a *private* repo
   or a local-only location while still publishing the *Python* package
   and a public-but-separate source repo, that's a clean way to keep
   both your research notes and a compliant public release.

I have not deleted, moved, or altered any of these files — this is
flagged for your decision, to be carried out (or explicitly overridden)
in Phase A below.

## Gap analysis: what's needed before this is "PyPI-ready"

| Gap | Why it matters | Phase |
| --- | --- | --- |
| No LICENSE | PyPI will publish an unlicensed package, which legally defaults to "all rights reserved" — nobody can safely use it | A |
| Manuscript/third-party-code redistribution risk | See compliance section above | A |
| Incomplete `pyproject.toml` metadata | PyPI listing will look unfinished (no classifiers/links); `pip download` tooling and search can't categorize it | B |
| No `src/` layout; flat top-level modules | `pip install picking-winners-nfl` then `import nfl_dfs` is a common source of end-user confusion (package name ≠ import name); flat top-level modules also risk shadowing other installed packages named `nfl_dfs` | B |
| Build not verified from a clean wheel install | Current testing is all from an editable install (`pip install -e .`); an sdist/wheel can accidentally include or exclude the wrong files | B |
| 3 outstanding type-checker diagnostics | A published library should type-check cleanly | C |
| No linter/formatter/type-checker wired into CI | Nothing currently enforces code style or catches regressions automatically | C, E |
| README reads as a development log, not user docs | First-time visitors need install/quickstart before a phase-by-phase implementation history | D |
| No CHANGELOG | Users/PyPI can't see what changed between versions | D |
| No CONTRIBUTING guide | Unclear how outside contributors should set up/test/submit changes | D |
| Not a git repository | No version history, no GitHub Actions, no PyPI Trusted Publishing, no public issue tracker | E |
| No CI | Nothing currently prevents a broken build or failing tests from being released | E |
| No release process documented | Risk of manual publishing mistakes (wrong files included, forgotten version bump, lost reproducibility) | F |
| Package name availability can change between now and publish | A name confirmed available today isn't guaranteed available later | F |

## Phase A: Legal and identity cleanup

1. Resolve the compliance decision above (PDF/`documents/`/Julia file
   handling) — **your call, needed before Phase E (pushing to GitHub)**.
2. Add a `LICENSE` file (MIT text recommended, with your name/organization
   as copyright holder — placeholder until you provide it).
3. Add a short citation of the academic paper (authors, title, and a link
   to wherever it's legitimately hosted) to `README.md` and
   `PAPER_IMPLEMENTATION_PLAN.md`, replacing any direct excerpt/PDF links
   if you choose the recommendation above.
4. Add a one-line "not affiliated with the NFL" disclaimer to the README.
   Using "NFL" in a project name/description is extremely common for
   unofficial fantasy-football tooling (e.g. the well-known `nfl-data-py`
   package), but an explicit non-affiliation disclaimer is cheap,
   standard practice, and reduces any ambiguity.

**Acceptance criteria:** `LICENSE` exists and is referenced from
`pyproject.toml`; no copyrighted third-party manuscript text or PDF is
present in whatever gets pushed publicly (per your decision); the Julia
file's redistribution status is either confirmed permissible or the file
is excluded from the public push.

## Phase B: Packaging metadata and structure

1. **Expand `pyproject.toml`** with (illustrative, not yet applied):

   ```toml
   [project]
   name = "picking-winners-nfl"
   version = "0.1.0"
   description = "Open-source mixed-integer NFL DFS lineup generation"
   readme = "README.md"
   license = "MIT"
   authors = [{ name = "PLACEHOLDER", email = "placeholder@example.com" }]
   requires-python = ">=3.11"
   keywords = ["nfl", "daily-fantasy-sports", "optimization", "mixed-integer-programming"]
   classifiers = [
       "Development Status :: 3 - Alpha",
       "Intended Audience :: Developers",
       "License :: OSI Approved :: MIT License",
       "Programming Language :: Python :: 3",
       "Programming Language :: Python :: 3.11",
       "Programming Language :: Python :: 3.12",
       "Programming Language :: Python :: 3.13",
       "Topic :: Scientific/Engineering :: Mathematics",
       "Topic :: Games/Entertainment",
   ]

   [project.urls]
   Homepage = "https://github.com/PLACEHOLDER/picking-winners-nfl"
   Repository = "https://github.com/PLACEHOLDER/picking-winners-nfl"
   Issues = "https://github.com/PLACEHOLDER/picking-winners-nfl/issues"
   Changelog = "https://github.com/PLACEHOLDER/picking-winners-nfl/blob/main/CHANGELOG.md"
   ```

   (Modern setuptools supports the PEP 639 `license = "MIT"` SPDX-string
   form shown above; confirm against the setuptools version pinned in
   `[build-system]` at execution time, since older releases expect the
   `{text = "MIT"}` table form instead.)

2. **Decide on a `src/` layout rename** (flagged as its own decision
   point, since it's a structural rename across the whole codebase, not
   a metadata tweak):
   - Current: `import nfl_dfs`, `import historical_evaluation`,
     `import probability_diagnostics` (top-level flat modules).
   - Proposed: `src/picking_winners_nfl/{lineups,evaluation,probability}.py`
     with `src/picking_winners_nfl/__init__.py` re-exporting the public
     API, so `pip install picking-winners-nfl` pairs with
     `import picking_winners_nfl` — no name/import mismatch.
   - This is a breaking rename of every internal and test import, and of
     the documented public API (`from nfl_dfs import Config, ...`
     becomes `from picking_winners_nfl import Config, ...` or similar).
     Since nothing has been published yet, there's no external backward
     compatibility to break — but it touches every module, every test
     file, the CLI entry point, and all current documentation examples.
   - **Recommendation:** do this rename now, before the first publish,
     specifically because it's free today and increasingly costly once
     real users depend on `import nfl_dfs`. I have not performed this
     rename; it belongs in an execution pass after you approve it.
3. Verify package contents with `python -m build` (produces `dist/*.tar.gz`
   and `dist/*.whl`); inspect the wheel's file list to confirm only the
   intended modules are included (no `tests/`, `examples/`, `documents/`,
   PDFs, or `.egg-info` leaking in); install the built wheel into a
   throwaway virtual environment and confirm `import <package>` and the
   `nfl-dfs` console script both work with **no other files present**
   (catches "only worked because of the editable install" bugs).
4. Delete the stale `picking_winners_nfl.egg-info/` build artifact before
   the first commit (it's already `.gitignore`d, so this is just local
   tidiness, not a packaging requirement).
5. Reconsider whether `examples/players.csv` should ship inside the
   package (for a bundled quickstart dataset) or stay example-only in the
   repo; recommend **not** bundling it in the wheel (keep it as a
   repo-only example, referenced from the README/docs via a relative
   path for people who clone the repo) to keep the installed package
   minimal.

**Acceptance criteria:** `python -m build` succeeds; the built wheel's
file list contains only intended package files; `pip install` of the
built wheel into an empty venv (no repo checkout present) succeeds and
the CLI/API both work; `pyproject.toml` has complete, accurate metadata.

## Phase C: Code and test cleanup

1. Fix the 2 genuine type-annotation issues noted above:
   - `Player.game` property: either declare the return type as
     `tuple[str, ...]` (matching what `sorted()` actually returns) or
     restructure to `tuple(sorted((self.team, self.opponent)))[0],
     tuple(...)[1]`-style unpacking so the 2-tuple type is provably exact.
   - `ModelData.add`: change the `coefficients` parameter type from
     `dict[int, float]` to `Mapping[int, float]` (covariant in value
     type), so passing a `dict[int, int]` type-checks correctly without
     requiring every caller to pre-cast to float.
2. Adopt `ruff` (lint + format) and `mypy` or `pyright --project` as
   repeatable, CI-enforced checks; run once locally and fix whatever
   surfaces (expect this to be a small amount of cleanup given the
   codebase's current discipline).
3. Add a lightweight secret-scanning step to CI (e.g., `gitleaks` or
   GitHub's built-in secret scanning once the repo is pushed) as an
   ongoing safeguard, not a one-time check.
4. Decide whether `tests/test_formulation_experiments.py` (which
   transitively requires `highspy` via `benchmark.py`) should be allowed
   to **skip gracefully** when `highspy` isn't installed
   (`pytest.importorskip("highspy")`), so `pip install -e .` without the
   `[dev]` extra doesn't hard-fail test collection. Recommended as a
   small robustness improvement, not required (today's `[dev]` extra
   already includes `highspy`, so this only matters for partial installs).
5. Re-run the full test suite (currently 206 tests) after any of the
   above changes to confirm no regressions.

**Acceptance criteria:** zero outstanding type-checker diagnostics in
shipped modules; `ruff`/linter run clean; all 206+ tests still pass.

## Phase D: Documentation overhaul

1. **Rewrite `README.md`** in a standard package order: badges (PyPI
   version, CI status, license — added once Phase E/F exist), one- or
   two-sentence pitch, install command, a single quickstart code block
   (generate a lineup from the bundled example CSV), a short feature
   list (one line per phase, linking onward rather than inlining the
   full narrative), CLI usage, a "Documentation" section linking to
   `PAPER_IMPLEMENTATION_PLAN.md` (or a renamed `docs/` equivalent) for
   the full technical/phase history, license section, and the paper
   citation/disclaimer from Phase A.
2. **Add `CHANGELOG.md`** (Keep a Changelog style), retroactively
   summarizing Phases 1-5 as the `0.1.0` entry, so future releases have
   somewhere to append to.
3. **Add `CONTRIBUTING.md`**: dev environment setup, how to run tests/lint,
   branch/PR expectations, and a pointer to `PAPER_IMPLEMENTATION_PLAN.md`
   for anyone implementing a new phase or extension.
4. **Optional**: `CODE_OF_CONDUCT.md` (standard Contributor Covenant) and
   `SECURITY.md` (how to report a vulnerability) if you want a fully
   conventional public-OSS-repo layout. Recommended but not blocking.
5. Decide the fate of `PAPER_IMPLEMENTATION_PLAN.md` itself: keep it at
   the repo root (simplest), or move it under a new `docs/` directory
   alongside the rewritten README's more detailed sections. Either is
   fine; repo-root is lower effort and matches the project's existing
   convention.

**Acceptance criteria:** a first-time reader can install and run a
working example within the first screen of the README; `CHANGELOG.md`
and `CONTRIBUTING.md` exist; no broken internal links (verify the same
way prior phases in this project did — a quick script checking every
Markdown link target exists).

## Phase E: Version control and CI/CD

1. `git init`; add a sensible initial `.gitignore` (already present and
   reasonable — extend if the `src/` rename or new build tooling
   introduces new artifact paths).
2. Resolve the Phase A compliance decision **before** this step, since
   once pushed to a public GitHub repo, prior commits remain recoverable
   from history even if files are later deleted.
3. Create the GitHub repository (public, assuming that's the intent —
   confirm), push the initial commit.
4. Add `.github/workflows/test.yml`: on push/PR, matrix over Python
   3.11/3.12/3.13 (Ubuntu at minimum; add Windows/macOS if you want
   cross-platform confidence, at the cost of slower CI), running
   `pip install -e .[dev]` then `pytest`, plus the Phase C lint/type
   checks.
5. Add `.github/workflows/publish.yml`: triggered on pushing a `v*` tag
   (or on a published GitHub Release), running `python -m build` and
   publishing via **PyPI Trusted Publishing** (OIDC-based; no long-lived
   API token stored as a GitHub secret) — this requires the PyPI project
   to have the GitHub repo/workflow registered as a trusted publisher
   first (a one-time manual step on PyPI, see Phase F).
6. Enable GitHub's built-in secret scanning / Dependabot alerts (free,
   one-click repo settings, not code).

**Acceptance criteria:** pushing a commit triggers the test workflow and
it passes; pushing a `v0.1.0`-style tag triggers (but does not
necessarily complete, until Phase F's PyPI-side setup exists) the
publish workflow.

## Phase F: First release to PyPI

1. Create a PyPI account (and a TestPyPI account) if you don't already
   have one.
2. **Dry run on TestPyPI first**: `python -m build`, then
   `twine upload --repository testpypi dist/*`; install from TestPyPI
   into a clean venv (`pip install --index-url
   https://test.pypi.org/simple/ picking-winners-nfl`) and smoke-test the
   CLI and a basic `generate_lineups` call.
3. On PyPI, create the project (either by reserving the name with a
   first manual upload, or via PyPI's newer "pending publisher" flow
   that lets you register a Trusted Publisher before any upload exists)
   and link it to the GitHub repo + `publish.yml` workflow for Trusted
   Publishing.
4. Cut the real `v0.1.0` release: tag, let CI build and publish (or
   publish manually with `twine upload dist/*` if you prefer a manual
   first release before automating).
5. **Post-publish verification**: in a fresh virtual environment with no
   local repo checkout, `pip install picking-winners-nfl`, run the CLI
   against the example CSV (or a small inline example from the README),
   and confirm it matches the behavior verified in Phase B/C.

**Acceptance criteria:** `pip install picking-winners-nfl` works from a
machine with no prior knowledge of this repo; the installed CLI and
Python API both function as documented.

## Phase G: Ongoing maintenance (lightweight, for your awareness)

- **Versioning**: SemVer going forward — patch for bugfixes, minor for
  backward-compatible additions (e.g., a future Phase 6), major for any
  breaking API change (hopefully rare, given the Phase B `src/` rename
  happens *before* the first release, not after).
- **Changelog discipline**: update `CHANGELOG.md` as part of each PR that
  changes user-visible behavior, not as an afterthought at release time.
- **Dependency upkeep**: periodically bump the `scipy`/`numpy` upper
  bounds as new major versions are released and verified compatible.
- **Issue/PR response expectations**: up to you to set (e.g., a note in
  `CONTRIBUTING.md` that this is a personal project maintained on a
  best-effort basis is a common, honest default for small libraries).

## Suggested execution order

Phases A and B's structural decisions (compliance handling, `src/`
rename, package name/license/author finalization) should be confirmed
**before** any execution begins, since later phases (D's README, E's
public push, F's PyPI listing) all depend on those choices being settled
first. Phases C and D can happen in parallel once B is confirmed. E and
F are necessarily sequential (can't publish before the repo and CI
exist). G is continuous, starting at first release.

## Open questions for your review

Repeating these here since you weren't available when I first asked,
so they're easy to answer in one pass when you do review this plan:

1. Keep `picking-winners-nfl` as the PyPI name, or choose something else? Change to multi-entry-optimizer
2. MIT, Apache-2.0, BSD-3-Clause, or another license? MIT
3. What name/email should appear in `pyproject.toml` authors and the
   LICENSE copyright line? sansbacon@gmail.com
4. GitHub (or another host) for the source repository — and should it
   be public or private? github public
5. How do you want to handle `Picking-Winners.pdf` / `documents/*.md` /
   `picking_winners.jl` given the compliance note above? Omit both files from repo. Add credits where appropriate.
6. Approve (or veto) the proposed `src/`-layout rename
   (`import nfl_dfs` → `import picking_winners_nfl`) before first release? import multi_entry_optimizer
7. Full CI (tests + automated publish-on-tag), tests-only CI, or no CI? Full CI
8. Full README rewrite, light cleanup, or keep as-is? Full README rewrite
