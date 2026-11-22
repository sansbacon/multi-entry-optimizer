# Notice: provenance, attribution, and third-party materials

This project's original Python source (`src/multi_entry_optimizer/`,
`benchmark.py`, `formulation_experiments.py`, and `tests/`) is made
available under the MIT License (see `LICENSE`). This file documents its
provenance and a licensing nuance worth your attention before the first
public release.

## Academic paper

This library implements, generalizes, and extends integer-programming
formulations described in:

> Hunter, D. S., Vielma, J. P., & Zaman, T. (2016). *Picking Winners
> Using Integer Programming.* Available freely on arXiv:
> <https://arxiv.org/abs/1604.01455> (PDF:
> <http://arxiv.org/pdf/1604.01455v2.pdf>).

Because the paper is freely available on arXiv, this repository links to
it directly rather than redistributing a local copy. No PDF or extracted
manuscript text is committed to this repository or published in the
PyPI package (the local working copy used during development,
`Picking-Winners.pdf` and `documents/*.md`, is excluded via
`.gitignore`); this keeps the citation unambiguous without raising a
separate redistribution question for the publisher's own PDF formatting
and layout.

## `picking_winners.jl` and its relationship to this project

`picking_winners.jl` in this repository is the original authors' Julia
reference implementation (for fantasy *hockey*), retained for provenance.
It comes from the paper authors' own repository,
<https://github.com/dscotthunter/Fantasy-Hockey-IP-Code>, which is
licensed under the **Mozilla Public License 2.0 (MPL-2.0)**.

**This Python package began as a conversion of `picking_winners.jl`**
(adapting its modeling approach from hockey to NFL classic-contest
rules), and Phases 1-5 then added substantial independent extensions
(structural heuristics, historical-contest evaluation, moment-aware
constraints, probability diagnostics, and formulation benchmarking) not
present in the original file. Whether the earliest, most directly-ported
portions of `src/multi_entry_optimizer/lineups.py` constitute an
MPL-covered "Modification" of `picking_winners.jl`, as opposed to an
independent implementation of the publicly described mathematical
formulations (which are not themselves protected by the original code's
copyright), is a genuine judgment call this project has not had
independent legal review to resolve.

Practically, MPL-2.0 is a permissive, file-level ("weak copyleft")
license: it permits combining MPL-covered code with differently
licensed code in a "Larger Work" under terms of your choosing (MPL-2.0
Section 3.3), provided any MPL-covered/derived files themselves remain
available in source form under MPL-2.0 with proper attribution. This
project already satisfies that spirit by publishing its complete source
under an OSI-approved open-source license with clear attribution here.
If you want certainty rather than a good-faith reading, consult a
lawyer before relying on this for commercial use; at minimum, retain
this notice and the attribution above if you redistribute.

`picking_winners.jl` itself is **not** covered by this repository's MIT
license grant -- it remains under its original authors' MPL-2.0 terms.
