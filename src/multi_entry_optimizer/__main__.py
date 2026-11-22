"""Enables `python -m multi_entry_optimizer ...` as a CLI entry point,
equivalent to the `multi-entry-optimizer` console script.
"""

import sys

from .lineups import main

if __name__ == "__main__":
    sys.exit(main())
