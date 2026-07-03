"""Pytest bootstrap: put the ``timebomb/`` source root on ``sys.path``.

The backend modules live in ``timebomb/`` and import each other by bare name
(``import UsefulFunctions as uf``), so that directory must be importable. It is a
source root, not an installed package — adding it here keeps the bare imports working
without an ``__init__.py`` or a packaging step.
"""
import sys
from pathlib import Path

_root = Path(__file__).parent
# Source roots on sys.path (same bare-import convention for the sim/ sub-project).
for _src in ("timebomb", "sim", "sim/agents"):
  sys.path.insert(0, str(_root / _src))

# tbgame/ (specs/002-host-local-game) is a real package (tbgame/__init__.py), not a bare
# source root: sim/state.py, sim/engine.py, sim/agents/base.py share bare filenames with
# their tbgame/ counterparts (both promoted from/shimming to the same names), so bare
# `import state` can only ever resolve to one of them per process. Dotted access
# (`tbgame.state`, `tbgame.engine.TableGame`, per contracts/engine.md) sidesteps the
# collision entirely. The repo root just needs to be on sys.path for that.
sys.path.insert(0, str(_root))
