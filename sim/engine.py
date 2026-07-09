"""Arena engine re-export shim (specs/002-host-local-game).

The referee was promoted to ``tbgame/driver.py::Engine`` (a blocking arena API kept
byte-compatible over the shared rules engine). This module re-exports it so existing
imports (``from engine import Engine, WIRE, BLANK, BOMB, ...``) keep working unchanged.
``tests/test_arena_engine.py`` -- the frozen root suite's regression harness for the
promotion -- imports through this shim and must pass unmodified.
"""

from tbgame.driver import (Engine, EngineHalted, WIRE, BLANK, BOMB, _validate_declaration,
                           _validate_target, log_cuts, log_statements)
