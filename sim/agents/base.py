"""Agent interface re-export shim (specs/002-host-local-game).

Promoted to ``tbgame/agents/base.py`` (same methods, plus a no-op ``claim`` hook that
does not affect existing agents). Re-exported here so existing imports
(``from base import Agent``) keep working unchanged.
"""

from tbgame.agents.base import Agent
