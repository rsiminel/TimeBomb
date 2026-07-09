"""The agent interface. One small contract shared by every tier and by the human CLI,
so 'attach the LLM' is a drop-in against something already verified (sim/SPEC.md §5).

Promoted from ``sim/agents/base.py`` verbatim, plus one addition: ``claim`` -- a no-op
by default, so every existing agent is unaffected. Hosts that use structured claims
(the web game) call it at the agent's decision points."""


class Agent:
  name = "agent"
  last_call_failed = False   # True right after a decision that exhausted its retries
                              # (LLMAgent); distinct from returning malformed-but-present
                              # output, which the engine's own fallback already handles.

  def declare(self, view):
    """Return a wire count in [0, view.public.hand_size]. Free to bluff."""
    raise NotImplementedError

  def choose_cut(self, view):
    """Return the index of the player to cut. Must be a legal target (someone else with
    a face-down card). The engine validates and falls back on illegal returns."""
    raise NotImplementedError

  def discuss(self, view):
    """Return a short public statement to the whole table (a claim, a read, an accusation,
    a defense, or a bluff), or ``None`` to stay silent. Called once per player each round,
    after declarations and before any cut, sequentially -- so a speaker hears those before
    it. No-op by default; only talking agents (the LLM, the human) override it."""
    return None

  def claim(self, view):
    """Return a Claim dict from the fixed menu (``{kind, target}`` -- see
    tbgame/state.py) or ``None``. Called by hosts that use structured claims (the web
    game) at the agent's declare/cut decision points. No-op by default."""
    return None

  def observe(self, event):
    """Called for every public event (any declaration, any cut). No-op by default;
    stateful agents override it. The engine passes only public event data here."""
    pass

  def describe(self):
    """A JSON-able descriptor of this agent's configuration. The engine logs it into
    every game (and run.py into the run manifest) so results stay analysable later --
    you can always recover which model / instructions produced a transcript. Subclasses
    extend it with their own knobs."""
    return {"type": type(self).__name__, "name": getattr(self, "name", "agent")}

  def to_state(self):
    """A JSON-able snapshot of whatever this agent needs to resume mid-game (e.g. a live
    session id). No-op by default; only stateful agents override it."""
    return {}

  def load_state(self, state):
    """Restore a snapshot produced by ``to_state()``. No-op by default."""
    pass
