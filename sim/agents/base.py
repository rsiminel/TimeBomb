"""The agent interface. One small contract shared by every tier and by the human CLI,
so 'attach the LLM' is a drop-in against something already verified (SPEC.md §5)."""


class Agent:
  name = "agent"

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
