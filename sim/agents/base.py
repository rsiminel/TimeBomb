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

  def observe(self, event):
    """Called for every public event (any declaration, any cut). No-op by default;
    stateful agents override it. The engine passes only public event data here."""
    pass
