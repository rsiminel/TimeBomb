"""Shared game state objects and the information firewall (specs/002-host-local-game).

Promoted from ``sim/state.py`` (the state->text renderer and agent-facing prompt copy
stay behind there -- those are arena/LLM presentation concerns, not rules). Three
disjoint information sets: ``GroundTruth`` is engine-only, ``PublicState`` is visible
to everyone, ``PrivateView`` is one seat's own hand. An ``AgentView`` is the only
object an agent is ever handed -- it carries ``PublicState`` plus that agent's
``PrivateView`` (and, optionally, the assistant panel). A field that lives in
``GroundTruth`` cannot reach an agent because it is not in the object the agent
receives; that is the firewall, enforced by construction rather than by discipline.
"""

import copy
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

# Bumped from sim's SCHEMA_VERSION (1): the hosted-game event log adds claim_log,
# phase, pending and the `claim` event kind (data-model.md). Independent counter --
# sim's own logs keep their own version and are never rewritten.
SCHEMA_VERSION = 2


# ---------------------------------------------------------------------------
# The three information sets
# ---------------------------------------------------------------------------

@dataclass
class GroundTruth:
  """Engine-only hidden truth. Never placed in an ``AgentView``."""
  roles: list             # roles[i] == 1 iff player i is a bad guy (fixed all game)
  wires: list             # wires[i]: active wires actually in i's hand THIS round
  bombs: list             # bombs[i] == 1 iff i holds the bomb THIS round
  num_bad: int            # the realized bad count (sampled from the prior)
  seed: Optional[int] = None


@dataclass
class Pending:
  """Who owes what decision right now. ``None`` on a finished game."""
  kind: str                # "declare" | "cut"
  seats: list              # declare: every seat without a declaration this round
                           # (simultaneous); cut: exactly [current_cutter]


@dataclass
class PublicState:
  """Everything visible at the table. Snapshotted into every agent view."""
  num_players: int
  num_bad_prior: dict      # {num_bad: P} -- the publicly known role-count prior
  num_bom: int
  player_names: list
  round_index: int         # 0-based round number
  hand_size: int           # cards per hand this round (H)
  round_start_active: int  # active wires at round start (the ProbDeclaration total)
  active_wires: int        # active wires still hidden right now (the ProbCut total)
  declarations: list       # this round; entry is None until that player has declared
  revealed: list           # cards already cut from each hand this round
  found: list              # active wires found in each hand this round
  cut_log: list = field(default_factory=list)   # [{round,cutter,target,result,message}, ...]
  discussion_log: list = field(default_factory=list)  # sim-only free-text table talk
  claim_log: list = field(default_factory=list)       # [{round,speaker,kind,target}, ...]
  declaration_history: list = field(default_factory=list)  # past rounds' final declarations
  current_cutter: int = 0
  phase: str = "awaiting_declarations"   # "awaiting_declarations" | "awaiting_cut" | "finished"
  pending: Optional[Pending] = None

  def snapshot(self):
    """A deep copy, so an agent can never mutate the live table state."""
    return copy.deepcopy(self)


@dataclass
class PrivateView:
  """One agent's private knowledge: its own role and its own hand this round."""
  my_index: int
  my_role: int             # 0 good, 1 bad
  my_wires: int            # active wires in my own hand this round
  i_hold_bomb: bool


@dataclass
class AgentView:
  """The only object an agent is handed. Firewall boundary."""
  public: PublicState
  private: PrivateView
  assistant_panel: Optional[dict] = None   # present only if this agent is shown the panel


# ---------------------------------------------------------------------------
# Event log -- the single source of replay/save/resume
# ---------------------------------------------------------------------------

class EventLog:
  """Append-only structured log of one game. All analysis reads from this; the live
  loop computes no statistics. May contain ground truth -- it is for offline analysis
  and resume, never handed to an agent."""

  def __init__(self, events=None):
    self.events = events if events is not None else []

  def append(self, event_type, **data):
    event = {"i": len(self.events), "type": event_type, **data}
    self.events.append(event)
    return event

  def to_jsonl(self, path):
    with open(path, "w") as f:
      for e in self.events:
        f.write(json.dumps(e, default=_jsonable) + "\n")


def _jsonable(o):
  if hasattr(o, "tolist"):       # numpy arrays/scalars
    return o.tolist()
  try:
    return asdict(o)
  except TypeError:
    return str(o)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def legal_targets(public, cutter):
  """Players ``cutter`` may legally cut: anyone else with a face-down card."""
  return [j for j in range(public.num_players)
          if j != cutter and public.revealed[j] < public.hand_size]
