"""Arena state objects, the information firewall, and the state->text renderer.

Three disjoint information sets (SPEC.md §3): ``GroundTruth`` is engine-only,
``PublicState`` is visible to everyone, ``PrivateView`` is one agent's own hand. An
``AgentView`` is the *only* object an agent is ever handed -- it carries ``PublicState``
plus that agent's ``PrivateView`` (and, optionally, the assistant panel). A field that
lives in ``GroundTruth`` cannot reach an agent because it is not in the object the agent
receives; that is the firewall, enforced by construction rather than by discipline.

The renderer is shared by ``HumanAgent`` and ``LLMAgent`` so a human at the CLI sees
exactly what the model sees.
"""

import copy
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


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
  cut_log: list = field(default_factory=list)   # [{round,cutter,target,result}, ...]
  declaration_history: list = field(default_factory=list)  # past rounds' final declarations
  current_cutter: int = 0

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
# Event log -- the single source of replay (SPEC.md §8)
# ---------------------------------------------------------------------------

class EventLog:
  """Append-only structured log of one game. All analysis reads from this; the live
  loop computes no statistics. May contain ground truth -- it is for offline analysis,
  never handed to an agent."""

  def __init__(self):
    self.events = []

  def append(self, kind, **data):
    self.events.append({"i": len(self.events), "type": kind, **data})

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
  """Players the ``cutter`` may legally cut: anyone else with a face-down card."""
  return [j for j in range(public.num_players)
          if j != cutter and public.revealed[j] < public.hand_size]


# ---------------------------------------------------------------------------
# State -> text (shared by HumanAgent and LLMAgent)
# ---------------------------------------------------------------------------

_GOOD_GOAL = ("Good guys win when EVERY active wire is cut before time runs out. You want "
              "the cuts to find all the safe wires -- and you must never let the bomb be cut.")
_BAD_GOAL = ("Bad guys win if the bomb is ever cut, OR if time runs out before all the safe "
             "wires are found. You want cuts to waste turns and edge toward the bomb, without "
             "getting exposed as a bad guy.")


def _bad_count_phrase(prior):
  keys = sorted(prior)
  if len(keys) == 1:
    return "there %s exactly %d bad guy%s" % ("is" if keys[0] == 1 else "are",
                                              keys[0], "" if keys[0] == 1 else "s")
  parts = ", ".join("%d (%.0f%%)" % (k, 100 * prior[k]) for k in keys)
  return "the number of bad guys is uncertain: " + parts


def render_agent(view, decision):
  """Render an ``AgentView`` as the prompt text an agent reasons over. ``decision`` is
  ``"declare"`` or ``"cut"`` and selects the closing instruction."""
  pub, priv = view.public, view.private
  me = priv.my_index
  name = pub.player_names[me]
  role = "BAD GUY" if priv.my_role == 1 else "GOOD GUY"
  goal = _BAD_GOAL if priv.my_role == 1 else _GOOD_GOAL
  bomb = "You ARE holding the bomb." if priv.i_hold_bomb else "You are NOT holding the bomb."

  lines = []
  lines.append("You are Player %d (%s) in a game of Time Bomb." % (me, name))
  lines.append("Your secret role: %s. %s" % (role, goal))
  lines.append("Your hand this round: %d cards, of which %d %s an active (safe) wire. %s"
               % (pub.hand_size, priv.my_wires,
                  "is" if priv.my_wires == 1 else "are", bomb))
  lines.append("")
  lines.append("Public situation:")
  lines.append("  Players: %d | %s | Bombs in play: %d"
               % (pub.num_players, _bad_count_phrase(pub.num_bad_prior), pub.num_bom))
  lines.append("  Round %d. Each hand has %d cards this round." % (pub.round_index + 1, pub.hand_size))
  lines.append("  Active (safe) wires still hidden across all hands: %d." % pub.active_wires)
  lines.append("")
  lines.append("Declarations this round (wire counts each player CLAIMS to hold):")
  for j in range(pub.num_players):
    d = pub.declarations[j]
    shown = "(not yet declared)" if d is None else str(d)
    tag = ", you" if j == me else ""
    lines.append("  Player %d (%s%s): %s" % (j, pub.player_names[j], tag, shown))

  this_round_cuts = [c for c in pub.cut_log if c["round"] == pub.round_index]
  lines.append("")
  if this_round_cuts:
    lines.append("Cuts so far this round:")
    for c in this_round_cuts:
      lines.append("  - Player %d cut Player %d -> %s"
                   % (c["cutter"], c["target"], c["result"]))
  else:
    lines.append("No cuts yet this round.")

  facedown = [pub.hand_size - pub.revealed[j] for j in range(pub.num_players)]
  lines.append("Face-down cards remaining per player: %s" % facedown)

  if view.assistant_panel is not None:
    lines.append("")
    lines.append("Assistant readout (computed from public info only):")
    lines.append("  " + json.dumps(view.assistant_panel))

  lines.append("")
  if decision == "declare":
    lines.append("It is your turn to DECLARE. State a wire count from 0 to %d. You may "
                 "tell the truth or bluff." % pub.hand_size)
  else:
    legal = legal_targets(pub, me)
    lines.append("It is your turn to CUT -- you hold the wire-cutters. Choose one OTHER "
                 "player's face-down card to cut.")
    lines.append("Legal targets (player indices): %s" % legal)
  return "\n".join(lines)
