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

# Bumped only on a breaking change to the on-disk log / manifest format. See sim/LOGS.md.
SCHEMA_VERSION = 1


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

RULES = (
    "TIME BOMB — HOW TO WIN (read carefully, this is often misremembered):\n"
    "- Two secret teams: GOOD guys and BAD guys. Your role is fixed for the whole game.\n"
    "- Hidden in the players' face-down cards are SAFE WIRES, some duds, and exactly ONE\n"
    "  BOMB (held by one player, reshuffled and re-dealt every round).\n"
    "- On a turn, whoever holds the wire-cutters cuts one OTHER player's face-down card,\n"
    "  revealing a safe wire, a dud, or the bomb. Whoever is cut takes the cutters next.\n"
    "- GOOD guys WIN by cutting ALL the safe wires before time runs out.\n"
    "- BAD guys WIN if the BOMB is ever cut (game ends INSTANTLY in their favour), OR if\n"
    "  time runs out before every safe wire is found.\n"
    "- So BAD guys WANT the bomb cut and want cuts wasted on duds; GOOD guys want to find\n"
    "  the safe wires and must NOT cut the bomb. (Bad guys do NOT 'protect' the bomb.)\n"
    "- Hands shrink by one card each round (5 down to 1). A declaration is a player's\n"
    "  claim about how many safe wires they hold this round, and may be a lie.")

_GOOD_GOAL = ("As a GOOD guy you win when every safe wire is cut in time; help the cuts "
              "find wires, and never let the bomb be cut.")
_BAD_GOAL = ("As a BAD guy you win if the bomb is cut OR time runs out; you want cuts wasted "
             "on duds and the bomb eventually cut, without being exposed.")

_RESULT = {"active wire": "a SAFE WIRE", "blank/inactive": "a dud", "BOMB": "THE BOMB"}


def _bad_count_phrase(prior):
  keys = sorted(prior)
  if len(keys) == 1:
    return "there %s exactly %d bad guy%s" % ("is" if keys[0] == 1 else "are",
                                              keys[0], "" if keys[0] == 1 else "s")
  parts = ", ".join("%d (%.0f%%)" % (k, 100 * prior[k]) for k in keys)
  return "the number of bad guys is uncertain: " + parts


def _round_hand_size(pub, r):
  """Hand size in (past or current) round ``r`` -- hands shrink by one per round."""
  return pub.hand_size + (pub.round_index - r)


def _render_round(out, r, hand_size, decls, cut_log, names, me, current, decision=None):
  """Append one round's declarations + cuts to ``out`` (full game history)."""
  out.append(("THIS ROUND — Round %d (hand size %d):" if current
              else "Round %d (hand size %d):") % (r + 1, hand_size))
  if decls is None or all(d is None for d in decls):
    if current and decision == "declare":
      out.append("  Declarations: being made now, simultaneously — you don't see others' yet.")
    else:
      out.append("  Declarations: (none recorded)")
  else:
    parts = ["%s%s=%s" % (names[j], " (you)" if j == me else "", "?" if d is None else d)
             for j, d in enumerate(decls)]
    out.append("  Declared wire counts — " + ", ".join(parts))
  cuts = [c for c in cut_log if c["round"] == r]
  if cuts:
    rendered = []
    for k, c in enumerate(cuts, 1):
      line = "%d. %s cut %s → %s" % (
          k, names[c["cutter"]] + (" (you)" if c["cutter"] == me else ""),
          names[c["target"]] + (" (you)" if c["target"] == me else ""),
          _RESULT.get(c["result"], c["result"]))
      if c.get("message"):                      # table talk the cutter said out loud
        line += ' — said: "%s"' % c["message"]
      rendered.append(line)
    out.append("  Cuts — " + "; ".join(rendered))
  elif current:
    out.append("  Cuts so far this round: none yet.")


def render_agent(view, decision):
  """Render an ``AgentView`` as the prompt text an agent reasons over. Includes the rules,
  the agent's private hand/role, and the FULL multi-round public history (every round's
  declarations and who cut whom, with results). ``decision`` is ``"declare"`` or ``"cut"``."""
  pub, priv = view.public, view.private
  me = priv.my_index
  names = pub.player_names
  role = "BAD GUY" if priv.my_role == 1 else "GOOD GUY"
  goal = _BAD_GOAL if priv.my_role == 1 else _GOOD_GOAL
  bomb = ("You ARE holding the bomb this round." if priv.i_hold_bomb
          else "You are NOT holding the bomb this round.")

  out = [RULES, ""]
  out.append("YOU are Player %d (%s). Your secret role: %s." % (me, names[me], role))
  out.append(goal)
  out.append("Your hand this round: %d cards, %d of them %s a safe wire. %s"
             % (pub.hand_size, priv.my_wires, "is" if priv.my_wires == 1 else "are", bomb))
  out.append("Table: %d players | %s | %d bomb in play."
             % (pub.num_players, _bad_count_phrase(pub.num_bad_prior), pub.num_bom))
  out.append("")
  out.append("GAME HISTORY (everything public, all rounds):")
  for r in range(pub.round_index):
    decls = pub.declaration_history[r] if r < len(pub.declaration_history) else None
    _render_round(out, r, _round_hand_size(pub, r), decls, pub.cut_log, names, me, current=False)
  _render_round(out, pub.round_index, pub.hand_size, pub.declarations, pub.cut_log,
                names, me, current=True, decision=decision)
  out.append("  Safe wires still hidden across all hands: %d." % pub.active_wires)
  out.append("  Face-down cards left per player this round: %s"
             % [pub.hand_size - pub.revealed[j] for j in range(pub.num_players)])

  if view.assistant_panel is not None:
    out += ["", "Assistant readout (computed from public info only):",
            "  " + json.dumps(view.assistant_panel)]

  out.append("")
  if decision == "declare":
    out.append("YOUR TURN TO DECLARE. Announce a safe-wire count from 0 to %d — the truth, "
               "or a bluff that serves your team." % pub.hand_size)
  else:
    legal = legal_targets(pub, me)
    out.append("YOUR TURN TO CUT — you hold the wire-cutters. Cut one OTHER player's "
               "face-down card. Legal targets (player indices): %s" % legal)
  return "\n".join(out)
