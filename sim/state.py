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
  cut_log: list = field(default_factory=list)   # [{round,cutter,target,result,message}, ...]
  discussion_log: list = field(default_factory=list)  # [{round,speaker,message}, ...] table talk
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
    "- When a face-down card is cut it turns out to be exactly one of three things:\n"
    "    - a WIRE — one of the cards the GOOD guys must cut to win;\n"
    "    - a DUD — a harmless blank; cutting it just wastes the cut;\n"
    "    - the BOMB — the single bomb card.\n"
    "  A DUD and the BOMB are NOT wires. 'Cutting a wire' always means a winning card.\n"
    "- Every round, ALL cards are gathered up, reshuffled, and dealt out fresh. So how many\n"
    "  wires you hold, and who holds the bomb, change each round and are independent from\n"
    "  one round to the next.\n"
    "- On a turn, whoever holds the wire-cutters cuts one OTHER player's face-down card,\n"
    "  revealing a wire, a dud, or the bomb. Whoever is cut takes the cutters next.\n"
    "- GOOD guys WIN by cutting ALL the wires before time runs out.\n"
    "- BAD guys WIN if the BOMB is ever cut (game ends INSTANTLY in their favour), OR if\n"
    "  time runs out before every wire is found.\n"
    "- So BAD guys WANT the bomb cut and want cuts wasted on duds; GOOD guys want to find\n"
    "  the wires and must NOT cut the bomb. (Bad guys do NOT 'protect' the bomb.)\n"
    "- Hands shrink by one card each round (5 down to 1). A declaration is a player's\n"
    "  claim about how many wires they hold this round, and may be a lie.")

_RESULT = {"wire": "a WIRE", "dud": "a dud", "bomb": "THE BOMB"}


def _wire_desc(n):
  """Grammatical 'N of them is/are a WIRE/WIRES' fragment for the hand line."""
  return "1 of them is a WIRE" if n == 1 else "%d of them are WIRES" % n


def _role_line(priv, names):
  """One line that fixes the agent's identity and win condition -- folds the old separate
  role line + goal block so the win condition isn't restated (it is already in RULES)."""
  me = priv.my_index
  if priv.my_role == 1:
    return ("You are Player %d (%s) — a BAD GUY. You win if the BOMB is cut, or if time "
            "runs out with wires still hidden." % (me, names[me]))
  return ("You are Player %d (%s) — a GOOD GUY. You win when every WIRE is cut, and you "
          "must never cut the bomb." % (me, names[me]))


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


def _render_round(out, r, hand_size, decls, cut_log, discussion_log, names, me, current,
                  decision=None):
  """Append one round's declarations (claimed), table talk (said), and cuts (revealed) to
  ``out`` -- the full game record, in the order they happen each round."""
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
  stmts = [s for s in discussion_log if s["round"] == r]
  if stmts:
    out.append("  Said — " + "; ".join(
        '%s%s: "%s"' % (names[s["speaker"]], " (you)" if s["speaker"] == me else "", s["message"])
        for s in stmts))
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
  """Render an ``AgentView`` as the full prompt an agent reasons over, in four labelled
  sections (rules · role & hand · the public record · the ask) so each fact appears once.
  The record carries every round's declarations (claimed), table talk (said), and cuts
  (revealed). ``decision`` is ``"declare"``, ``"discuss"``, or ``"cut"``. Used for the
  session opener; later turns send only ``render_session_delta``."""
  pub, priv = view.public, view.private
  me, names = priv.my_index, pub.player_names
  bomb = ("You ARE holding the bomb this round." if priv.i_hold_bomb
          else "You are NOT holding the bomb this round.")

  out = [RULES, "", "YOUR ROLE & HAND"]
  out.append(_role_line(priv, names))
  out.append("Your hand this round: %d cards, %s. %s"
             % (pub.hand_size, _wire_desc(priv.my_wires), bomb))
  out.append("Table: %d players; %s." % (pub.num_players, _bad_count_phrase(pub.num_bad_prior)))
  out.append("")
  out.append("GAME SO FAR (public record — what players CLAIMED, SAID, and what cuts REVEALED):")
  for r in range(pub.round_index):
    decls = pub.declaration_history[r] if r < len(pub.declaration_history) else None
    _render_round(out, r, _round_hand_size(pub, r), decls, pub.cut_log, pub.discussion_log,
                  names, me, current=False)
  _render_round(out, pub.round_index, pub.hand_size, pub.declarations, pub.cut_log,
                pub.discussion_log, names, me, current=True, decision=decision)
  out.append("  Wires still hidden across all hands: %d." % pub.active_wires)
  out.append("  Face-down cards left per player this round: %s"
             % [pub.hand_size - pub.revealed[j] for j in range(pub.num_players)])

  if view.assistant_panel is not None:
    out += ["", "Assistant readout (computed from public info only):",
            "  " + json.dumps(view.assistant_panel)]

  out += ["", "NOW", _decision_ask(pub, me, decision)]
  return "\n".join(out)


def _decision_ask(pub, me, decision):
  if decision == "declare":
    return ("YOUR TURN TO DECLARE. Announce a wire count from 0 to %d — the truth, "
            "or a bluff that serves your team." % pub.hand_size)
  if decision == "discuss":
    return "YOUR TURN TO SPEAK to the whole table, before anyone cuts this round."
  return ("YOUR TURN TO CUT — you hold the wire-cutters. Cut one OTHER player's "
          "face-down card. Legal targets (player indices): %s" % legal_targets(pub, me))


# ---------------------------------------------------------------------------
# Incremental rendering for a resumed `claude -p` session (LLMAgent session mode)
# ---------------------------------------------------------------------------
# When a player keeps one persistent session per model, the model already holds everything
# up to its last turn in its own context. We send only what is NEW since then -- the first
# call seeds full context via ``render_agent``; every later call sends a small delta, and
# the unchanged prefix is served from the prompt cache. A ``cursor`` (see below) tracks how
# far each session has been narrated. The firewall is unchanged: a session is fed only its
# own player's legal views.

def new_session_cursor():
  """Fresh cursor for a session that has narrated nothing yet."""
  return {"cuts": 0, "statements": 0, "round": -1, "decls_round": -2}


def cursor_after_opener(view, decision):
  """The cursor state implied by a full ``render_agent`` opener: it has shown every cut and
  statement so far, the current round + hand, and (for a sighted turn -- discuss or cut)
  this round's declarations."""
  pub = view.public
  return {"cuts": len(pub.cut_log), "statements": len(pub.discussion_log),
          "round": pub.round_index,
          "decls_round": pub.round_index if decision in ("discuss", "cut")
                         else pub.round_index - 1}


def render_session_delta(view, decision, cursor):
  """Narrate only what changed since ``cursor`` and return ``(text, new_cursor)``. Covers a
  new round (re-deal + the player's fresh hand), this round's declarations the first time a
  sighted turn needs them, any table talk and cuts since the player's last turn (in that
  chronological order -- declare, then discuss, then cut), the hidden-wire / face-down
  snapshot, and the decision ask."""
  pub, priv = view.public, view.private
  names, me = pub.player_names, priv.my_index
  cur = dict(cursor)
  out = []

  if cur["round"] != pub.round_index:
    bomb = ("You ARE holding the bomb this round." if priv.i_hold_bomb
            else "You are NOT holding the bomb this round.")
    out.append("--- Round %d begins (hand size %d). All cards were collected, reshuffled, "
               "and dealt out fresh. ---" % (pub.round_index + 1, pub.hand_size))
    out.append("Your hand now: %d cards, %s. %s"
               % (pub.hand_size, _wire_desc(priv.my_wires), bomb))
    cur["round"] = pub.round_index

  if decision in ("discuss", "cut") and cur["decls_round"] < pub.round_index:
    parts = ["%s%s=%s" % (names[j], " (you)" if j == me else "", "?" if d is None else d)
             for j, d in enumerate(pub.declarations)]
    out.append("Declared wire counts this round — " + ", ".join(parts))
    cur["decls_round"] = pub.round_index

  new_stmts = pub.discussion_log[cur["statements"]:]
  if new_stmts:
    out.append("Table talk since your last turn — " + "; ".join(
        '%s%s: "%s"' % (names[s["speaker"]], " (you)" if s["speaker"] == me else "", s["message"])
        for s in new_stmts))
    cur["statements"] = len(pub.discussion_log)

  new_cuts = pub.cut_log[cur["cuts"]:]
  if new_cuts:
    rendered = []
    for c in new_cuts:
      line = "%s cut %s → %s" % (
          names[c["cutter"]] + (" (you)" if c["cutter"] == me else ""),
          names[c["target"]] + (" (you)" if c["target"] == me else ""),
          _RESULT.get(c["result"], c["result"]))
      if c.get("message"):
        line += ' — said: "%s"' % c["message"]
      rendered.append(line)
    out.append("Cuts since your last turn — " + "; ".join(rendered))
    cur["cuts"] = len(pub.cut_log)

  out.append("Wires still hidden across all hands: %d." % pub.active_wires)
  out.append("Face-down cards left per player: %s"
             % [pub.hand_size - pub.revealed[j] for j in range(pub.num_players)])
  if view.assistant_panel is not None:
    out.append("Assistant readout (public info only): " + json.dumps(view.assistant_panel))
  out.append(_decision_ask(pub, me, decision))
  return "\n".join(out), cur
