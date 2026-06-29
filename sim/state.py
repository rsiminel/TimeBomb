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

import prompts as P   # all agent-facing copy lives in prompts.py; this module only assembles it

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

# RULES, the action instructions, and every other agent-facing string now live in
# prompts.py; this module only assembles them. Edit the wording there.


def _wire_desc(n):
  """Grammatical 'N of them is/are a WIRE/WIRES' fragment for the hand line."""
  return P.WIRE_DESC_ONE if n == 1 else P.WIRE_DESC_MANY % n


def _you(i, me):
  """The ' (you)' marker appended to a player's name when it is the viewing agent."""
  return P.YOU_MARKER if i == me else ""


def _role_line(priv, names):
  """One line that fixes the agent's identity and win condition -- folds the old separate
  role line + goal block so the win condition isn't restated (it is already in RULES)."""
  me = priv.my_index
  return (P.ROLE_BAD if priv.my_role == 1 else P.ROLE_GOOD) % (me, names[me])


def _bad_count_phrase(prior):
  keys = sorted(prior)
  if len(keys) == 1:
    tmpl = P.BAD_COUNT_SINGULAR if keys[0] == 1 else P.BAD_COUNT_PLURAL
    return tmpl % keys[0]
  parts = ", ".join(P.BAD_COUNT_PART % (k, 100 * prior[k]) for k in keys)
  return P.BAD_COUNT_UNCERTAIN % parts


def _round_hand_size(pub, r):
  """Hand size in (past or current) round ``r`` -- hands shrink by one per round."""
  return pub.hand_size + (pub.round_index - r)


def _render_round(out, r, hand_size, decls, cut_log, discussion_log, names, me, current,
                  decision=None):
  """Append one round's declarations (claimed), table talk (said), and cuts (revealed) to
  ``out`` -- the full game record, in the order they happen each round."""
  out.append((P.ROUND_HEADER_CURRENT if current else P.ROUND_HEADER) % (r + 1, hand_size))
  if decls is None or all(d is None for d in decls):
    out.append(P.DECLS_PENDING if (current and decision == "declare") else P.DECLS_NONE)
  else:
    parts = [P.DECL_PART % (names[j], _you(j, me), "?" if d is None else d)
             for j, d in enumerate(decls)]
    out.append(P.DECLS_LINE % ", ".join(parts))
  stmts = [s for s in discussion_log if s["round"] == r]
  if stmts:
    out.append(P.SAID_LINE % "; ".join(
        P.STMT_PART % (names[s["speaker"]], _you(s["speaker"], me), s["message"]) for s in stmts))
  cuts = [c for c in cut_log if c["round"] == r]
  if cuts:
    rendered = []
    for k, c in enumerate(cuts, 1):
      line = P.CUT_PART % (k, names[c["cutter"]] + _you(c["cutter"], me),
                           names[c["target"]] + _you(c["target"], me),
                           P.RESULT_WORDS.get(c["result"], c["result"]))
      if c.get("message"):                      # table talk the cutter said out loud
        line += P.CUT_PART_SAID % c["message"]
      rendered.append(line)
    out.append(P.CUTS_LINE % "; ".join(rendered))
  elif current:
    out.append(P.CUTS_NONE_YET)


def render_agent(view, decision):
  """Render an ``AgentView`` as the full prompt an agent reasons over, in four labelled
  sections (rules · role & hand · the public record · the ask) so each fact appears once.
  The record carries every round's declarations (claimed), table talk (said), and cuts
  (revealed). ``decision`` is ``"declare"``, ``"discuss"``, or ``"cut"``. Used for the
  session opener; later turns send only ``render_session_delta``."""
  pub, priv = view.public, view.private
  me, names = priv.my_index, pub.player_names
  bomb = P.BOMB_HELD if priv.i_hold_bomb else P.BOMB_NOT_HELD

  out = [P.RULES, "", P.HEADER_ROLE]
  out.append(_role_line(priv, names))
  out.append(P.HAND_LINE % (pub.hand_size, _wire_desc(priv.my_wires), bomb))
  out.append(P.TABLE_LINE % (pub.num_players, _bad_count_phrase(pub.num_bad_prior)))
  out.append("")
  out.append(P.HEADER_HISTORY)
  for r in range(pub.round_index):
    decls = pub.declaration_history[r] if r < len(pub.declaration_history) else None
    _render_round(out, r, _round_hand_size(pub, r), decls, pub.cut_log, pub.discussion_log,
                  names, me, current=False)
  _render_round(out, pub.round_index, pub.hand_size, pub.declarations, pub.cut_log,
                pub.discussion_log, names, me, current=True, decision=decision)
  out.append(P.SNAPSHOT_HIDDEN % pub.active_wires)
  out.append(P.SNAPSHOT_FACEDOWN
             % [pub.hand_size - pub.revealed[j] for j in range(pub.num_players)])

  if view.assistant_panel is not None:
    out += ["", P.ASSISTANT_HEADER, "  " + json.dumps(view.assistant_panel)]

  out += ["", P.HEADER_NOW, _decision_ask(pub, me, decision)]
  return "\n".join(out)


def _decision_ask(pub, me, decision):
  if decision == "declare":
    return P.ASK_DECLARE % pub.hand_size
  if decision == "discuss":
    return P.ASK_DISCUSS
  return P.ASK_CUT % legal_targets(pub, me)


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
    bomb = P.BOMB_HELD if priv.i_hold_bomb else P.BOMB_NOT_HELD
    out.append(P.ROUND_BANNER % (pub.round_index + 1, pub.hand_size))
    out.append(P.HAND_LINE_DELTA % (pub.hand_size, _wire_desc(priv.my_wires), bomb))
    cur["round"] = pub.round_index

  if decision in ("discuss", "cut") and cur["decls_round"] < pub.round_index:
    parts = [P.DECL_PART % (names[j], _you(j, me), "?" if d is None else d)
             for j, d in enumerate(pub.declarations)]
    out.append(P.DELTA_DECLS % ", ".join(parts))
    cur["decls_round"] = pub.round_index

  new_stmts = pub.discussion_log[cur["statements"]:]
  if new_stmts:
    out.append(P.DELTA_TALK % "; ".join(
        P.STMT_PART % (names[s["speaker"]], _you(s["speaker"], me), s["message"])
        for s in new_stmts))
    cur["statements"] = len(pub.discussion_log)

  new_cuts = pub.cut_log[cur["cuts"]:]
  if new_cuts:
    rendered = []
    for c in new_cuts:
      line = P.DELTA_CUT_PART % (names[c["cutter"]] + _you(c["cutter"], me),
                                 names[c["target"]] + _you(c["target"], me),
                                 P.RESULT_WORDS.get(c["result"], c["result"]))
      if c.get("message"):
        line += P.CUT_PART_SAID % c["message"]
      rendered.append(line)
    out.append(P.DELTA_CUTS % "; ".join(rendered))
    cur["cuts"] = len(pub.cut_log)

  out.append(P.DELTA_HIDDEN % pub.active_wires)
  out.append(P.DELTA_FACEDOWN % [pub.hand_size - pub.revealed[j] for j in range(pub.num_players)])
  if view.assistant_panel is not None:
    out.append(P.DELTA_ASSISTANT % json.dumps(view.assistant_panel))
  out.append(_decision_ask(pub, me, decision))
  return "\n".join(out), cur
