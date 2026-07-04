"""Arena state re-export shim + the state->text renderer (specs/002-host-local-game).

The state dataclasses, event log, and firewall helper were promoted to
``tbgame/state.py`` (the shared engine's state, reused by the web game). This module
re-exports them so existing imports (``from state import GroundTruth, ...``) keep
working unchanged, and keeps the renderer and prompt assembly here -- those are
arena/LLM presentation concerns, not shared rules state.

sim's own on-disk log schema version is pinned here, independent of
``tbgame.state.SCHEMA_VERSION`` (bumped separately for the web game's log format,
sim/LOGS.md is untouched).
"""

import json

from tbgame.state import (GroundTruth, PublicState, PrivateView, AgentView, EventLog,
                          legal_targets)

import prompts as P   # all agent-facing copy lives in prompts.py; this module only assembles it

# Bumped only on a breaking change to the on-disk log format. See sim/LOGS.md.
# Independent of tbgame.state.SCHEMA_VERSION -- do not conflate the two (CLAUDE.md).
SCHEMA_VERSION = 1


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
  return (P.ROLE_BAD if priv.my_role == 1 else P.ROLE_GOOD) % names[priv.my_index]


def _hands_snapshot(pub, me):
  """Per-hand public bookkeeping, by name: wires already found in each hand this round vs
  cards still face-down. Pure arithmetic over the public record (no hidden info)."""
  names = pub.player_names
  return "; ".join(P.HAND_PART % (names[j], _you(j, me), pub.found[j],
                                  pub.hand_size - pub.revealed[j])
                   for j in range(pub.num_players))


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

  out = [P.RULES_OPEN, P.RULES, P.RULES_CLOSE, "", P.ROLE_OPEN]
  out.append(_role_line(priv, names))
  out.append(P.HAND_LINE % (pub.hand_size, _wire_desc(priv.my_wires), bomb))
  roster = ", ".join(names[j] + _you(j, me) for j in range(pub.num_players))
  out.append(P.TABLE_LINE % (pub.num_players, roster, _bad_count_phrase(pub.num_bad_prior)))
  out.append(P.ROLE_CLOSE)
  out.append("")
  out.append(P.HISTORY_OPEN)
  for r in range(pub.round_index):
    decls = pub.declaration_history[r] if r < len(pub.declaration_history) else None
    _render_round(out, r, _round_hand_size(pub, r), decls, pub.cut_log, pub.discussion_log,
                  names, me, current=False)
  _render_round(out, pub.round_index, pub.hand_size, pub.declarations, pub.cut_log,
                pub.discussion_log, names, me, current=True, decision=decision)
  out.append(P.SNAPSHOT_HIDDEN % pub.active_wires)
  out.append(P.SNAPSHOT_HANDS % _hands_snapshot(pub, me))
  out.append(P.HISTORY_CLOSE)

  if view.assistant_panel is not None:
    out += ["", P.ASSISTANT_OPEN, "  " + json.dumps(view.assistant_panel), P.ASSISTANT_CLOSE]

  out += ["", P.NOW_OPEN, _decision_ask(pub, me, decision), P.NOW_CLOSE]
  return "\n".join(out)


def _decision_ask(pub, me, decision):
  if decision == "declare":
    return P.ASK_DECLARE % pub.hand_size
  if decision == "discuss":
    return P.ASK_DISCUSS
  return P.ASK_CUT % ", ".join(pub.player_names[t] for t in legal_targets(pub, me))


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
  out.append(P.DELTA_HANDS % _hands_snapshot(pub, me))
  if view.assistant_panel is not None:
    out.append(P.DELTA_ASSISTANT % json.dumps(view.assistant_panel))
  out.append(_decision_ask(pub, me, decision))
  return "\n".join(out), cur
