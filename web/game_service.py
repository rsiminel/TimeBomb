"""The one active hosted game: seat locks and the AI turn worker (data-model.md
ActiveGame). Transport-adjacent bookkeeping only -- every rule decision comes from
``tbgame.engine.TableGame``; every probability (opt-in panel) from ``panel_bridge.py``
-> ``replay.py`` -> ``General.py``. This module holds no math and no rule logic of its
own (constitution I).
"""
import datetime
import json
import re
import threading
import time
from pathlib import Path

import panel_bridge
import replay
from tbgame.engine import (TableGame, SetupError, IllegalIntent, SchemaError,
                           public_event)
from tbgame.state import AgentView, legal_targets
from tbgame.agents.solver import SolverBot

AI_POLL_INTERVAL_S = 0.2

# Manual named saves (data-model.md SaveFile), git-ignored local storage.
SAVES_DIR = Path(__file__).resolve().parent / "saves"
_SAVE_NAME_RE = re.compile(r"^[\w][\w \-]{0,39}$")   # filesystem-safe, no traversal


class ActiveGameError(Exception):
  """A game-lifecycle violation. ``status`` is the HTTP status the route returns;
  ``str(self)`` is the reason forwarded verbatim (contracts/api.md)."""

  def __init__(self, reason, status):
    super().__init__(reason)
    self.reason = reason
    self.status = status


class _ActiveGame:
  def __init__(self, game, setup, panel_allowed, num_bad_override):
    self.game = game
    self.setup = setup          # as received; persisted verbatim into SaveFiles
    self.version = 1
    self.unlocked_seat = None
    self.thinking = set()
    self.panel_allowed = panel_allowed
    # The *declared* setup override (None = official deal); what the panel bridge
    # feeds replay.py as numBadOverride. Never the sampled truth.
    self.num_bad_override = num_bad_override


_lock = threading.RLock()
_active = None
_generation = 0


def _ai_agent_for(occupant):
  if occupant == "solver_bot":
    return SolverBot()
  return None   # human, or an llm:* seat not yet wired (T037-039)


def create_game(setup):
  global _active, _generation
  with _lock:
    if _active is not None:
      raise ActiveGameError("a game is already active", 409)
    try:
      game = TableGame(setup)
    except SetupError as err:
      raise ActiveGameError(err.reason, 422) from err
    _active = _new_active(game, setup)
    _generation += 1
    generation = _generation
    version = _active.version
  _start_worker(generation)
  return version


def _new_active(game, setup):
  role_deal = setup.get("role_deal", "official")
  return _ActiveGame(game, setup,
                     panel_allowed=bool(setup.get("panel_allowed", False)),
                     num_bad_override=None if role_deal == "official" else role_deal)


def abandon_game():
  global _active, _generation
  with _lock:
    if _active is None:
      raise ActiveGameError("no active game", 404)
    _active = None
    _generation += 1


def reset_for_tests():
  """Test-only: clear the active game and bump the generation so any lingering AI
  worker thread from a previous test notices and exits. Never called by app.py."""
  global _active, _generation
  with _lock:
    _active = None
    _generation += 1


def _require_active():
  if _active is None:
    raise ActiveGameError("no active game", 404)
  return _active


def table_view():
  with _lock:
    return _table_view_locked(_require_active())


def _table_view_locked(active):
  pub = active.game.public_state()
  pending = pub.pending
  view = {
      "version": active.version,
      "numPlayers": pub.num_players,
      "playerNames": pub.player_names,
      "roundIndex": pub.round_index,
      "handSize": pub.hand_size,
      "activeWires": pub.active_wires,
      "declarations": pub.declarations,
      "revealed": pub.revealed,
      "found": pub.found,
      "cutLog": pub.cut_log,
      "claimLog": pub.claim_log,
      "declarationHistory": pub.declaration_history,
      "currentCutter": pub.current_cutter,
      "phase": pub.phase,
      "pending": (None if pending is None
                 else {"kind": pending.kind, "seats": list(pending.seats)}),
      # The engine's rule, surfaced for the cut picker (constitution I: no rule
      # logic client-side; the engine re-validates on submit regardless).
      "legalTargets": (legal_targets(pub, pending.seats[0])
                       if pending is not None and pending.kind == "cut" else None),
      "occupants": list(active.game.occupants),
      "thinking": sorted(active.thinking),
      "panelAllowed": active.panel_allowed,
      "unlockedSeat": active.unlocked_seat,
  }
  if pub.phase == "finished" or active.game.exhibition:
    view["reveal"] = active.game.reveal()
  return view


def panel():
  """The opt-in public-info panel: the v1 replay pipeline over the hosted game's
  public record (FR-019/020/021). Numbers are replay.py's verbatim."""
  with _lock:
    active = _require_active()
    if not active.panel_allowed:
      raise ActiveGameError("this game did not opt in to the stats panel", 403)
    record = panel_bridge.game_record(active.game.public_state(),
                                      active.num_bad_override)
  return replay.replay_record(record)


def unlock(seat, version):
  with _lock:
    active = _require_active()
    if version != active.version:
      raise ActiveGameError("version mismatch", 409)
    occupants = active.game.occupants
    if (not isinstance(seat, int) or isinstance(seat, bool)
        or not 0 <= seat < len(occupants) or occupants[seat] != "human"):
      raise ActiveGameError("seat is not human", 403)
    if active.unlocked_seat is not None and active.unlocked_seat != seat:
      raise ActiveGameError("another seat is unlocked", 403)
    active.unlocked_seat = seat
    priv = active.game.private_view(seat)
    return {"myIndex": priv.my_index, "myRole": priv.my_role, "myWires": priv.my_wires,
            "iHoldBomb": priv.i_hold_bomb}


def lock():
  with _lock:
    active = _require_active()
    active.unlocked_seat = None


def submit_intent(seat, kind, value, claim, version):
  with _lock:
    active = _require_active()
    if version != active.version:
      raise ActiveGameError("version mismatch", 409)
    try:
      events = active.game.submit({"seat": seat, "kind": kind, "value": value, "claim": claim})
    except IllegalIntent as err:
      raise ActiveGameError(err.reason, 422) from err
    active.version += 1
    active.unlocked_seat = None
    # A round-crossing cut appends the next round_start, which records the fresh
    # deal for replay -- redact it (and declaration truth) before it leaves (FR-008).
    return active.version, [public_event(e) for e in events]


# ---------------------------------------------------------------------------
# Manual named saves (FR-023): write/list/load/delete web/saves/<name>.json.
# The event log alone reduces back to an identical TableGame (from_events).
# ---------------------------------------------------------------------------

def _save_path(name):
  if not isinstance(name, str) or not _SAVE_NAME_RE.match(name):
    raise ActiveGameError("save name must be 1-40 letters, digits, spaces or dashes", 422)
  return SAVES_DIR / (name + ".json")


def save_game(name):
  path = _save_path(name)
  with _lock:
    active = _require_active()
    if path.exists():
      raise ActiveGameError("a save named %r already exists" % name, 409)
    data = {
        "schema_version": active.game.events[0]["schema_version"],
        "name": name,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "setup": active.setup,
        "events": active.game.events,
        # Convenience cache for the resume list; the events stay authoritative.
        "phase": active.game.public_state().phase,
    }
    SAVES_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(data))


def list_saves():
  out = []
  for path in sorted(SAVES_DIR.glob("*.json")):
    try:
      data = json.loads(path.read_text())
      out.append({"name": data["name"], "created": data["created"],
                  "seats": [s["name"] for s in data["setup"]["seats"]],
                  "phase": data.get("phase")})
    except (ValueError, KeyError, TypeError):
      continue   # an unreadable file never breaks the listing
  return out


def resume_game(name):
  global _active, _generation
  path = _save_path(name)
  if not path.exists():
    raise ActiveGameError("no save named %r" % name, 404)
  with _lock:
    if _active is not None:
      raise ActiveGameError("a game is already active", 409)
    data = json.loads(path.read_text())
    try:
      game = TableGame.from_events(data["events"])
    except SchemaError as err:
      raise ActiveGameError(err.reason, 422) from err
    _active = _new_active(game, data["setup"])
    _generation += 1
    generation = _generation
    version = _active.version
  _start_worker(generation)
  return version


def delete_save(name):
  path = _save_path(name)
  if not path.exists():
    raise ActiveGameError("no save named %r" % name, 404)
  path.unlink()


# ---------------------------------------------------------------------------
# AI turn worker -- one background thread per created game (identified by
# ``generation``, so a replaced/abandoned game's worker notices and stops).
# ---------------------------------------------------------------------------

def _start_worker(generation):
  threading.Thread(target=_worker_loop, args=(generation,), daemon=True).start()


def _worker_loop(generation):
  while True:
    with _lock:
      if _generation != generation or _active is None or _active.game.finished:
        return
      active = _active
      pending = active.game.pending
      occupants = active.game.occupants
      ai_seats = [s for s in pending.seats if _ai_agent_for(occupants[s]) is not None]
      if ai_seats:
        active.thinking.update(ai_seats)
        pending_kind = pending.kind
        # Snapshot views under the lock (PublicState.snapshot() deep-copies); the
        # decision itself runs outside it so a human's request is never blocked on
        # a bot "thinking" (research R3).
        views = {s: AgentView(public=active.game.public_state(),
                              private=active.game.private_view(s)) for s in ai_seats}
        agents = {s: _ai_agent_for(occupants[s]) for s in ai_seats}

    # The idle sleep must sit outside the lock: sleeping while holding it would
    # starve every request in an all-human game.
    if not ai_seats:
      time.sleep(AI_POLL_INTERVAL_S)
      continue

    decisions = {}
    for seat, agent in agents.items():
      view = views[seat]
      if pending_kind == "declare":
        decisions[seat] = ("declare", agent.declare(view), agent.claim(view))
      else:
        decisions[seat] = ("cut", agent.choose_cut(view), agent.claim(view))

    with _lock:
      if _generation != generation or _active is None or _active.game.finished:
        return
      active = _active
      for seat, (kind, value, claim) in decisions.items():
        active.thinking.discard(seat)
        if active.game.finished or seat not in active.game.pending.seats:
          continue   # phase moved on (or game ended) since we computed
        try:
          active.game.submit({"seat": seat, "kind": kind, "value": value, "claim": claim})
          active.version += 1
        except IllegalIntent:
          pass   # SolverBot guarantees legal moves; a stale view is the only way here
