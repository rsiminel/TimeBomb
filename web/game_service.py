"""The one active hosted game: seat locks and the AI turn worker (data-model.md
ActiveGame). Transport-adjacent bookkeeping only -- every rule decision comes from
``tbgame.engine.TableGame``; every probability (opt-in panel) from ``panel_bridge.py``
-> ``replay.py`` -> ``General.py``. This module holds no math and no rule logic of its
own (constitution I).
"""
import datetime
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import panel_bridge
import replay
from tbgame.engine import (TableGame, SetupError, IllegalIntent, SchemaError,
                           CLAIM_KINDS, public_event)
from tbgame.state import AgentView, legal_targets
from tbgame.agents.solver import SolverBot

AI_POLL_INTERVAL_S = 0.2

# Manual named saves (data-model.md SaveFile), git-ignored local storage.
SAVES_DIR = Path(__file__).resolve().parent / "saves"
_SAVE_NAME_RE = re.compile(r"^[\w][\w \-]{0,39}$")   # filesystem-safe, no traversal

# LLM seat settings (FR-017): env var wins; else web/instance/settings.json.
INSTANCE_DIR = Path(__file__).resolve().parent / "instance"
LLM_KEY_ENV = "ANTHROPIC_API_KEY"


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
    self.agents = {}            # seat -> live agent (LLM sessions must persist)
    self.agent_error = None     # why the last agent build failed (pause reason)
    self.paused_llm = None      # PauseInfo {seat, kind, error} | None (data-model.md)
    # The *declared* setup override (None = official deal); what the panel bridge
    # feeds replay.py as numBadOverride. Never the sampled truth.
    self.num_bad_override = num_bad_override


_lock = threading.RLock()
_active = None
_generation = 0


def _agent_for_seat(active, seat):
  """The (cached) live agent for an AI seat; ``None`` when it can't be built --
  the reason lands in ``active.agent_error`` and the worker pauses the seat."""
  if seat in active.agents:
    return active.agents[seat]
  occupant = active.game.occupants[seat]
  if occupant == "solver_bot":
    agent = SolverBot()
  else:
    try:
      agent = _make_llm_agent(occupant)
    except LLMSeatError as err:
      active.agent_error = str(err)
      return None
  active.agents[seat] = agent
  return agent


def create_game(setup):
  global _active, _generation
  with _lock:
    if _active is not None:
      raise ActiveGameError("a game is already active", 409)
    try:
      game = TableGame(setup)
    except SetupError as err:
      raise ActiveGameError(err.reason, 422) from err
    if (any(occ.startswith("llm:") for occ in game.occupants)
        and _resolve_llm_key()[0] is None):
      raise ActiveGameError("LLM seats need an API key configured first (FR-017)", 422)
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
      "pausedLlm": active.paused_llm,
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
# LLM seats (FR-017, research R7): the arena's LLM agent behind the shared
# interface, configured web-side. Env key wins; file key is injected into the
# agent subprocess env; the key is never echoed to a browser.
# ---------------------------------------------------------------------------

class LLMSeatError(Exception):
  """An LLM seat that cannot act (no key, agent unavailable)."""


# Web-side prompt addition (NOT an arena prompt change): constrain any claim to
# the hosted game's fixed menu, riding the same JSON reply.
CLAIM_ADDENDUM = (
    '\nOptionally add "claim" to your JSON to make one public claim from this fixed '
    'menu: {"kind": "trust", "target": <seat>} | {"kind": "distrust", "target": '
    '<seat>} | {"kind": "accuse_lie", "target": <seat>} | {"kind": "self_honest"}. '
    "The target must be another player's seat number. Omit \"claim\" to say nothing; "
    "free-text table talk is not delivered in this game.")


def _resolve_llm_key():
  env = os.environ.get(LLM_KEY_ENV)
  if env:
    return env, "env"
  try:
    data = json.loads((INSTANCE_DIR / "settings.json").read_text())
    key = data.get("llm_api_key")
    if key:
      return key, "file"
  except (OSError, ValueError):
    pass
  return None, None


def llm_settings():
  key, source = _resolve_llm_key()
  return {"configured": key is not None, "source": source}


def set_llm_key(api_key):
  if os.environ.get(LLM_KEY_ENV):
    raise ActiveGameError("%s is set in the environment; it wins" % LLM_KEY_ENV, 409)
  if not isinstance(api_key, str) or not api_key.strip():
    raise ActiveGameError("api_key must be a non-empty string", 422)
  INSTANCE_DIR.mkdir(exist_ok=True)
  (INSTANCE_DIR / "settings.json").write_text(
      json.dumps({"llm_api_key": api_key.strip()}))


def _sanitize_claim(claim, view):
  """An external model's claim, kept only if it fits the menu exactly."""
  if not isinstance(claim, dict) or claim.get("kind") not in CLAIM_KINDS:
    return None
  kind, target = claim["kind"], claim.get("target")
  if kind == "self_honest":
    return {"kind": kind}
  me = view.private.my_index
  if (isinstance(target, int) and not isinstance(target, bool)
      and 0 <= target < view.public.num_players and target != me):
    return {"kind": kind, "target": target}
  return None


_web_llm_cls = None


def _web_llm_agent_cls():
  """Lazily import the arena agent (needs the sim source roots) and derive the
  web-configured subclass: claim-menu-constrained ask + optional file-sourced key."""
  global _web_llm_cls
  if _web_llm_cls is not None:
    return _web_llm_cls
  _root = Path(__file__).resolve().parents[1]
  for sub in ("sim", "sim/agents"):
    path = str(_root / sub)
    if path not in sys.path:
      sys.path.insert(0, path)
  import llm as arena_llm

  class WebLLMAgent(arena_llm.LLMAgent):
    def __init__(self, model, api_key):
      super().__init__(**({"model": model} if model else {}))
      self._api_key = api_key
      self._pending_claim = None

    def _env(self):
      env = super()._env()
      if self._api_key:
        env[LLM_KEY_ENV] = self._api_key
      return env

    def declare(self, view):
      value, self.last_reasoning, obj = self._decide(
          view, "declare", "declaration",
          arena_llm.DECLARE_INSTRUCTION + CLAIM_ADDENDUM)
      self._pending_claim = _sanitize_claim(obj.get("claim"), view)
      if value is not None:
        self._remember(view.public.round_index, "declared %d" % value)
      return value

    def choose_cut(self, view):
      value, self.last_reasoning, obj = self._decide(
          view, "cut", "target", arena_llm.CUT_INSTRUCTION + CLAIM_ADDENDUM)
      self._pending_claim = _sanitize_claim(obj.get("claim"), view)
      if value is not None:
        who = view.public.player_names[value] if 0 <= value < view.public.num_players else value
        self._remember(view.public.round_index, "cut %s (Player %s)." % (who, value))
      return value

    def claim(self, view):
      claim, self._pending_claim = self._pending_claim, None
      return claim

  _web_llm_cls = WebLLMAgent
  return _web_llm_cls


def _make_llm_agent(occupant):
  """Agent for an ``llm:<model>`` seat, or raise ``LLMSeatError``."""
  key, source = _resolve_llm_key()
  if key is None:
    raise LLMSeatError("no LLM API key configured -- substitute a solver bot "
                       "or configure a key in Settings")
  model = occupant.split(":", 1)[1] or None
  try:
    cls = _web_llm_agent_cls()
  except ImportError as exc:
    raise LLMSeatError("LLM agent unavailable: %s" % exc)
  return cls(model, key if source == "file" else None)


def llm_recover(seat, action):
  """Resolve a paused LLM seat (FR-017): retry keeps the agent (and its session);
  substitute swaps in a solver bot for the rest of the game."""
  with _lock:
    active = _require_active()
    paused = active.paused_llm
    if paused is None or paused["seat"] != seat:
      raise ActiveGameError("that seat is not paused", 422)
    if action == "substitute":
      active.agents[seat] = SolverBot()
    elif action != "retry":
      raise ActiveGameError("action must be retry or substitute", 422)
    active.paused_llm = None
    active.version += 1
    return active.version


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
      paused_seat = active.paused_llm["seat"] if active.paused_llm else None
      ai_seats, agents = [], {}
      for s in pending.seats:
        if occupants[s] == "human" or s == paused_seat:
          continue
        # While one LLM seat is paused, other LLM seats wait it out (one pause at
        # a time; also avoids hammering a flaky backend every worker tick).
        if occupants[s].startswith("llm:") and active.paused_llm is not None:
          continue
        agent = _agent_for_seat(active, s)
        if agent is None:      # unbuildable LLM seat (e.g. resumed with no key)
          if active.paused_llm is None:   # one pause at a time, no version churn
            active.paused_llm = {"seat": s, "kind": pending.kind,
                                 "error": active.agent_error or "LLM seat unavailable"}
            active.version += 1
          continue
        ai_seats.append(s)
        agents[s] = agent
      if ai_seats:
        active.thinking.update(ai_seats)
        pending_kind = pending.kind
        # Snapshot views under the lock (PublicState.snapshot() deep-copies); the
        # decision itself runs outside it so a human's request is never blocked on
        # a bot "thinking" (research R3).
        views = {s: AgentView(public=active.game.public_state(),
                              private=active.game.private_view(s)) for s in ai_seats}

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
        if value is None:      # a failed LLM decision pauses the game (FR-017)
          if active.paused_llm is None:
            agent = active.agents.get(seat)
            active.paused_llm = {"seat": seat, "kind": kind,
                                 "error": getattr(agent, "last_error", None)
                                          or "decision failed"}
            active.version += 1
          continue
        try:
          active.game.submit({"seat": seat, "kind": kind, "value": value, "claim": claim})
          active.version += 1
        except IllegalIntent:
          pass   # SolverBot guarantees legal moves; a stale view is the only way here
