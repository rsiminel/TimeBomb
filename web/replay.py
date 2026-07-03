"""Event-log replay for the Time Bomb web assistant (specs/001-web-cut-panel).

Validates a GameRecord ``{setup, events}`` and replays it into derived rules state
plus the solver-computed belief. Constitution I: this module is rules bookkeeping
only — every probability comes from ``timebomb/General.py``, called in exactly the
pattern of ``General.Play`` (the reference orchestration). Undo and reload need no
code here: the client truncates or re-sends the log and this replay is pure.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "timebomb"))

import numpy as np

import General as gen

INITIAL_HAND_SIZE = 5
FINAL_HAND_SIZE = 2
MIN_PLAYERS = 4
MAX_PLAYERS = 8
RESULTS = ("safe", "nothing", "bomb")


class RecordError(Exception):
  """A record that violates the data model. ``event_index`` pinpoints the first
  offending event (-1 = the setup itself) for the 422 body of contracts/api.md."""

  def __init__(self, message, event_index):
    super().__init__(message)
    self.event_index = event_index


def _validate_setup(record):
  if not isinstance(record, dict):
    raise RecordError("record must be an object", -1)
  setup = record.get("setup")
  if not isinstance(setup, dict):
    raise RecordError("missing setup object", -1)
  players = setup.get("players")
  if not isinstance(players, list) or not all(isinstance(p, str) for p in players):
    raise RecordError("setup.players must be a list of names", -1)
  players = [p.strip() for p in players]
  if not (MIN_PLAYERS <= len(players) <= MAX_PLAYERS):
    raise RecordError("Time Bomb supports 4-8 players", -1)
  if any(p == "" for p in players):
    raise RecordError("player names must be non-empty", -1)
  if len(set(players)) != len(players):
    raise RecordError("player names must be unique", -1)
  bomb = setup.get("bomb")
  if not isinstance(bomb, bool):
    raise RecordError("setup.bomb must be true or false", -1)
  override = setup.get("numBadOverride")
  if override is not None:
    if not isinstance(override, int) or isinstance(override, bool):
      raise RecordError("setup.numBadOverride must be an integer or null", -1)
    if not 1 <= override <= len(players) - 1:  # the range the solver supports
      raise RecordError("numBadOverride must be between 1 and players-1", -1)
  events = record.get("events", [])
  if not isinstance(events, list):
    raise RecordError("events must be a list", -1)
  return players, bomb, override, events


class _Replay:
  """The state machine of data-model.md, advanced one event at a time."""

  def __init__(self, players, bomb, override):
    self.players = players
    self.n = len(players)
    self.num_bom = 1 if bomb else 0
    self.prior_b = {override: 1.0} if override is not None else gen.NUM_BAD_PRIOR(self.n)
    self.candidate_bs = list(self.prior_b)
    self.log_u_by_b = {b: np.zeros([self.n] * b) for b in self.candidate_bs}
    self.hand_size = INITIAL_HAND_SIZE
    self.active_wires = self.n
    self.round_num = 0          # completed or in-progress rounds (0 = none started)
    self.awaiting = "declarations"
    self.game_over = None
    # current-round state (valid while awaiting == "cut" or the game just ended)
    self.decls = None
    self.total_active = None
    self.revealed = None
    self.found = None
    self.cuts_made = 0
    self.warnings = []
    self.decl_warned = False    # at most one of each warning per round, as in Play
    self.cut_warned = False

  def apply(self, event, idx):
    if not isinstance(event, dict):
      raise RecordError("event must be an object", idx)
    if self.game_over is not None:
      raise RecordError("no entries accepted after game over", idx)
    etype = event.get("type")
    if etype == "declarations":
      self._apply_declarations(event, idx)
    elif etype == "cut":
      self._apply_cut(event, idx)
    else:
      raise RecordError("unknown event type", idx)

  def _apply_declarations(self, event, idx):
    if self.awaiting != "declarations":
      raise RecordError("declarations only open a round", idx)
    values = event.get("values")
    if not isinstance(values, list) or len(values) != self.n:
      raise RecordError("declarations need one value per player", idx)
    for v in values:
      if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v <= self.hand_size:
        raise RecordError("declared wire counts must be integers in [0, hand size]", idx)
    self.round_num += 1
    self.decls = np.array(values, dtype=int)
    self.total_active = self.active_wires
    self.revealed = np.zeros(self.n, dtype=int)
    self.found = np.zeros(self.n, dtype=int)
    self.cuts_made = 0
    self.decl_warned = False
    self.cut_warned = False
    self.awaiting = "cut"

  def _apply_cut(self, event, idx):
    if self.awaiting != "cut":
      raise RecordError("cut before this round's declarations", idx)
    player = event.get("player")
    if not isinstance(player, int) or isinstance(player, bool) or not 0 <= player < self.n:
      raise RecordError("cut.player must be a seat index", idx)
    result = event.get("result")
    if result not in RESULTS:
      raise RecordError("cut.result must be safe, nothing or bomb", idx)
    if self.revealed[player] >= self.hand_size:
      raise RecordError("cut on player with no face-down cards", idx)
    if result == "bomb":
      if self.num_bom == 0:
        raise RecordError("no bomb in this game", idx)
      # Game over. As in PlayAuto, the detonating cut is NOT folded into the belief:
      # the cut likelihood conditions on "no bomb drawn", which is now false.
      self.game_over = {"winner": "bad", "reason": "bomb"}
      self.awaiting = "over"
      return
    self.revealed[player] += 1
    if result == "safe":
      self.found[player] += 1
      self.active_wires -= 1
    self.cuts_made += 1
    if self.active_wires <= 0:
      self.game_over = {"winner": "good", "reason": "wires"}
      self.awaiting = "over"
      return
    if self.cuts_made == self.n:
      self._close_round()

  def _close_round(self):
    """The round's cut budget is spent: fold its evidence and advance (Play's round end)."""
    self._fold_round()
    if self.hand_size == FINAL_HAND_SIZE:
      self.game_over = {"winner": "bad", "reason": "time"}
      self.awaiting = "over"
      return
    self.hand_size -= 1
    self.decls = None
    self.awaiting = "declarations"

  def _fold_round(self):
    for b in self.candidate_bs:
      self.log_u_by_b[b] = self.log_u_by_b[b] + gen.RoundLogU(
          self.decls, self.revealed, self.found, self.hand_size,
          self.total_active, self.active_wires, b, self.num_bom)

  def state(self):
    over = self.game_over is not None
    display_round = self.round_num + (0 if over or self.awaiting == "cut" else 1)
    return {
        "round": max(display_round, 1),
        "handSize": self.hand_size,
        "activeWires": self.active_wires,
        "cutsMade": self.cuts_made,
        "cutsThisRound": self.n,
        "revealed": [] if self.revealed is None else [int(r) for r in self.revealed],
        "found": [] if self.found is None else [int(f) for f in self.found],
        "awaiting": self.awaiting,
        "gameOver": self.game_over,
    }

  def belief(self):
    return None  # solver orchestration lands with User Story 1


def replay_record(record):
  """Replay a full GameRecord into the response dict of contracts/api.md."""
  players, bomb, override, events = _validate_setup(record)
  replay = _Replay(players, bomb, override)
  for idx, event in enumerate(events):
    replay.apply(event, idx)
  return {
      "state": replay.state(),
      "belief": replay.belief(),
      "warnings": replay.warnings,
  }
