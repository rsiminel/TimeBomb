"""The shared rules engine: a stepwise, event-sourced core (specs/002-host-local-game).

``TableGame`` inverts the referee from a blocking agent-driven loop (``sim/engine.py``,
now ``tbgame/driver.py``) into "ask what's pending, accept one intent at a time" -- the
shape the web needs to drive a game over HTTP, and the shape that gives save/resume and
post-game replay for free (the event log is the single source of truth; everything else
is derived). Rule authority -- hand sizes 5->2, one cut turn per player per round,
cutter-passes-to-target, win/loss judgment -- matches ``tbgame/driver.py`` (the arena's
regression harness for this promotion) exactly; the tricky cut-resolution arithmetic is
shared by both via ``_resolve_cut_draw`` so there is exactly one formula, not two.

The hidden deal reuses ``General.DistributeWires`` (the multivariate-hypergeometric law
the solver's inference assumes, model.md §3.1/§3.4.1); ``General.py`` is frozen and reads
random state from the global ``random`` module, so ``TableGame`` reseeds that module from
its own ``random.Random(seed)`` instance immediately before each call (research R2) --
bots keep their own unseeded RNG, and nothing else observes the reseed.
"""

import random

import General as tb

from .state import (GroundTruth, PublicState, PrivateView, Pending, EventLog,
                    legal_targets, SCHEMA_VERSION)

WIRE, BLANK, BOMB = "wire", "dud", "bomb"
INITIAL_HAND_SIZE = 5
FINAL_HAND_SIZE = 2
MIN_SEATS, MAX_SEATS = 4, 8
OCCUPANT_KINDS_BASE = ("human", "solver_bot")
CLAIM_KINDS = ("trust", "distrust", "accuse_lie", "self_honest")


class SetupError(Exception):
  """An invalid ``GameSetup``. ``reason`` is forwarded verbatim to the web layer."""

  def __init__(self, reason):
    super().__init__(reason)
    self.reason = reason


class IllegalIntent(Exception):
  """A ``submit()`` call that violates the rules. Nothing is applied. ``reason`` is
  forwarded verbatim to the web layer (422, contracts/api.md)."""

  def __init__(self, reason):
    super().__init__(reason)
    self.reason = reason


class SchemaError(Exception):
  """``from_events`` was handed a log whose ``schema_version`` this engine can't read."""

  def __init__(self, reason):
    super().__init__(reason)
    self.reason = reason


# ---------------------------------------------------------------------------
# Setup validation
# ---------------------------------------------------------------------------

def _is_llm_occupant(occupant):
  return isinstance(occupant, str) and occupant.startswith("llm:")


def _validate_setup(setup):
  if not isinstance(setup, dict):
    raise SetupError("setup must be an object")
  seats = setup.get("seats")
  if not isinstance(seats, list) or not (MIN_SEATS <= len(seats) <= MAX_SEATS):
    raise SetupError("seats must be a list of %d-%d entries" % (MIN_SEATS, MAX_SEATS))
  names, occupants = [], []
  for seat in seats:
    if not isinstance(seat, dict):
      raise SetupError("each seat must be an object")
    name = seat.get("name")
    if not isinstance(name, str) or not name.strip():
      raise SetupError("seat names must be non-empty")
    names.append(name.strip())
    occupant = seat.get("occupant")
    if occupant not in OCCUPANT_KINDS_BASE and not _is_llm_occupant(occupant):
      raise SetupError("seat occupant must be human, solver_bot, or llm:<model-tag>")
    occupants.append(occupant)
  if len(set(names)) != len(names):
    raise SetupError("seat names must be unique")
  n = len(seats)
  role_deal = setup.get("role_deal", "official")
  if role_deal != "official":
    if (not isinstance(role_deal, int) or isinstance(role_deal, bool)
        or not 1 <= role_deal <= n - 1):
      raise SetupError("role_deal override must be an integer between 1 and seats-1")
  seed = setup.get("seed")
  if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
    raise SetupError("seed must be an integer or null")
  panel_allowed = bool(setup.get("panel_allowed", False))
  return names, occupants, role_deal, seed, panel_allowed


def _validate_claim(seat, claim, num_players):
  if claim is None:
    return
  if not isinstance(claim, dict):
    raise IllegalIntent("claim must be an object or null")
  kind = claim.get("kind")
  target = claim.get("target")
  if kind not in CLAIM_KINDS:
    raise IllegalIntent("claim kind must be one of %s" % (CLAIM_KINDS,))
  if kind == "self_honest":
    if target is not None:
      raise IllegalIntent("a self_honest claim must not have a target")
  else:
    if (not isinstance(target, int) or isinstance(target, bool)
        or not 0 <= target < num_players):
      raise IllegalIntent("claim target must be a seat index")
    if target == seat:
      raise IllegalIntent("claim target must not be the speaker")


# ---------------------------------------------------------------------------
# Shared cut-resolution arithmetic (used by TableGame and tbgame.driver.Engine)
# ---------------------------------------------------------------------------

def _resolve_cut_draw(gt, pub, target, rng):
  """Reveal one uniformly random face-down card of ``target``. Treats the bomb as the
  last remaining slot (uniform over the rest). ``rng`` need only expose ``randint`` --
  the ``random`` module itself or a ``random.Random`` instance both qualify, so the
  formula is written once and used with either RNG source."""
  facedown = pub.hand_size - pub.revealed[target]
  draw = rng.randint(1, facedown)
  if gt.bombs[target] == 1 and draw == facedown:
    return BOMB
  if draw <= gt.wires[target] - pub.found[target]:
    return WIRE
  return BLANK


def _distribute_wires_seeded(rng, num_players, hand_size, active_wires, num_bom):
  """``General.DistributeWires`` draws from the global ``random`` module (frozen code,
  not ours to change). Reseed that module from our own instance immediately before
  calling it, so the deal is deterministic w.r.t. our seed without leaking our
  instance's state anywhere else."""
  random.seed(rng.randrange(2**63))
  wires, bombs = tb.DistributeWires(num_players, hand_size, active_wires, num_bom)
  return [int(w) for w in wires], [int(b) for b in bombs]


def _cuts_so_far(events):
  return [{"round": e["round"], "cutter": e["cutter"], "target": e["target"],
           "result": e["result"]} for e in events if e["type"] == "cut"]


def _claims_so_far(events):
  return [{"round": e["round"], "speaker": e["speaker"], "kind": e["kind"],
           "target": e.get("target")} for e in events if e["type"] == "claim"]


# Event fields that stay server-side until reveal(): the deal (round_start carries
# wires/bombs so from_events can replay without re-rolling) and declaration truth.
_HIDDEN_EVENT_FIELDS = ("wires", "bombs", "true_wires")


def public_event(event):
  """The event as safe to show any player mid-game (FR-008)."""
  return {k: v for k, v in event.items() if k not in _HIDDEN_EVENT_FIELDS}


# ---------------------------------------------------------------------------
# TableGame -- the stepwise core
# ---------------------------------------------------------------------------

class TableGame:
  """One hosted game, driven one intent at a time. See contracts/engine.md."""

  def __init__(self, setup):
    names, occupants, role_deal, seed, panel_allowed = _validate_setup(setup)
    n = len(names)
    rng = random.Random(seed)
    prior_b = tb.NUM_BAD_PRIOR(n)
    if role_deal == "official":
      bs = list(prior_b)
      num_bad = bs[0] if len(bs) == 1 else rng.choices(bs, [prior_b[b] for b in bs])[0]
    else:
      num_bad = role_deal
    roles = [0] * n
    for i in rng.sample(range(n), num_bad):
      roles[i] = 1

    self._rng = rng
    self._names = names
    self._occupants = occupants
    self._prior_b = prior_b
    self._num_bom = 1
    self._panel_allowed = panel_allowed
    self._exhibition = "human" not in occupants
    self._seed = seed
    self._roles = roles
    self._current_cutter = 0
    self._round_cuts_taken = 0
    self._decl_history = []
    self._finished = False
    self._outcome = None
    self.events = []
    self._log = EventLog(self.events)
    self._gt = None
    self._pub = None

    self._log.append("game_start", schema_version=SCHEMA_VERSION, num_players=n,
                     num_bad=num_bad, num_bom=self._num_bom, roles=roles,
                     player_names=names, seats=occupants, seed=seed,
                     panel_allowed=panel_allowed)
    self._start_round(0, INITIAL_HAND_SIZE, n)

  @classmethod
  def from_events(cls, events):
    """Rebuild a ``TableGame`` by reducing a saved event log. Never re-rolls the RNG --
    every dealt/cut outcome is read from the recorded events (research R2/R8)."""
    if not events or events[0].get("type") != "game_start":
      raise SchemaError("event log must start with a game_start event")
    start = events[0]
    if start.get("schema_version") != SCHEMA_VERSION:
      raise SchemaError("schema_version %r does not match %d"
                        % (start.get("schema_version"), SCHEMA_VERSION))
    self = object.__new__(cls)
    self._rng = random.Random()   # unused for pure replay; live post-resume submits use it
    self._names = list(start["player_names"])
    self._occupants = list(start["seats"])
    n = start["num_players"]
    self._prior_b = tb.NUM_BAD_PRIOR(n)
    self._num_bom = start["num_bom"]
    self._panel_allowed = bool(start.get("panel_allowed", False))
    self._exhibition = "human" not in self._occupants
    self._seed = start.get("seed")
    self._roles = list(start["roles"])
    self._current_cutter = 0
    self._round_cuts_taken = 0
    self._decl_history = []
    self._finished = False
    self._outcome = None
    self.events = list(events)
    self._log = EventLog(self.events)
    self._gt = None
    self._pub = None
    for event in events[1:]:
      self._replay_event(event)
    return self

  # -- pending / views -------------------------------------------------------

  @property
  def pending(self):
    return self._pub.pending

  @property
  def finished(self):
    return self._finished

  @property
  def outcome(self):
    return self._outcome

  @property
  def occupants(self):
    return list(self._occupants)

  @property
  def exhibition(self):
    return self._exhibition

  def public_state(self):
    """Deep-copied ``PublicState`` snapshot."""
    return self._pub.snapshot()

  def private_view(self, seat):
    return PrivateView(my_index=seat, my_role=self._roles[seat],
                       my_wires=int(self._gt.wires[seat]),
                       i_hold_bomb=bool(self._gt.bombs[seat]))

  def reveal(self):
    """Full ground truth + truth-annotated event list. Only when finished, or any time
    in an exhibition game (no human seats -- clarification Q5)."""
    if not self._finished and not self._exhibition:
      raise RuntimeError("reveal() is only available once finished (or in exhibition mode)")
    annotated = []
    for event in self.events:
      if event["type"] == "declaration":
        event = dict(event, lie=event["declared"] != event["true_wires"])
      annotated.append(event)
    return {"roles": list(self._roles), "outcome": self._outcome, "events": annotated}

  # -- submitting intents -----------------------------------------------------

  def submit(self, intent):
    """Apply one intent. Raises ``IllegalIntent`` (nothing applied) on any violation.
    Returns the list of newly appended events."""
    if self._finished:
      raise IllegalIntent("the game is already finished")
    pending = self._pub.pending
    seat, kind = intent.get("seat"), intent.get("kind")
    if pending is None or kind != pending.kind or seat not in pending.seats:
      raise IllegalIntent("seat %r may not submit a %r intent right now" % (seat, kind))
    claim = intent.get("claim")
    _validate_claim(seat, claim, len(self._names))

    before = len(self.events)
    if kind == "declare":
      value = intent.get("value")
      hand_size = self._pub.hand_size
      if (not isinstance(value, int) or isinstance(value, bool)
          or not 0 <= value <= hand_size):
        raise IllegalIntent("declared wire count must be an integer in [0, hand size]")
      self._commit_declare(seat, value, claim)
    else:
      target = intent.get("value")
      legal = legal_targets(self._pub, seat)
      if not isinstance(target, int) or isinstance(target, bool) or target not in legal:
        raise IllegalIntent("cut target must be a legal target: one of %s" % (legal,))
      self._commit_cut(seat, target, claim)
    return self.events[before:]

  def _apply_claim_if_any(self, seat, claim):
    if claim is None:
      return
    self._log.append("claim", round=self._pub.round_index, speaker=seat,
                     kind=claim["kind"], target=claim.get("target"))
    self._pub.claim_log.append({"round": self._pub.round_index, "speaker": seat,
                                "kind": claim["kind"], "target": claim.get("target")})

  def _commit_declare(self, seat, value, claim):
    self._pub.declarations[seat] = value
    self._log.append("declaration", round=self._pub.round_index, player=seat,
                     declared=value, true_wires=self._gt.wires[seat])
    self._pub.pending.seats.remove(seat)
    self._apply_claim_if_any(seat, claim)
    if not self._pub.pending.seats:
      self._decl_history.append(list(self._pub.declarations))
      self._pub.declaration_history = list(self._decl_history)
      self._pub.phase = "awaiting_cut"
      self._pub.pending = Pending("cut", [self._current_cutter])

  def _commit_cut(self, seat, target, claim):
    result = _resolve_cut_draw(self._gt, self._pub, target, self._rng)
    self._pub.revealed[target] += 1
    self._log.append("cut", round=self._pub.round_index, cutter=seat, target=target,
                     result=result)
    self._pub.cut_log.append({"round": self._pub.round_index, "cutter": seat,
                              "target": target, "result": result})
    self._apply_claim_if_any(seat, claim)
    self._round_cuts_taken += 1

    if result == BOMB:
      self._finish(won=False, reason="bomb detonated")
      return
    if result == WIRE:
      self._pub.found[target] += 1
      self._pub.active_wires -= 1
      if self._pub.active_wires <= 0:
        self._finish(won=True, reason="all wires cut")
        return

    self._current_cutter = target
    self._pub.current_cutter = target
    if not legal_targets(self._pub, target):
      self._log.append("cut_skipped", round=self._pub.round_index, cutter=target)
      self._close_round()
      return
    if self._round_cuts_taken >= len(self._names):
      self._close_round()
      return
    self._pub.pending = Pending("cut", [target])

  def _close_round(self):
    self._log.append("round_end", round=self._pub.round_index)
    new_hand_size = self._pub.hand_size - 1
    if new_hand_size < FINAL_HAND_SIZE:
      self._finish(won=False, reason="out of time")
      return
    self._start_round(self._pub.round_index + 1, new_hand_size, self._pub.active_wires)

  def _finish(self, won, reason):
    outcome = {"good_guys_won": won, "reason": reason, "roles": list(self._roles)}
    self._log.append("game_end", **outcome)
    self._finished = True
    self._outcome = outcome
    self._pub.phase = "finished"
    self._pub.pending = None

  def _start_round(self, round_index, hand_size, active_wires, wires=None, bombs=None):
    n = len(self._names)
    if wires is None:
      wires, bombs = _distribute_wires_seeded(self._rng, n, hand_size, active_wires,
                                              self._num_bom)
      self._log.append("round_start", round=round_index, hand_size=hand_size,
                       active_wires=active_wires, wires=list(wires), bombs=list(bombs))
    self._gt = GroundTruth(roles=list(self._roles), wires=list(wires), bombs=list(bombs),
                           num_bad=sum(self._roles), seed=self._seed)
    self._pub = PublicState(
        num_players=n, num_bad_prior=self._prior_b, num_bom=self._num_bom,
        player_names=self._names, round_index=round_index, hand_size=hand_size,
        round_start_active=active_wires, active_wires=active_wires,
        declarations=[None] * n, revealed=[0] * n, found=[0] * n,
        cut_log=_cuts_so_far(self.events), discussion_log=[],
        claim_log=_claims_so_far(self.events),
        declaration_history=list(self._decl_history), current_cutter=self._current_cutter,
        phase="awaiting_declarations", pending=Pending("declare", list(range(n))))
    self._round_cuts_taken = 0

  # -- replay (from_events) ---------------------------------------------------

  def _replay_event(self, event):
    t = event["type"]
    if t == "round_start":
      self._start_round(event["round"], event["hand_size"], event["active_wires"],
                        wires=event["wires"], bombs=event["bombs"])
    elif t == "declaration":
      seat = event["player"]
      self._pub.declarations[seat] = event["declared"]
      self._pub.pending.seats.remove(seat)
      if not self._pub.pending.seats:
        self._decl_history.append(list(self._pub.declarations))
        self._pub.declaration_history = list(self._decl_history)
        self._pub.phase = "awaiting_cut"
        self._pub.pending = Pending("cut", [self._current_cutter])
    elif t == "claim":
      self._pub.claim_log.append({"round": event["round"], "speaker": event["speaker"],
                                  "kind": event["kind"], "target": event.get("target")})
    elif t == "cut":
      target, result = event["target"], event["result"]
      self._pub.revealed[target] += 1
      self._pub.cut_log.append({"round": event["round"], "cutter": event["cutter"],
                                "target": target, "result": result})
      self._round_cuts_taken += 1
      if result == WIRE:
        self._pub.found[target] += 1
        self._pub.active_wires -= 1
      self._current_cutter = target
      self._pub.current_cutter = target
      self._pub.pending = Pending("cut", [target])
    elif t in ("cut_skipped", "round_end"):
      pass   # no-op: a following round_start/game_end event fully rebuilds live state
    elif t == "game_end":
      self._finished = True
      self._outcome = {"good_guys_won": event["good_guys_won"], "reason": event["reason"],
                       "roles": event["roles"]}
      self._pub.phase = "finished"
      self._pub.pending = None
    else:
      raise SchemaError("unknown event type %r" % (t,))
