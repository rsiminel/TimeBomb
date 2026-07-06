"""The sim arena's compatibility driver (specs/002-host-local-game).

Reproduces today's blocking ``Engine.play_game(agents, seed)`` API so ``sim/engine.py``
can shim to it and ``tests/test_arena_engine.py`` -- the untouchable root suite's
regression harness for this promotion -- passes byte-for-byte unmodified. The state
dataclasses come from ``tbgame.state`` and the tricky cut-resolution arithmetic from
``tbgame.engine`` (one formula, shared with ``TableGame``); the loop structure and its
richer, reasoning-annotated event log are kept here because they differ from the web
game's structured-claim protocol by design (arena free-text discussion vs. the web
game's fixed claim menu, research R6) and because this loop is what the frozen test
exercises directly (it calls ``_build_view``/``_resolve_cut`` as instance methods).
"""

import random
from concurrent.futures import ThreadPoolExecutor

import General as tb

from .state import GroundTruth, PublicState, PrivateView, AgentView, EventLog, legal_targets
from .engine import _resolve_cut_draw, WIRE, BLANK, BOMB

# sim's own on-disk log/manifest schema version -- untouched by the tbgame promotion
# (tbgame.state.SCHEMA_VERSION is a separate, bumped counter for the web game's logs).
SCHEMA_VERSION = 1


class Engine:
  """One referee instance plays one or more games with a fixed roster and settings."""

  def __init__(self, num_players=6, initial_hand_size=5, player_names=None,
               panel_for=(), assistant=None, max_workers=None, talk_between_cuts=False,
               talk_top_k=None):
    self.num_players = num_players
    self.initial_hand_size = initial_hand_size
    self.player_names = player_names or _default_names(num_players)
    self.panel_for = set(panel_for)        # indices shown the assistant panel
    self.assistant = assistant             # optional public-state -> panel adapter
    self.max_workers = max_workers or num_players   # concurrency for the declaration phase
    # False: one discussion pass per round, after declarations (the original structure).
    # True: a full pass before EVERY cut -- the first reacts to declarations, later ones to
    # the previous cut -- each ordered to END with the next cutter (see the cut loop).
    self.talk_between_cuts = talk_between_cuts
    # None: every player gets a discuss call each pass. k: turns are rationed -- only the k
    # highest urgency bidders (each reply's piggybacked "urgency" field) are called at all,
    # so a silence-inclined player costs zero calls. No reserved seat for the next cutter:
    # the cut reply's own "message" field is their mic.
    self.talk_top_k = talk_top_k

  # -- view construction (the firewall) -------------------------------------

  def _build_view(self, gt, pub, i):
    private = PrivateView(my_index=i, my_role=gt.roles[i],
                          my_wires=int(gt.wires[i]), i_hold_bomb=bool(gt.bombs[i]))
    panel = None
    if self.assistant is not None and i in self.panel_for:
      panel = self.assistant.panel(pub)    # adapter sees PublicState only
    return AgentView(public=pub.snapshot(), private=private, assistant_panel=panel)

  def _broadcast(self, agents, kind, **data):
    for a in agents:
      a.observe({"type": kind, **data})

  def _bid(self, agents, i, bids):
    """Fold player ``i``'s piggybacked speak-bid (its last reply's clamped "urgency" field)
    into ``bids``; an absent/unparsed bid keeps the previous one. Returns the raw read."""
    u = getattr(agents[i], "last_urgency", None)
    if u is not None:
      bids[i] = u
    return u

  def _talk_pass(self, agents, gt, pub, log, round_index, order, cuts_before, bids=None):
    """One sequential discussion pass in ``order``; each speaker's view already holds every
    earlier statement. ``cuts_before`` -- how many cuts this round precede the pass -- is
    recorded on each statement so renderers can interleave talk with cuts chronologically.
    With ``talk_top_k`` set, only the k highest bidders in ``order`` are called (ties keep
    pass order); everyone else is skipped without a call and re-bids on its next reply."""
    if self.talk_top_k is not None and bids is not None:
      eager = sorted(order, key=lambda p: -bids[p])[:self.talk_top_k]
      order = [p for p in order if p in eager]
    for speaker in order:
      view = self._build_view(gt, pub, speaker)
      statement = agents[speaker].discuss(view)
      urgency = self._bid(agents, speaker, bids) if bids is not None else None
      if not statement:
        continue
      reasoning = getattr(agents[speaker], "last_reasoning", None)
      log.append("statement", round=round_index, player=speaker, message=statement,
                 reasoning=reasoning, cuts_before=cuts_before, urgency=urgency)
      pub.discussion_log.append({"round": round_index, "speaker": speaker,
                                 "message": statement, "cuts_before": cuts_before})
      self._broadcast(agents, "statement", player=speaker, message=statement)

  # -- the game loop --------------------------------------------------------

  def play_game(self, agents, seed=None):
    """Play one full game. ``agents`` is a list of length ``num_players``. Returns
    ``(outcome, log)`` where ``outcome`` is a dict and ``log`` is the ``EventLog``."""
    assert len(agents) == self.num_players
    if seed is not None:
      random.seed(seed)                    # one RNG source (also drives DistributeWires)

    N, log = self.num_players, EventLog()
    prior_b = tb.NUM_BAD_PRIOR(N)
    bs = list(prior_b)
    num_bad = bs[0] if len(bs) == 1 else random.choices(bs, [prior_b[b] for b in bs])[0]
    roles = [0] * N
    for i in random.sample(range(N), num_bad):
      roles[i] = 1
    num_bom = 1
    log.append("game_start", schema_version=SCHEMA_VERSION, num_players=N, num_bad=num_bad,
               num_bom=num_bom, roles=roles, player_names=self.player_names, seed=seed,
               talk_between_cuts=self.talk_between_cuts, talk_top_k=self.talk_top_k,
               panel_for=sorted(self.panel_for), agents=[a.describe() for a in agents])

    hand_size = self.initial_hand_size
    active_wires = N
    current_cutter = 0
    decl_history = []
    bids = [0] * N        # each player's latest speak-bid; refreshed by every reply

    while hand_size > 1:
      round_index = self.initial_hand_size - hand_size
      round_start_active = active_wires
      wires, bombs = tb.DistributeWires(N, hand_size, active_wires, num_bom)
      wires, bombs = [int(w) for w in wires], [int(b) for b in bombs]
      gt = GroundTruth(roles=roles, wires=wires, bombs=bombs, num_bad=num_bad, seed=seed)
      pub = PublicState(
          num_players=N, num_bad_prior=prior_b, num_bom=num_bom,
          player_names=self.player_names, round_index=round_index, hand_size=hand_size,
          round_start_active=round_start_active, active_wires=active_wires,
          declarations=[None] * N, revealed=[0] * N, found=[0] * N,
          cut_log=log_cuts(log), discussion_log=log_statements(log),
          declaration_history=list(decl_history), current_cutter=current_cutter)
      log.append("round_start", round=round_index, hand_size=hand_size,
                 active_wires=active_wires, wires=wires, bombs=bombs)

      # -- declaration phase (simultaneous; calls run in parallel) -----------
      # No player sees another's same-round declaration (every view is built before any
      # call), so the N calls are independent and run concurrently -- the main per-game
      # speedup, since each LLM call is seconds of mostly-startup latency.
      views = [self._build_view(gt, pub, i) for i in range(N)]
      with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
        raws = list(ex.map(lambda a_v: a_v[0].declare(a_v[1]),
                           [(agents[i], views[i]) for i in range(N)]))
      for i in range(N):
        reasoning = getattr(agents[i], "last_reasoning", None)
        d = _validate_declaration(raws[i], hand_size, fallback=wires[i])
        pub.declarations[i] = d
        log.append("declaration", round=round_index, player=i, declared=d,
                   true_wires=wires[i], reasoning=reasoning,
                   urgency=self._bid(agents, i, bids))
        self._broadcast(agents, "declaration", player=i, declared=d)
      decl_history.append(list(pub.declarations))

      # -- discussion phase (sequential; each speaker hears those before it) --
      # Declarations are public now; players speak before any cut, so a later speaker can
      # react to (and accuse) earlier ones. Silent agents (default ``discuss`` -> None)
      # simply add nothing. With ``talk_between_cuts`` the table instead gets a word before
      # every cut (inside the cut loop below); this standalone pass is the original
      # one-pass-per-round structure, kept as the default.
      if not self.talk_between_cuts:
        self._talk_pass(agents, gt, pub, log, round_index, range(N), cuts_before=0, bids=bids)

      # -- cut phase (N cuts; the cut target takes the cutters next) ---------
      for cut_no in range(N):
        pub.current_cutter = current_cutter
        legal = legal_targets(pub, current_cutter)
        if not legal:                       # cutter has no legal target; skip
          log.append("cut_skipped", round=round_index, cutter=current_cutter)
          break
        if self.talk_between_cuts:
          # A full pass before this cut: the first reacts to the declarations, later ones
          # to the cut just made. Speaking starts at the seat after the cutter and ends
          # WITH the cutter -- the about-to-act player hears everyone, gets the last word,
          # then cuts; no seat is structurally first, since the start rotates with play.
          order = [(current_cutter + 1 + k) % N for k in range(N)]
          self._talk_pass(agents, gt, pub, log, round_index, order, cuts_before=cut_no,
                          bids=bids)
        view = self._build_view(gt, pub, current_cutter)
        raw = agents[current_cutter].choose_cut(view)
        reasoning = getattr(agents[current_cutter], "last_reasoning", None)
        message = getattr(agents[current_cutter], "last_message", None)   # public table talk
        target = _validate_target(raw, legal)
        result = _resolve_cut_draw(gt, pub, target, random)
        pub.revealed[target] += 1
        log.append("cut", round=round_index, cutter=current_cutter, target=target,
                   result=result, reasoning=reasoning, message=message,
                   urgency=self._bid(agents, current_cutter, bids))
        # Keep pub.cut_log live so the NEXT cutter's view shows this cut (it must agree
        # with pub.revealed, which is already updated). Refreshing only at round end left
        # mid-round views incoherent: face-down counts changed while "cuts this round"
        # stayed empty.
        pub.cut_log.append({"round": round_index, "cutter": current_cutter, "target": target,
                            "result": result, "message": message})
        self._broadcast(agents, "cut", cutter=current_cutter, target=target,
                        result=result, message=message)

        if result == BOMB:
          return self._end(log, won=False, reason="bomb detonated", p_bad_truth=roles)
        if result == WIRE:
          pub.found[target] += 1
          active_wires -= 1
          pub.active_wires = active_wires
          if active_wires <= 0:
            return self._end(log, won=True, reason="all wires cut", p_bad_truth=roles)
        current_cutter = target

      log.append("round_end", round=round_index)
      hand_size -= 1

    return self._end(log, won=False, reason="out of time", p_bad_truth=roles)

  # -- cut resolution against the hidden deal -------------------------------

  def _resolve_cut(self, gt, pub, target):
    """Reveal one uniformly random face-down card of ``target``. Mirrors PlayAuto's
    indexing trick: treat the bomb as the last remaining slot (uniform over the rest)."""
    return _resolve_cut_draw(gt, pub, target, random)

  def _end(self, log, won, reason, p_bad_truth):
    outcome = {"good_guys_won": won, "reason": reason, "roles": p_bad_truth}
    log.append("game_end", **outcome)
    return outcome, log


# ---------------------------------------------------------------------------
# Validation / fallbacks -- malformed agent output must never crash a batch
# ---------------------------------------------------------------------------

def _validate_declaration(d, hand_size, fallback):
  try:
    d = int(d)
  except (TypeError, ValueError):
    return fallback
  return d if 0 <= d <= hand_size else fallback


def _validate_target(t, legal):
  try:
    t = int(t)
  except (TypeError, ValueError):
    return random.choice(legal)
  return t if t in legal else random.choice(legal)


def log_cuts(log):
  return [{"round": e["round"], "cutter": e["cutter"], "target": e["target"],
           "result": e["result"], "message": e.get("message")}
          for e in log.events if e["type"] == "cut"]


def log_statements(log):
  return [{"round": e["round"], "speaker": e["player"], "message": e["message"],
           "cuts_before": e.get("cuts_before", 0)}
          for e in log.events if e["type"] == "statement"]


def _default_names(n):
  base = ["Alice", "Bob", "Clara", "Darryl", "Eve", "Frank", "Grace", "Henry"]
  return base[:n]
