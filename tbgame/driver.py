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

Resume (see ``_reconstruct``): ``play_game`` accepts an optional pre-built ``log``. If
it already carries events, the game is rebuilt from them -- exactly the pattern
``tbgame.engine.TableGame.from_events`` uses for the web game -- and the dealt wires
and bombs are always read back off the logged ``round_start``, never re-rolled. A game
never crashes on a failed LLM call (it retries, then falls back to a legal move) unless
``halt_on_failure`` is set, in which case it stops at the exact failed decision instead
of degrading into silent random play, so there is something precise to resume from.
"""

import random
from concurrent.futures import ThreadPoolExecutor

import General as tb

from .state import GroundTruth, PublicState, PrivateView, AgentView, EventLog, legal_targets
from .engine import _resolve_cut_draw, WIRE, BLANK, BOMB

# sim's own on-disk log/manifest schema version -- untouched by the tbgame promotion
# (tbgame.state.SCHEMA_VERSION is a separate, bumped counter for the web game's logs).
# Bumped 1 -> 2 for resume support: the new `discuss_turn` event (so a silent turn still
# leaves a trace -- otherwise "asked and stayed silent" and "not asked yet" look the
# same on replay). Logs written before the bump remain fully valid for transcripts and
# analysis; they just predate --resume.
SCHEMA_VERSION = 2


class EngineHalted(Exception):
  """Raised out of ``play_game`` when ``halt_on_failure`` is set and one decision
  exhausts its retries. Carries just enough for a clear message -- resume itself is
  driven entirely by the event log already flushed to disk (sim/run.py), not by
  anything on this exception."""

  def __init__(self, round_index, phase, detail):
    self.round_index, self.phase, self.detail = round_index, phase, detail
    super().__init__("round %d, %s phase: %s" % (round_index + 1, phase, detail))


class Engine:
  """One referee instance plays one or more games with a fixed roster and settings."""

  def __init__(self, num_players=6, initial_hand_size=5, player_names=None,
               panel_for=(), assistant=None, max_workers=None, talk_between_cuts=False,
               talk_top_k=None, halt_on_failure=True):
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
    # Stop the game the instant one decision exhausts its retries, instead of the old
    # silent-fallback-to-a-legal-move behavior -- so there is an exact point to resume
    # from. False restores the old behavior (a flaky call never stops the game).
    self.halt_on_failure = halt_on_failure

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

  def _talk_pass(self, agents, gt, pub, log, round_index, order, cuts_before, bids=None,
                 already_asked=frozenset()):
    """One sequential discussion pass in ``order``; each speaker's view already holds every
    earlier statement. ``cuts_before`` -- how many cuts this round precede the pass -- is
    recorded on each statement so renderers can interleave talk with cuts chronologically.
    With ``talk_top_k`` set, only the k highest bidders in ``order`` are called (ties keep
    pass order); everyone else is skipped without a call and re-bids on its next reply.
    ``already_asked`` (resume only) additionally skips seats already asked this exact pass."""
    if self.talk_top_k is not None and bids is not None:
      order = _eager_order(order, bids, self.talk_top_k)
    order = [p for p in order if p not in already_asked]
    for speaker in order:
      view = self._build_view(gt, pub, speaker)
      statement = agents[speaker].discuss(view)
      if self.halt_on_failure and getattr(agents[speaker], "last_call_failed", False):
        raise EngineHalted(round_index, "discuss", "%s's turn to speak" % pub.player_names[speaker])
      urgency = self._bid(agents, speaker, bids) if bids is not None else None
      # Logged unconditionally (even on silence) so resume can tell "asked, stayed quiet"
      # apart from "never asked" -- a silent turn otherwise leaves no trace at all.
      log.append("discuss_turn", round=round_index, player=speaker, cuts_before=cuts_before,
                 spoke=bool(statement))
      if not statement:
        continue
      reasoning = getattr(agents[speaker], "last_reasoning", None)
      log.append("statement", round=round_index, player=speaker, message=statement,
                 reasoning=reasoning, cuts_before=cuts_before, urgency=urgency)
      pub.discussion_log.append({"round": round_index, "speaker": speaker,
                                 "message": statement, "cuts_before": cuts_before})
      self._broadcast(agents, "statement", player=speaker, message=statement)

  # -- the game loop --------------------------------------------------------

  def play_game(self, agents, seed=None, log=None):
    """Play one full game. ``agents`` is a list of length ``num_players``. Returns
    ``(outcome, log)`` where ``outcome`` is a dict and ``log`` is the ``EventLog``.

    ``log=None`` (every existing caller) is the original, byte-identical path: a fresh
    ``EventLog``, roles freshly sampled from ``seed``. Passing a ``log`` whose events
    already hold a partial (or complete) game resumes it instead -- see
    ``_reconstruct``; a completed game's outcome is simply returned. Raises
    ``EngineHalted`` (never crashes) if ``halt_on_failure`` and a decision exhausts its
    retries; the caller (sim/run.py) is expected to have been flushing ``log`` to disk
    incrementally, so nothing already committed is lost."""
    assert len(agents) == self.num_players
    if log is None:
      log = EventLog()
    resuming = bool(log.events)
    N = self.num_players

    if resuming:
      state = _reconstruct(log.events, self.talk_between_cuts, self.talk_top_k)
      if state["finished"]:
        return state["outcome"], log
      roles, num_bad, num_bom = state["roles"], state["num_bad"], state["num_bom"]
      prior_b = tb.NUM_BAD_PRIOR(N)
      hand_size, active_wires = state["hand_size"], state["active_wires"]
      current_cutter = state["current_cutter"]
      decl_history, bids = state["decl_history"], state["bids"]
      resume_here = None if state["round_index"] is None else state
    else:
      if seed is not None:
        random.seed(seed)                  # one RNG source (also drives DistributeWires)
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
      hand_size, active_wires, current_cutter = self.initial_hand_size, N, 0
      decl_history, bids = [], [0] * N
      resume_here = None

    while hand_size > 1:
      round_index = self.initial_hand_size - hand_size
      here = resume_here if (resume_here and resume_here["round_index"] == round_index) else None
      resume_here = None      # one-shot: only the first round touched can carry it
      # ProbDeclaration's round-start total: for a fresh round this equals the live
      # `active_wires` (nothing's been cut yet); a resumed round's `active_wires` is
      # already decremented by its pre-halt cuts, so it must come from the reconstructed
      # round_start event instead (CLAUDE.md: mixing the two produces a nonsense prior).
      round_start_active = here["round_start_active"] if here else active_wires

      if here:
        wires, bombs = here["wires"], here["bombs"]
      else:
        wires, bombs = tb.DistributeWires(N, hand_size, active_wires, num_bom)
        wires, bombs = [int(w) for w in wires], [int(b) for b in bombs]
      gt = GroundTruth(roles=roles, wires=wires, bombs=bombs, num_bad=num_bad, seed=seed)
      pub = PublicState(
          num_players=N, num_bad_prior=prior_b, num_bom=num_bom,
          player_names=self.player_names, round_index=round_index, hand_size=hand_size,
          round_start_active=round_start_active, active_wires=active_wires,
          declarations=(here["declarations"] if here else [None] * N),
          revealed=(here["revealed"] if here else [0] * N),
          found=(here["found"] if here else [0] * N),
          cut_log=log_cuts(log), discussion_log=log_statements(log),
          declaration_history=list(decl_history), current_cutter=current_cutter)
      if not here:
        log.append("round_start", round=round_index, hand_size=hand_size,
                   active_wires=active_wires, wires=wires, bombs=bombs)

      pending = here["pending"] if here else None

      # -- declaration phase (simultaneous; calls run in parallel) -----------
      # No player sees another's same-round declaration (every view is built before any
      # call), so the N calls are independent and run concurrently -- the main per-game
      # speedup, since each LLM call is seconds of mostly-startup latency.
      if pending is None or pending["phase"] == "declare":
        pending_declare = pending["seats"] if pending else list(range(N))
        if pending_declare:
          views = {i: self._build_view(gt, pub, i) for i in pending_declare}
          with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            raws = dict(zip(pending_declare, ex.map(
                lambda a_v: a_v[0].declare(a_v[1]),
                [(agents[i], views[i]) for i in pending_declare])))
          failed = [i for i in pending_declare if agents[i].last_call_failed]
          skip = set(failed) if self.halt_on_failure else set()
          for i in pending_declare:
            if i in skip:
              continue
            reasoning = getattr(agents[i], "last_reasoning", None)
            d = _validate_declaration(raws[i], hand_size, fallback=wires[i])
            pub.declarations[i] = d
            log.append("declaration", round=round_index, player=i, declared=d,
                       true_wires=wires[i], reasoning=reasoning,
                       urgency=self._bid(agents, i, bids))
            self._broadcast(agents, "declaration", player=i, declared=d)
          if failed and self.halt_on_failure:
            raise EngineHalted(round_index, "declare", "seat(s) %s" % failed)
        pending = None                     # fully done now; later phases run fresh
      decl_history.append(list(pub.declarations))   # unconditional: once per round, resumed or not

      # -- discussion phase (sequential; each speaker hears those before it) --
      # Declarations are public now; players speak before any cut, so a later speaker can
      # react to (and accuse) earlier ones. Silent agents (default ``discuss`` -> None)
      # simply add nothing. With ``talk_between_cuts`` the table instead gets a word before
      # every cut (inside the cut loop below); this standalone pass is the original
      # one-pass-per-round structure, kept as the default.
      if not self.talk_between_cuts:
        if pending is None or pending["phase"] == "discuss":
          already = pending["already_asked"] if pending else frozenset()
          self._talk_pass(agents, gt, pub, log, round_index, range(N), cuts_before=0,
                          bids=bids, already_asked=already)
        pending = pending if (pending and pending["phase"] == "cut") else None

      # -- cut phase (N cuts; the cut target takes the cutters next) ---------
      cut_no_start = pending["cut_no"] if pending else 0
      for cut_no in range(cut_no_start, N):
        pub.current_cutter = current_cutter
        legal = legal_targets(pub, current_cutter)
        if not legal:                       # cutter has no legal target; skip
          log.append("cut_skipped", round=round_index, cutter=current_cutter)
          break
        # On resume, cut_no_start's own pass may already be fully done (pending["phase"]
        # == "cut") -- re-running it would re-pick eager speakers off the CURRENT bids,
        # which have moved on since (each speaker's own reply updates its bid for its
        # NEXT turn), producing a different, wrong winner for a pass that already happened.
        pass_already_done = (cut_no == cut_no_start and pending
                             and pending["phase"] == "cut")
        if self.talk_between_cuts and not pass_already_done:
          # A full pass before this cut: the first reacts to the declarations, later ones
          # to the cut just made. Speaking starts at the seat after the cutter and ends
          # WITH the cutter -- the about-to-act player hears everyone, gets the last word,
          # then cuts; no seat is structurally first, since the start rotates with play.
          order = [(current_cutter + 1 + k) % N for k in range(N)]
          already = (pending["already_asked"]
                    if pending and pending["phase"] == "discuss" and cut_no == cut_no_start
                    else frozenset())
          self._talk_pass(agents, gt, pub, log, round_index, order, cuts_before=cut_no,
                          bids=bids, already_asked=already)
        view = self._build_view(gt, pub, current_cutter)
        raw = agents[current_cutter].choose_cut(view)
        if self.halt_on_failure and agents[current_cutter].last_call_failed:
          raise EngineHalted(round_index, "cut", "%s's cut" % pub.player_names[current_cutter])
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
# Resume -- rebuild everything play_game needs from a (partial) event log
# ---------------------------------------------------------------------------

def _eager_order(order, bids, talk_top_k):
  """The ``talk_top_k`` highest bidders in ``order`` (ties keep pass order) -- the exact
  rationing rule ``_talk_pass`` applies live; shared with ``_reconstruct`` so the two can
  never drift apart."""
  keep = set(sorted(order, key=lambda p: -bids[p])[:talk_top_k])
  return [p for p in order if p in keep]


def _reconstruct(events, talk_between_cuts, talk_top_k):
  """Replay ``events`` (a non-empty log) into everything ``play_game`` needs to resume:
  the deal (read back off the logged ``round_start``, never re-rolled -- same rule as
  ``TableGame.from_events``), ``decl_history`` for every closed round, and exactly what's
  still ``pending`` in the last (open) round. If the log ends in ``game_end``, returns
  the finished outcome directly -- resuming a completed game is a no-op, not an error.
  If it ends cleanly between rounds (``round_end`` with nothing after -- the process died
  outside any agent call, not via ``EngineHalted``), ``round_index`` comes back ``None``
  and the caller just continues the loop fresh from the next round."""
  start = events[0]
  if start["type"] != "game_start":
    raise ValueError("event log must start with a game_start event")
  if start.get("schema_version") != SCHEMA_VERSION:
    raise ValueError("schema_version %r predates resume support (need %d)"
                     % (start.get("schema_version"), SCHEMA_VERSION))

  N = start["num_players"]
  bids = [0] * N
  # Live bids, mutated by every reply -- what play_game seeds its own ``bids`` array with,
  # for passes that start AFTER resume. ``bids_snapshot`` is different: it's frozen at the
  # last declare/cut boundary, i.e. exactly what the live ``_talk_pass`` had in hand when it
  # picked the still-open pass's top-k eager set. A pass's own statements update ``bids``
  # (each speaker's bid moves for its NEXT turn) but must not retroactively change who was
  # eligible for the pass already in progress -- that's frozen the moment the pass starts.
  bids_snapshot = [0] * N

  def _bump(player, urgency):
    if urgency is not None:
      bids[player] = urgency

  decl_history = []
  round_index = hand_size = active_wires = round_start_active = None
  wires = bombs = declarations = revealed = found = None
  discuss_turns, cut_count = [], 0
  current_cutter = 0
  round_closed = True          # no round_start seen yet, or the last one fully closed
  decl_folded = True           # this round's declarations already folded into decl_history

  for e in events[1:]:
    t = e["type"]
    if t == "round_start":
      round_index, hand_size, active_wires = e["round"], e["hand_size"], e["active_wires"]
      round_start_active = active_wires   # kept unmutated -- ProbDeclaration's round-start
                                           # total, distinct from the live, decrementing count
      wires, bombs = list(e["wires"]), list(e["bombs"])
      declarations, revealed, found = [None] * N, [0] * N, [0] * N
      discuss_turns, cut_count, round_closed, decl_folded = [], 0, False, False
    elif t == "declaration":
      declarations[e["player"]] = e["declared"]
      _bump(e["player"], e.get("urgency"))
      bids_snapshot = list(bids)          # the boundary the next (first) pass starts from
    elif t == "discuss_turn":
      discuss_turns.append((e["player"], e["cuts_before"]))
    elif t == "statement":
      _bump(e["player"], e.get("urgency"))   # live only -- see bids_snapshot note above
    elif t == "cut":
      revealed[e["target"]] += 1
      if e["result"] == WIRE:
        found[e["target"]] += 1
        active_wires -= 1
      current_cutter = e["target"]
      cut_count += 1
      if cut_count >= N:
        round_closed = True             # cut through; round_end is just a trailing marker
      _bump(e["cutter"], e.get("urgency"))
      bids_snapshot = list(bids)         # the boundary the NEXT pass (if any) starts from
    elif t == "cut_skipped":
      round_closed = True               # ends the round early, same as reaching N cuts
    elif t == "round_end":
      round_closed = True
    elif t == "game_end":
      outcome = {"good_guys_won": e["good_guys_won"], "reason": e["reason"], "roles": e["roles"]}
      return dict(finished=True, outcome=outcome)
    if round_closed and not decl_folded:
      decl_history.append(list(declarations))
      decl_folded = True

  base = dict(finished=False, roles=list(start["roles"]), num_bad=start["num_bad"],
             num_bom=start["num_bom"], decl_history=decl_history, bids=bids,
             current_cutter=current_cutter)

  if round_closed:
    # Between rounds, or this round is effectively over (cut through / skipped) and just
    # missing its trailing round_end marker: nothing to narrow, the next round is fresh.
    return dict(base, round_index=None, hand_size=hand_size - 1, active_wires=active_wires)

  # -- what's left in this open round? ---------------------------------------------
  missing_decl = [i for i in range(N) if declarations[i] is None]
  if missing_decl:
    pending = {"phase": "declare", "seats": missing_decl, "cut_no": 0}
  else:
    if talk_between_cuts:
      cb, order = cut_count, [(current_cutter + 1 + k) % N for k in range(N)]
    elif cut_count == 0:
      cb, order = 0, list(range(N))       # the one standalone pass, before any cuts
    else:
      cb, order = None, None              # standalone pass already happened
    if order is not None:
      eager = (_eager_order(order, bids_snapshot, talk_top_k) if talk_top_k is not None
               else order)
      asked = {p for p, pcb in discuss_turns if pcb == cb}
      remaining = [p for p in eager if p not in asked]
    else:
      remaining = []
    if remaining:
      pending = {"phase": "discuss", "already_asked": asked, "cut_no": cb}
    else:
      pending = {"phase": "cut", "cut_no": cut_count}

  return dict(base, round_index=round_index, hand_size=hand_size, active_wires=active_wires,
             round_start_active=round_start_active, wires=wires, bombs=bombs,
             declarations=declarations, revealed=revealed, found=found, pending=pending)


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
