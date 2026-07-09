"""Resume coverage for the arena engine (tbgame/driver.py). The frozen
tests/test_arena_engine.py is never touched; this is new, additive coverage for
EngineHalted / play_game(log=...) / _reconstruct.

Runs entirely on deterministic stub agents -- no LLM, no network -- so it is fast and
reproducible, same spirit as test_arena_engine.py.
"""

import pytest

from engine import Engine, EngineHalted
from state import EventLog, legal_targets
from base import Agent
from llm import LLMAgent


class StubAgent(Agent):
    """Declares a fixed count; always cuts the lowest-index legal target. No opinions."""

    def __init__(self, decl=1):
        self.decl = decl
        self.last_reasoning = "stub"

    def declare(self, view):
        return min(self.decl, view.public.hand_size)

    def choose_cut(self, view):
        return min(legal_targets(view.public, view.private.my_index))


class FlakyAgent(StubAgent):
    """A StubAgent that also speaks, and fails its next call (declare/discuss/cut) exactly
    once when armed via the corresponding ``fail_*`` flag -- for exercising EngineHalted
    at each phase. Mirrors LLMAgent's own contract: ``last_call_failed`` reset at the top
    of every call, set only on the (here: deliberately forced) failure.

    ``declare_urgency``/``discuss_urgency`` -- when given -- make ``last_urgency`` report
    a DIFFERENT bid after speaking than it did at declare time, the way a real LLM's
    urgency field moves reply to reply. Needed to exercise the "eager set must freeze at
    pass-start, not drift with a speaker's own reply" resume regression."""

    def __init__(self, decl=1, fail_declare=False, fail_discuss=False, fail_cut=False,
                declare_urgency=None, discuss_urgency=None):
        super().__init__(decl)
        self.fail_declare = fail_declare
        self.fail_discuss = fail_discuss
        self.fail_cut = fail_cut
        self.declare_urgency = declare_urgency
        self.discuss_urgency = discuss_urgency
        self.last_call_failed = False
        self.last_urgency = None

    def declare(self, view):
        self.last_call_failed = False
        if self.fail_declare:
            self.fail_declare, self.last_call_failed = False, True
            return None
        if self.declare_urgency is not None:
            self.last_urgency = self.declare_urgency
        return super().declare(view)

    def discuss(self, view):
        self.last_call_failed = False
        if self.fail_discuss:
            self.fail_discuss, self.last_call_failed = False, True
            return None
        if self.discuss_urgency is not None:
            self.last_urgency = self.discuss_urgency
        return "seat %d checking in" % view.private.my_index

    def choose_cut(self, view):
        self.last_call_failed = False
        if self.fail_cut:
            self.fail_cut, self.last_call_failed = False, True
            return None
        return super().choose_cut(view)


def _events(log, kind):
    return [e for e in log.events if e["type"] == kind]


def _cuts(log):
    return [(e["cutter"], e["target"], e["result"]) for e in _events(log, "cut")]


# ---------------------------------------------------------------------------
# Halting -- one decision exhausts its retries, the game stops precisely there
# ---------------------------------------------------------------------------

def test_halts_mid_declare_wave_and_keeps_the_successful_declarations():
    agents = [StubAgent(), StubAgent(), FlakyAgent(fail_declare=True), StubAgent()]
    log = EventLog()
    with pytest.raises(EngineHalted) as exc:
        Engine(num_players=4).play_game(agents, seed=1, log=log)
    assert exc.value.phase == "declare"

    declared = {e["player"]: e["declared"] for e in _events(log, "declaration")}
    assert set(declared) == {0, 1, 3}          # seat 2's failed declaration was never logged
    assert _events(log, "cut") == []           # halted before any cut this round


def test_halts_mid_discuss_pass_and_marks_only_asked_seats():
    agents = [FlakyAgent(), FlakyAgent(), FlakyAgent(fail_discuss=True), FlakyAgent()]
    log = EventLog()
    with pytest.raises(EngineHalted) as exc:
        Engine(num_players=4, talk_between_cuts=False).play_game(agents, seed=2, log=log)
    assert exc.value.phase == "discuss"

    turns = {e["player"] for e in _events(log, "discuss_turn") if e["cuts_before"] == 0}
    assert turns == {0, 1}                     # 0 and 1 spoke before seat 2 failed
    assert 2 not in turns and 3 not in turns    # 2 failed (no trace); 3 never got asked
    assert len(_events(log, "declaration")) == 4   # declare phase had already completed


def test_halts_mid_cut_phase():
    agents = [StubAgent(), FlakyAgent(fail_cut=True), StubAgent(), StubAgent()]
    log = EventLog()
    with pytest.raises(EngineHalted) as exc:
        Engine(num_players=4).play_game(agents, seed=3, log=log)
    assert exc.value.phase == "cut"
    assert len(_events(log, "cut")) == 1        # seat 0's cut (-> seat 1) landed; seat 1's didn't


def test_halt_on_failure_off_falls_back_like_before():
    agents = [StubAgent(), StubAgent(), FlakyAgent(fail_declare=True), StubAgent()]
    outcome, log = Engine(num_players=4, halt_on_failure=False).play_game(agents, seed=1)
    assert outcome["reason"] in ("all wires cut", "bomb detonated", "out of time")
    assert len(_events(log, "declaration")) >= 4   # seat 2 still got a (fallback) declaration


def test_resume_does_not_replay_a_pass_already_won_by_top_k():
    # Regression: talk_between_cuts + talk_top_k means a per-cut pass picks ONE eager
    # speaker off the bids frozen at pass-start. That speaker's OWN reply then updates
    # its bid for its NEXT turn -- if resume recomputes eligibility off the post-reply
    # bids instead of a frozen snapshot, a pass that's already over looks re-runnable
    # and picks a DIFFERENT (wrong) winner. Seat 2 wins the pass at declare time (urgency
    # 7 vs everyone else's 3), then reports a much lower urgency (1) once it actually
    # speaks -- exactly the real-world sequence that surfaced this bug.
    agents = [
        FlakyAgent(declare_urgency=3),
        FlakyAgent(declare_urgency=3, fail_cut=True),
        FlakyAgent(declare_urgency=7, discuss_urgency=1),
        FlakyAgent(declare_urgency=3),
    ]
    log = EventLog()
    with pytest.raises(EngineHalted) as exc:
        Engine(num_players=4, talk_between_cuts=True, talk_top_k=1).play_game(
            agents, seed=4, log=log)
    assert exc.value.phase == "cut"                       # the pass already finished cleanly
    pass0_turns = [e for e in _events(log, "discuss_turn")
                  if e["cuts_before"] == 0 and e["round"] == 0]
    assert [e["player"] for e in pass0_turns] == [2]       # only seat 2 (top bidder) was asked

    resumed = [StubAgent(), StubAgent(), StubAgent(), StubAgent()]
    outcome, rlog = Engine(num_players=4, talk_between_cuts=True, talk_top_k=1).play_game(
        resumed, log=log)
    assert outcome["reason"] in ("all wires cut", "bomb detonated", "out of time")
    pass0_after = [e for e in _events(rlog, "discuss_turn")
                  if e["cuts_before"] == 0 and e["round"] == 0]
    assert pass0_after == pass0_turns                      # not re-run, not re-asked


# ---------------------------------------------------------------------------
# Resume correctness -- reconstruction must reach the same place an uninterrupted
# game would have, since the deal is replayed from the log, never re-rolled.
# ---------------------------------------------------------------------------

def test_resume_reaches_the_same_outcome_as_an_uninterrupted_game():
    seed = 42
    baseline, blog = Engine(num_players=4).play_game([StubAgent() for _ in range(4)], seed=seed)

    agents = [StubAgent(), StubAgent(), FlakyAgent(fail_declare=True), StubAgent()]
    log = EventLog()
    with pytest.raises(EngineHalted):
        Engine(num_players=4).play_game(agents, seed=seed, log=log)

    outcome, rlog = Engine(num_players=4).play_game([StubAgent() for _ in range(4)], log=log)
    assert outcome == baseline
    assert _cuts(rlog) == _cuts(blog)


def test_resume_after_mid_discuss_halt_finishes_the_pass_then_the_game():
    agents = [FlakyAgent(), FlakyAgent(), FlakyAgent(fail_discuss=True), FlakyAgent()]
    log = EventLog()
    with pytest.raises(EngineHalted):
        Engine(num_players=4, talk_between_cuts=False).play_game(agents, seed=2, log=log)

    resumed = [FlakyAgent(), FlakyAgent(), FlakyAgent(), FlakyAgent()]
    outcome, rlog = Engine(num_players=4, talk_between_cuts=False).play_game(resumed, log=log)
    assert outcome["reason"] in ("all wires cut", "bomb detonated", "out of time")
    round0_turns = [e for e in _events(rlog, "discuss_turn")
                   if e["cuts_before"] == 0 and e["round"] == 0]
    assert {e["player"] for e in round0_turns} == {0, 1, 2, 3}   # every seat asked, none skipped
    assert len(round0_turns) == 4                                # ... and none re-asked


def test_resume_after_mid_cut_halt_finishes_the_round():
    agents = [StubAgent(), FlakyAgent(fail_cut=True), StubAgent(), StubAgent()]
    log = EventLog()
    with pytest.raises(EngineHalted):
        Engine(num_players=4).play_game(agents, seed=3, log=log)

    resumed = [StubAgent() for _ in range(4)]
    outcome, rlog = Engine(num_players=4).play_game(resumed, log=log)
    assert outcome["reason"] in ("all wires cut", "bomb detonated", "out of time")
    cuts = _cuts(rlog)
    assert cuts[0] == (0, 1, cuts[0][2])        # the pre-halt cut is preserved, not redone
    assert len(cuts) >= 2


def test_resume_on_a_finished_game_is_a_noop():
    outcome, log = Engine(num_players=4).play_game([StubAgent() for _ in range(4)], seed=5)
    outcome2, log2 = Engine(num_players=4).play_game([StubAgent() for _ in range(4)], log=log)
    assert outcome2 == outcome
    assert log2 is log


# ---------------------------------------------------------------------------
# Sidecar round-trip -- LLMAgent's to_state()/load_state() (sim/run.py's sessions.json)
# ---------------------------------------------------------------------------

def test_llm_agent_state_round_trips():
    a = LLMAgent()
    a.session_id = "sess-123"
    a.cursor = {"cuts": 2, "statements": 1, "round": 1, "decls_round": 1}
    a.transcript = [{"decision": "declare", "prompt": "p", "reply": "r"}]
    a.usage["calls"] = 3

    b = LLMAgent()
    b.load_state(a.to_state())
    assert b.session_id == "sess-123"
    assert b.cursor == a.cursor
    assert b.transcript == a.transcript
    assert b.usage["calls"] == 3


def test_base_agent_state_hooks_are_inert():
    a = StubAgent()
    assert a.to_state() == {}
    a.load_state({"anything": 1})              # no-op, must not raise
    assert a.last_call_failed is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
