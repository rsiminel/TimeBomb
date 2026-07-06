"""The 2026-07-04 conversation-structure + assistant-readout features (SPEC.md §6/§7).

Covers the ``talk_between_cuts`` discussion structure (a full pass before every cut,
ordered to end with the next cutter; silence supported everywhere) and the
``PanelAssistant`` public-info readout (orchestration checks + solver invariants; the
probability *math* is validated against the math.comb oracle in test_General.py).
``tests/test_arena_engine.py`` is frozen, so everything new lands here.
"""

import numpy as np
import pytest

from engine import Engine
from state import (PublicState, PrivateView, AgentView, render_agent,
                   render_session_delta, new_session_cursor, cursor_after_opener)
from base import Agent
from assistant import PanelAssistant

import General as gen


class Talker(Agent):
    """Speaks on every discuss turn; declares 1; cuts the lowest-index legal target."""

    def __init__(self):
        self.last_reasoning = "stub"
        self.n_discuss = 0

    def declare(self, view):
        return min(1, view.public.hand_size)

    def discuss(self, view):
        self.n_discuss += 1
        return "statement %d" % self.n_discuss

    def choose_cut(self, view):
        from state import legal_targets
        return min(legal_targets(view.public, view.private.my_index))


class Mute(Talker):
    def discuss(self, view):
        self.n_discuss += 1
        return ""                        # an LLM staying silent returns ""


def _events(log, kind):
    return [e for e in log.events if e["type"] == kind]


# ---------------------------------------------------------------------------
# talk_between_cuts: a full pass before every cut, ending with the cutter
# ---------------------------------------------------------------------------

def test_pass_before_every_cut_ends_with_the_cutter():
    eng = Engine(num_players=4, talk_between_cuts=True)
    _, log = eng.play_game([Talker() for _ in range(4)], seed=11)
    cuts = _events(log, "cut")
    stmts = _events(log, "statement")
    assert cuts and stmts
    for cut in cuts:
        # The 4 statements immediately preceding this cut are its pass: same round,
        # cuts_before == number of this-round cuts already made, last speaker = cutter.
        pass_stmts = [s for s in stmts
                      if s["round"] == cut["round"] and s["i"] < cut["i"]
                      and s["cuts_before"] == len([c for c in cuts
                                                   if c["round"] == cut["round"]
                                                   and c["i"] < cut["i"]])]
        assert len(pass_stmts) == 4                      # everyone spoke in the pass
        assert pass_stmts[-1]["player"] == cut["cutter"]  # the cutter has the last word
        order = [s["player"] for s in pass_stmts]
        expect = [(cut["cutter"] + 1 + k) % 4 for k in range(4)]
        assert order == expect                            # starts at the seat after the cutter


def test_no_pass_after_a_rounds_last_cut():
    eng = Engine(num_players=4, talk_between_cuts=True)
    _, log = eng.play_game([Talker() for _ in range(4)], seed=11)
    for e in _events(log, "round_start"):
        r = e["round"]
        cuts = [c for c in _events(log, "cut") if c["round"] == r]
        stmts = [s for s in _events(log, "statement") if s["round"] == r]
        if len(cuts) == 4:                                # a full, uninterrupted round
            assert len(stmts) == 16                       # 4 passes of 4, none after cut 4
            assert max(s["cuts_before"] for s in stmts) == 3


def test_silent_table_plays_clean_and_default_structure_unchanged():
    # All-mute game with the new structure: no statements, game completes.
    _, log = Engine(num_players=4, talk_between_cuts=True).play_game(
        [Mute() for _ in range(4)], seed=5)
    assert _events(log, "statement") == []
    # Default engine keeps the original one-pass structure and stamps it in game_start.
    _, log2 = Engine(num_players=4).play_game([Talker() for _ in range(4)], seed=5)
    start = log2.events[0]
    assert start["talk_between_cuts"] is False
    for r in {e["round"] for e in _events(log2, "round_start")}:
        stmts = [s for s in _events(log2, "statement") if s["round"] == r]
        assert [s["player"] for s in stmts] == [0, 1, 2, 3]   # seating order
        assert all(s["cuts_before"] == 0 for s in stmts)      # all before any cut


# ---------------------------------------------------------------------------
# Rendering: talk interleaves with cuts chronologically
# ---------------------------------------------------------------------------

def _pub(**over):
    base = dict(num_players=4, num_bad_prior={2: 1.0}, num_bom=1,
                player_names=["A", "B", "C", "D"], round_index=0, hand_size=5,
                round_start_active=4, active_wires=4,
                declarations=[1, 1, 0, 2], revealed=[0] * 4, found=[0] * 4)
    base.update(over)
    return PublicState(**base)


def test_render_interleaves_talk_and_cuts():
    pub = _pub(revealed=[0, 1, 1, 0], found=[0, 1, 0, 0],
               cut_log=[{"round": 0, "cutter": 0, "target": 1, "result": "wire",
                         "message": None},
                        {"round": 0, "cutter": 1, "target": 2, "result": "dud",
                         "message": None}],
               discussion_log=[
                   {"round": 0, "speaker": 3, "message": "opening", "cuts_before": 0},
                   {"round": 0, "speaker": 2, "message": "reaction", "cuts_before": 1}])
    text = render_agent(AgentView(public=pub, private=PrivateView(0, 0, 1, False)), "cut")
    i_open = text.index('"opening"')
    i_cut1 = text.index("A (you) cut B")
    i_react = text.index('"reaction"')
    i_cut2 = text.index("B cut C")
    assert i_open < i_cut1 < i_react < i_cut2


def test_delta_interleaves_talk_and_cuts_chronologically():
    pub = _pub(revealed=[0, 1, 0, 0], found=[0, 1, 0, 0],
               cut_log=[{"round": 0, "cutter": 0, "target": 1, "result": "wire",
                         "message": None}],
               discussion_log=[
                   {"round": 0, "speaker": 1, "message": "before", "cuts_before": 0},
                   {"round": 0, "speaker": 2, "message": "after", "cuts_before": 1}])
    view = AgentView(public=pub, private=PrivateView(3, 0, 1, False))
    cursor = cursor_after_opener(view, "declare")
    cursor["statements"] = 0                     # narrate both statements and the cut
    cursor["cuts"] = 0
    text, cur = render_session_delta(view, "discuss", cursor)
    i_before = text.index('"before"')
    i_cut = text.index("A cut B")
    i_after = text.index('"after"')
    assert i_before < i_cut < i_after            # three chunks, true order
    assert cur["statements"] == 2 and cur["cuts"] == 1
    assert "YOUR TURN TO SPEAK" in text


def test_discuss_ask_names_the_next_cutter():
    pub = _pub(current_cutter=2)
    me_other = AgentView(public=pub, private=PrivateView(0, 0, 1, False))
    me_cutter = AgentView(public=pub, private=PrivateView(2, 0, 1, False))
    assert "C cuts next" in render_agent(me_other, "discuss")
    assert "YOU cut next" in render_agent(me_cutter, "discuss")


# ---------------------------------------------------------------------------
# PanelAssistant: the public-info readout
# ---------------------------------------------------------------------------

def test_panel_none_before_any_public_evidence():
    pub = _pub(declarations=[None] * 4)          # round 1, blind declaration phase
    assert PanelAssistant().panel(pub) is None


def test_panel_symmetric_state_is_uniform_and_sums_to_num_bad():
    pub = _pub(num_players=4, declarations=[1, 1, 1, 1], round_start_active=4,
               active_wires=4)
    out = PanelAssistant().panel(pub)
    p_bad = out["P(bad)"]
    assert set(p_bad) == {"A", "B", "C", "D"}
    assert all(abs(p - 0.5) < 1e-9 for p in p_bad.values())      # 2 bad / 4 players
    assert abs(sum(p_bad.values()) - 2.0) < 1e-9
    rows = out["next cut"]
    assert set(rows) == {"A", "B", "C", "D"}
    vals = [v for row in rows.values() for v in row.values()]
    assert all(0.0 <= v <= 1.0 and not np.isnan(v) for v in vals)
    first = rows["A"]
    assert all(row == first for row in rows.values())            # symmetry
    assert "P(num_bad)" not in out                               # deal fixes the count


def test_panel_matches_direct_general_call_mid_round():
    decls = [2, 0, 1, 1]
    pub = _pub(declarations=decls, revealed=[1, 0, 0, 0], found=[1, 0, 0, 0],
               round_start_active=4, active_wires=3,
               cut_log=[{"round": 0, "cutter": 1, "target": 0, "result": "wire",
                         "message": None}])
    out = PanelAssistant().panel(pub)
    d = np.asarray(decls)
    revealed, found = np.array([1, 0, 0, 0]), np.array([1, 0, 0, 0])
    probs = gen.ProbDeclaration(d, 5, 4, 2, 1)
    probs = gen.ProbCut(d, probs, revealed, found, 5, 3, 2, 1)
    p_wire = gen.P_wire(d, probs, revealed, found, 5, 3, 2, 1)
    p_bad_direct, prob_bom = gen.Separate(probs, 2, 1)
    from state import PublicState  # noqa: F401  (imported for parity with the adapter)
    for i, name in enumerate("ABCD"):
        assert out["next cut"][name]["P(wire)"] == pytest.approx(p_wire[i], abs=5e-4)
        assert out["next cut"][name]["P(bomb)"] == pytest.approx(
            np.asarray(prob_bom).reshape(-1)[i], abs=5e-4)


def test_panel_declare_phase_uses_only_completed_rounds():
    # Round 2's blind declare: P(bad) reflects round 1, and there is no cut block.
    pub = _pub(round_index=1, hand_size=4, declarations=[None] * 4,
               round_start_active=3, active_wires=3,
               declaration_history=[[2, 0, 1, 1]],
               cut_log=[{"round": 0, "cutter": 0, "target": 1, "result": "wire",
                         "message": None}])
    out = PanelAssistant().panel(pub)
    assert "next cut" not in out
    p = list(out["P(bad)"].values())
    assert all(0.0 <= x <= 1.0 for x in p)
    assert abs(sum(p) - 2.0) < 1e-6


def test_engine_attaches_panel_only_to_flagged_players():
    seen = {}

    class Peek(Talker):
        def __init__(self, i):
            super().__init__()
            self.i = i

        def declare(self, view):
            seen.setdefault(self.i, []).append(view.assistant_panel)
            return super().declare(view)

        def discuss(self, view):
            seen.setdefault(self.i, []).append(view.assistant_panel)
            return ""

    eng = Engine(num_players=4, talk_between_cuts=True,
                 assistant=PanelAssistant(), panel_for=[0, 2])
    _, log = eng.play_game([Peek(i) for i in range(4)], seed=7)
    assert log.events[0]["panel_for"] == [0, 2]
    assert any(p is not None for p in seen[0])   # flagged players get the readout
    assert any(p is not None for p in seen[2])
    assert all(p is None for p in seen[1])       # unflagged never do
    assert all(p is None for p in seen[3])


def test_render_shows_assistant_readout_when_present():
    pub = _pub()
    panel = {"P(bad)": {"A": 0.5, "B": 0.5, "C": 0.5, "D": 0.5}}
    view = AgentView(public=pub, private=PrivateView(0, 0, 1, False),
                     assistant_panel=panel)
    text = render_agent(view, "cut")
    assert "<assistant_readout>" in text and '"P(bad)"' in text
    delta, _ = render_session_delta(view, "cut", new_session_cursor())
    assert "Assistant readout" in delta
    bare = AgentView(public=pub, private=PrivateView(0, 0, 1, False))
    assert "<assistant_readout>" not in render_agent(bare, "cut")


# ---------------------------------------------------------------------------
# Urgency bids: talk_top_k rations discuss calls to the highest bidders
# ---------------------------------------------------------------------------

class Bidder(Talker):
    """Talker whose every reply carries the same fixed urgency bid."""

    def __init__(self, urgency):
        super().__init__()
        self.last_urgency = urgency


def _pass_stmts(stmts, cuts, cut):
    """The statements of the discussion pass immediately preceding ``cut``."""
    before = len([c for c in cuts if c["round"] == cut["round"] and c["i"] < cut["i"]])
    return [s for s in stmts
            if s["round"] == cut["round"] and s["i"] < cut["i"]
            and s["cuts_before"] == before]


def test_top_k_pass_calls_only_the_highest_bidders():
    bids = (3, 9, 0, 5)
    eng = Engine(num_players=4, talk_between_cuts=True, talk_top_k=2)
    _, log = eng.play_game([Bidder(u) for u in bids], seed=11)
    assert log.events[0]["talk_top_k"] == 2
    cuts, stmts = _events(log, "cut"), _events(log, "statement")
    assert cuts
    for cut in cuts:
        ps = _pass_stmts(stmts, cuts, cut)
        assert sorted(s["player"] for s in ps) == [1, 3]      # the two highest bids
        # pass order is preserved among the selected — no reserved seat for the cutter
        expect = [p for p in [(cut["cutter"] + 1 + k) % 4 for k in range(4)] if p in (1, 3)]
        assert [s["player"] for s in ps] == expect


def test_top_k_ties_resolve_in_pass_order():
    eng = Engine(num_players=4, talk_between_cuts=True, talk_top_k=2)
    _, log = eng.play_game([Bidder(5) for _ in range(4)], seed=3)
    cuts, stmts = _events(log, "cut"), _events(log, "statement")
    for cut in cuts:
        expect = [(cut["cutter"] + 1 + k) % 4 for k in range(2)]   # first seats after cutter
        assert [s["player"] for s in _pass_stmts(stmts, cuts, cut)] == expect


def test_no_ration_by_default_even_at_bid_zero():
    _, log = Engine(num_players=4, talk_between_cuts=True).play_game(
        [Bidder(0) for _ in range(4)], seed=2)
    assert log.events[0]["talk_top_k"] is None
    cuts, stmts = _events(log, "cut"), _events(log, "statement")
    assert all(len(_pass_stmts(stmts, cuts, cut)) == 4 for cut in cuts)


def test_bids_are_logged_on_every_reply_event():
    bids = (3, 9, 0, 5)
    eng = Engine(num_players=4, talk_between_cuts=True, talk_top_k=2)
    _, log = eng.play_game([Bidder(u) for u in bids], seed=11)
    for e in _events(log, "declaration") + _events(log, "statement"):
        assert e["urgency"] == bids[e["player"]]
    for e in _events(log, "cut"):
        assert e["urgency"] == bids[e["cutter"]]


def test_llm_discuss_reply_is_message_only_with_a_bid():
    from llm import LLMAgent
    a = LLMAgent()
    a._call = lambda prompt: ('{"message": "I trust B", "urgency": 7}', "sid")
    view = AgentView(public=_pub(), private=PrivateView(0, 0, 1, False))
    assert a.discuss(view) == "I trust B"
    assert a.last_urgency == 7
    assert a.last_reasoning is None          # discuss replies carry no private reasoning


def test_llm_urgency_is_clamped_and_junk_reads_as_no_bid():
    from llm import LLMAgent
    view = AgentView(public=_pub(), private=PrivateView(0, 0, 1, False))
    a = LLMAgent()
    a._call = lambda prompt: ('{"reasoning": "r", "declaration": 1, "urgency": 15}', "sid")
    assert a.declare(view) == 1
    assert a.last_urgency == 9               # clamped into [0, 9]
    a._call = lambda prompt: ('{"message": "", "urgency": "loud"}', "sid")
    assert a.discuss(view) is None           # "" is silence
    assert a.last_urgency is None            # junk bid -> engine keeps the previous one


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
