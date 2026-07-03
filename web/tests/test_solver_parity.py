"""Solver-parity tests (SC-003, tasks T009/T015): every number in the API's ``belief``
block must exactly equal a direct ``timebomb/General.py`` computation for the same
scripted game — the web layer adds bookkeeping, never arithmetic (Constitution I).

The expected values are derived here by hand-scripting General.Play's call pattern
with hardcoded game quantities (hand sizes, wire counts, tallies) — independent of
web/replay.py's bookkeeping code.
"""
import numpy as np
import pytest

import General as gen
from replay import DEPTH_CAP, replay_record


def record(names, bomb=True, override=None, events=()):
    return {
        "setup": {"players": names, "bomb": bomb, "numBadOverride": override},
        "events": list(events),
    }


def play_readout(decls, revealed, found, hand_size, total_active, active_wires,
                 prior_b, num_bom, log_u_by_b=None):
    """General.Play's per-cut readout, scripted directly against the solver."""
    n = len(decls)
    decls = np.asarray(decls, dtype=int)
    revealed = np.asarray(revealed, dtype=int)
    found = np.asarray(found, dtype=int)
    candidate_bs = list(prior_b)
    b0 = candidate_bs[0]
    if log_u_by_b is None:
        log_u_by_b = {b: np.zeros([n] * b) for b in candidate_bs}
    probs = gen.ProbDeclaration(decls, hand_size, total_active, b0, num_bom)
    probs = gen.ProbCut(decls, probs, revealed, found, hand_size, active_wires,
                        b0, num_bom)
    provisional = {
        b: log_u_by_b[b] + gen.RoundLogU(decls, revealed, found, hand_size,
                                         total_active, active_wires, b, num_bom)
        for b in candidate_bs
    }
    p_bad, p_num_bad, _ = gen.JointBadBelief(provisional, prior_b)
    panel = gen.CutPanel(decls, probs, revealed, found, hand_size, active_wires,
                         b0, num_bom, max_depth=DEPTH_CAP[n])
    return p_bad, p_num_bad, panel


def assert_belief_matches(belief, p_bad, p_num_bad, panel):
    assert belief["pBad"] == [float(p) for p in p_bad]
    assert belief["pNumBad"] == {str(b): float(p) for b, p in p_num_bad.items()}
    for i, row in enumerate(belief["panel"]):
        if row.get("noCards"):
            assert np.all(np.isnan(panel[i]))
        else:
            assert [row["pSafe"], row["pBomb"], row["onePly"], row["horizon"]] == \
                [float(v) for v in panel[i]]


class TestSingleRoundParity:
    def test_n4_uncertain_deal_after_declarations_and_cuts(self):
        """N=4: the official deal leaves num_bad uncertain (1 or 2, ADR 0008)."""
        names = ["Alice", "Bob", "Clara", "Darryl"]
        events = [{"type": "declarations", "values": [2, 1, 1, 1]}]
        cuts = [(2, "safe"), (0, "nothing")]
        revealed = [0, 0, 0, 0]
        found = [0, 0, 0, 0]
        wires = 4
        # after declarations only, then after each cut
        checkpoints = [(list(revealed), list(found), wires)]
        for player, result in cuts:
            events.append({"type": "cut", "player": player, "result": result})
            revealed[player] += 1
            if result == "safe":
                found[player] += 1
                wires -= 1
            checkpoints.append((list(revealed), list(found), wires))
        for k, (rev, fnd, act) in enumerate(checkpoints):
            out = replay_record(record(names, events=events[:k + 1]))
            expected = play_readout([2, 1, 1, 1], rev, fnd, hand_size=5,
                                    total_active=4, active_wires=act,
                                    prior_b=gen.NUM_BAD_PRIOR(4), num_bom=1)
            assert_belief_matches(out["belief"], *expected)

    def test_n5_fixed_deal(self):
        names = ["A", "B", "C", "D", "E"]
        events = [{"type": "declarations", "values": [1, 2, 1, 0, 1]},
                  {"type": "cut", "player": 3, "result": "nothing"}]
        out = replay_record(record(names, events=events))
        expected = play_readout([1, 2, 1, 0, 1], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0],
                                hand_size=5, total_active=5, active_wires=5,
                                prior_b=gen.NUM_BAD_PRIOR(5), num_bom=1)
        assert_belief_matches(out["belief"], *expected)

    def test_num_bad_override(self):
        """A fixed manual count replaces the official prior entirely."""
        names = ["Alice", "Bob", "Clara", "Darryl"]
        events = [{"type": "declarations", "values": [2, 1, 1, 1]},
                  {"type": "cut", "player": 1, "result": "safe"}]
        out = replay_record(record(names, override=2, events=events))
        expected = play_readout([2, 1, 1, 1], [0, 1, 0, 0], [0, 1, 0, 0],
                                hand_size=5, total_active=4, active_wires=3,
                                prior_b={2: 1.0}, num_bom=1)
        assert_belief_matches(out["belief"], *expected)
        assert list(out["belief"]["pNumBad"]) == ["2"]

    def test_no_bomb_game(self):
        """bomb: false runs the num_bom=0 path; the pBomb column is exactly zero."""
        names = ["Alice", "Bob", "Clara", "Darryl"]
        events = [{"type": "declarations", "values": [1, 1, 2, 1]},
                  {"type": "cut", "player": 0, "result": "safe"}]
        out = replay_record(record(names, bomb=False, events=events))
        expected = play_readout([1, 1, 2, 1], [1, 0, 0, 0], [1, 0, 0, 0],
                                hand_size=5, total_active=4, active_wires=3,
                                prior_b=gen.NUM_BAD_PRIOR(4), num_bom=0)
        assert_belief_matches(out["belief"], *expected)
        assert all(row["pBomb"] == 0.0 for row in out["belief"]["panel"])


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q", "-n0"]))
