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


class TestMultiRoundParity:
    def test_two_round_accumulation(self):
        """Round-2 belief conditions on round-1 evidence (US2 / FR-008): it equals the
        direct solver computation with round 1's RoundLogU accumulated, and differs
        from a fresh game fed only the round-2 entries."""
        names = ["Alice", "Bob", "Clara", "Darryl"]
        d1 = [2, 1, 1, 1]
        round1_cuts = [(0, "safe"), (1, "nothing"), (2, "safe"), (3, "nothing")]
        events = [{"type": "declarations", "values": d1}]
        events += [{"type": "cut", "player": p, "result": r} for p, r in round1_cuts]
        d2 = [1, 0, 1, 1]
        events.append({"type": "declarations", "values": d2})
        events.append({"type": "cut", "player": 1, "result": "safe"})
        out = replay_record(record(names, events=events))

        # Direct solver: round 1 folded at its final cut state (hand 5, started at 4
        # wires, ended at 2), then the round-2 readout at hand 4.
        prior_b = gen.NUM_BAD_PRIOR(4)
        log_u = {
            b: gen.RoundLogU(np.array(d1), np.array([1, 1, 1, 1]),
                             np.array([1, 0, 1, 0]), 5, 4, 2, b, 1)
            for b in prior_b
        }
        expected = play_readout(d2, [0, 1, 0, 0], [0, 1, 0, 0], hand_size=4,
                                total_active=2, active_wires=1, prior_b=prior_b,
                                num_bom=1, log_u_by_b=log_u)
        assert_belief_matches(out["belief"], *expected)

        # And the carried evidence matters: a fresh game given only round-2 entries
        # (its own hand size and wire count) believes something else.
        fresh = play_readout(d2, [0, 1, 0, 0], [0, 1, 0, 0], hand_size=4,
                             total_active=2, active_wires=1, prior_b=prior_b,
                             num_bom=1, log_u_by_b=None)
        assert out["belief"]["pBad"] != [float(p) for p in fresh[0]]

    def test_between_rounds_belief_has_no_panel(self):
        """After a round closes, pBad/pNumBad come from accumulated evidence alone."""
        names = ["Alice", "Bob", "Clara", "Darryl"]
        d1 = [1, 1, 1, 1]
        events = [{"type": "declarations", "values": d1}]
        events += [{"type": "cut", "player": p, "result": "nothing"} for p in range(4)]
        out = replay_record(record(names, events=events))
        assert out["state"]["awaiting"] == "declarations"
        assert out["belief"]["panel"] is None
        prior_b = gen.NUM_BAD_PRIOR(4)
        log_u = {
            b: gen.RoundLogU(np.array(d1), np.ones(4, dtype=int),
                             np.zeros(4, dtype=int), 5, 4, 4, b, 1)
            for b in prior_b
        }
        p_bad, p_num_bad, _ = gen.JointBadBelief(log_u, prior_b)
        assert out["belief"]["pBad"] == [float(p) for p in p_bad]
        assert out["belief"]["pNumBad"] == {str(b): float(p) for b, p in p_num_bad.items()}


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q", "-n0"]))
