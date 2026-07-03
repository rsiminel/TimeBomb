"""Post-game reveal & truth annotation (specs/002-host-local-game, SC-009, T029):
``reveal()`` is refused mid-game (unless exhibition), and once available exposes
roles, the dealt hands (in round_start events), and a truthful/lie verdict on every
declaration, checked against the recorded ground truth. The stored log itself is
never mutated by the annotation."""

import pytest

from tbgame.engine import TableGame, INITIAL_HAND_SIZE


def _setup(n=4, seed=5, occupant="human"):
    return {"seats": [{"name": "P%d" % i, "occupant": occupant} for i in range(n)],
            "seed": seed}


def _play_to_finish(game, declare_value=1):
    while not game.finished:
        pending = game.pending
        seat = pending.seats[0]
        if pending.kind == "declare":
            game.submit({"seat": seat, "kind": "declare", "value": declare_value})
        else:
            targets = [j for j in range(4) if j != seat]
            legal = [j for j in targets
                     if game.public_state().revealed[j] < game.public_state().hand_size]
            game.submit({"seat": seat, "kind": "cut", "value": legal[0]})


def test_reveal_refused_before_finish():
    game = TableGame(_setup())
    with pytest.raises(RuntimeError):
        game.reveal()


def test_reveal_always_available_in_exhibition():
    game = TableGame(_setup(occupant="solver_bot"))
    reveal = game.reveal()          # not finished, but nothing is hidden (Q5)
    assert len(reveal["roles"]) == 4


def test_reveal_after_finish_exposes_roles_hands_and_outcome():
    game = TableGame(_setup(seed=5))
    _play_to_finish(game)
    reveal = game.reveal()

    assert sorted(set(reveal["roles"]) | {0, 1}) == [0, 1]
    assert len(reveal["roles"]) == 4
    assert reveal["outcome"]["reason"] in ("all wires cut", "bomb detonated",
                                           "out of time")

    round_starts = [e for e in reveal["events"] if e["type"] == "round_start"]
    assert round_starts, "the deal must be part of the revealed history"
    first = round_starts[0]
    assert len(first["wires"]) == 4 and len(first["bombs"]) == 4
    assert sum(first["wires"]) == 4          # round 0: one safe wire per player
    assert sum(first["bombs"]) == 1
    assert all(0 <= w <= INITIAL_HAND_SIZE for w in first["wires"])


def test_every_declaration_annotated_against_recorded_truth():
    game = TableGame(_setup(seed=9))
    _play_to_finish(game, declare_value=1)
    reveal = game.reveal()

    declarations = [e for e in reveal["events"] if e["type"] == "declaration"]
    assert declarations
    for event in declarations:
        assert event["lie"] == (event["declared"] != event["true_wires"])

    # The truth-annotation is on the returned copy only; the stored log is pristine.
    assert all("lie" not in e for e in game.events if e["type"] == "declaration")


def test_annotation_flags_actual_lies():
    """Force a lie: declare 0 for everyone; whoever holds wires is lying."""
    game = TableGame(_setup(seed=13))
    _play_to_finish(game, declare_value=0)
    reveal = game.reveal()
    first_round = [e for e in reveal["events"]
                   if e["type"] == "declaration" and e["round"] == 0]
    assert len(first_round) == 4
    # Round 0 deals exactly one wire per seat count spread over hands; with all
    # declaring 0, exactly the wire-holders are flagged.
    round0 = [e for e in reveal["events"] if e["type"] == "round_start"][0]
    for event in first_round:
        assert event["lie"] == (round0["wires"][event["player"]] != 0)
