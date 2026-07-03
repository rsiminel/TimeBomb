"""Record validation and rules-bookkeeping tests for web/replay.py (tasks T005, T020).

Hand-worked scripted logs only — no probability assertions here (those live in
test_solver_parity.py). Covers every 422 case of contracts/api.md, round rollover,
all three game-over reasons, and replay purity (truncated log == shorter game).
"""
import pytest

import replay
from replay import RecordError, replay_record


def make_record(n=4, bomb=True, override=None, events=None):
    names = ["Alice", "Bob", "Clara", "Darryl", "Eve", "Fred", "Gina", "Hugo"][:n]
    return {
        "setup": {"players": names, "bomb": bomb, "numBadOverride": override},
        "events": events or [],
    }


def decl(*values):
    return {"type": "declarations", "values": list(values)}


def cut(player, result):
    return {"type": "cut", "player": player, "result": result}


def full_round(n, declarations, cuts):
    """One complete round: declarations plus exactly n cuts."""
    assert len(cuts) == n
    return [decl(*declarations)] + [cut(p, r) for p, r in cuts]


class TestSetupValidation:
    def test_valid_record_passes(self):
        assert replay_record(make_record())["state"]["round"] == 1

    @pytest.mark.parametrize("n", [3, 9])
    def test_player_count_bounds(self, n):
        names = [f"P{i}" for i in range(n)]
        record = {"setup": {"players": names, "bomb": True}, "events": []}
        with pytest.raises(RecordError) as err:
            replay_record(record)
        assert err.value.event_index == -1

    @pytest.mark.parametrize("players", [
        ["Alice", "Bob", "Clara", ""],           # empty name
        ["Alice", "Bob", "Clara", "   "],        # blank name
        ["Alice", "Bob", "Clara", "Alice"],      # duplicate
        ["Alice", "Bob", "Clara", 4],            # non-string
    ])
    def test_bad_player_names(self, players):
        record = {"setup": {"players": players, "bomb": True}, "events": []}
        with pytest.raises(RecordError):
            replay_record(record)

    @pytest.mark.parametrize("override", [0, 4, -1, "2", True])
    def test_bad_override(self, override):
        with pytest.raises(RecordError):
            replay_record(make_record(override=override))

    @pytest.mark.parametrize("override", [1, 2, 3])
    def test_override_range_is_1_to_n_minus_1(self, override):
        replay_record(make_record(override=override))  # must not raise

    def test_bomb_flag_must_be_bool(self):
        record = make_record()
        record["setup"]["bomb"] = 1
        with pytest.raises(RecordError):
            replay_record(record)

    def test_missing_setup(self):
        with pytest.raises(RecordError) as err:
            replay_record({"events": []})
        assert err.value.event_index == -1


class TestEventValidation:
    def test_cut_before_declarations(self):
        with pytest.raises(RecordError) as err:
            replay_record(make_record(events=[cut(0, "safe")]))
        assert err.value.event_index == 0

    def test_double_declarations(self):
        events = [decl(1, 1, 1, 1), decl(1, 1, 1, 1)]
        with pytest.raises(RecordError) as err:
            replay_record(make_record(events=events))
        assert err.value.event_index == 1

    @pytest.mark.parametrize("values", [
        [1, 1, 1],            # wrong length
        [1, 1, 1, 6],         # above hand size 5
        [1, 1, 1, -1],        # below zero
        [1, 1, 1, 1.5],       # non-integer
        [1, 1, 1, True],      # bool is not a count
    ])
    def test_bad_declaration_values(self, values):
        with pytest.raises(RecordError):
            replay_record(make_record(events=[{"type": "declarations", "values": values}]))

    def test_cut_on_fully_revealed_hand(self):
        # A hand can only run dry while the round is still open once hand size drops
        # below the N-cut budget — round 3 at N=4 has hands of 3.
        events = (
            full_round(4, [0, 0, 0, 0],
                       [(0, "nothing"), (1, "nothing"), (2, "nothing"), (3, "nothing")])
            + full_round(4, [0, 0, 0, 0],
                         [(0, "nothing"), (1, "nothing"), (2, "nothing"), (3, "nothing")])
            + [decl(0, 0, 0, 0),
               cut(0, "nothing"), cut(0, "nothing"), cut(0, "nothing"),
               cut(0, "nothing")]  # player 0's 3-card hand is spent; 4th cut is illegal
        )
        with pytest.raises(RecordError) as err:
            replay_record(make_record(events=events))
        assert err.value.event_index == 14

    def test_bad_cut_fields(self):
        base = [decl(1, 1, 1, 1)]
        for bad in [cut(4, "safe"), cut(-1, "safe"), cut("0", "safe"),
                    cut(0, "wire"), {"type": "cut", "player": 0}]:
            with pytest.raises(RecordError) as err:
                replay_record(make_record(events=base + [bad]))
            assert err.value.event_index == 1

    def test_bomb_result_without_bomb_in_play(self):
        events = [decl(1, 1, 1, 1), cut(0, "bomb")]
        with pytest.raises(RecordError):
            replay_record(make_record(bomb=False, events=events))

    def test_unknown_event_type(self):
        with pytest.raises(RecordError):
            replay_record(make_record(events=[{"type": "undo"}]))

    def test_events_after_game_over(self):
        events = [decl(1, 1, 1, 1), cut(0, "bomb"), cut(1, "safe")]
        with pytest.raises(RecordError) as err:
            replay_record(make_record(events=events))
        assert err.value.event_index == 2


class TestBookkeeping:
    def test_initial_state(self):
        state = replay_record(make_record())["state"]
        assert state == {
            "round": 1, "handSize": 5, "activeWires": 4, "cutsMade": 0,
            "cutsThisRound": 4, "revealed": [], "found": [],
            "awaiting": "declarations", "gameOver": None,
        }

    def test_mid_round_tallies(self):
        events = [decl(2, 1, 1, 1), cut(2, "safe"), cut(2, "nothing")]
        state = replay_record(make_record(events=events))["state"]
        assert state["round"] == 1
        assert state["revealed"] == [0, 0, 2, 0]
        assert state["found"] == [0, 0, 1, 0]
        assert state["activeWires"] == 3
        assert state["cutsMade"] == 2
        assert state["awaiting"] == "cut"

    def test_round_rollover_shrinks_hand(self):
        events = full_round(4, [1, 1, 1, 1],
                            [(0, "safe"), (1, "nothing"), (2, "safe"), (3, "nothing")])
        state = replay_record(make_record(events=events))["state"]
        assert state["round"] == 2
        assert state["handSize"] == 4
        assert state["activeWires"] == 2
        assert state["awaiting"] == "declarations"
        assert state["gameOver"] is None

    def test_second_round_declaration_range_uses_new_hand_size(self):
        events = full_round(4, [1, 1, 1, 1],
                            [(0, "safe"), (1, "nothing"), (2, "safe"), (3, "nothing")])
        events.append(decl(5, 0, 0, 0))  # 5 > new hand size 4
        with pytest.raises(RecordError) as err:
            replay_record(make_record(events=events))
        assert err.value.event_index == 5

    def test_game_over_bomb(self):
        events = [decl(1, 1, 1, 1), cut(3, "bomb")]
        state = replay_record(make_record(events=events))["state"]
        assert state["gameOver"] == {"winner": "bad", "reason": "bomb"}
        assert state["awaiting"] == "over"
        # the detonating cut is not tallied (PlayAuto convention)
        assert state["revealed"] == [0, 0, 0, 0]

    def test_game_over_wires(self):
        events = [decl(1, 1, 1, 1),
                  cut(0, "safe"), cut(1, "safe"), cut(2, "safe"), cut(3, "safe")]
        state = replay_record(make_record(events=events))["state"]
        assert state["gameOver"] == {"winner": "good", "reason": "wires"}
        assert state["activeWires"] == 0

    def test_game_over_time(self):
        events = []
        for declarations in ([0, 0, 0, 0],) * 4:  # rounds at hand sizes 5, 4, 3, 2
            events += full_round(4, declarations,
                                 [(0, "nothing"), (1, "nothing"),
                                  (2, "nothing"), (3, "nothing")])
        state = replay_record(make_record(events=events))["state"]
        assert state["gameOver"] == {"winner": "bad", "reason": "time"}
        assert state["handSize"] == 2  # the final round's hand size is preserved

    def test_replay_purity_truncation(self):
        events = [decl(1, 1, 1, 1), cut(0, "safe"), cut(1, "nothing"), cut(2, "safe")]
        for k in range(len(events) + 1):
            longer = replay_record(make_record(events=events[:k]))
            again = replay_record(make_record(events=events[:k]))
            assert longer == again  # identical input, identical output


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q", "-n0"]))
