"""Endpoint behaviour tests (tasks T010/T017/T020): response shapes, status codes,
seat order, the approx flag, game end, and warning discipline — no probability values
asserted here (that's test_solver_parity.py)."""
import pytest


def record(n=4, bomb=True, override=None, events=None):
    names = ["Alice", "Bob", "Clara", "Darryl", "Eve", "Fred", "Gina", "Hugo"][:n]
    return {
        "setup": {"players": names, "bomb": bomb, "numBadOverride": override},
        "events": events or [],
    }


def decl(*values):
    return {"type": "declarations", "values": list(values)}


def cut(player, result):
    return {"type": "cut", "player": player, "result": result}


class TestPanelResponses:
    def test_belief_null_before_declarations(self, client):
        body = client.post("/api/panel", json=record()).get_json()
        assert body["belief"] is None
        assert body["state"]["awaiting"] == "declarations"

    def test_panel_present_after_declarations(self, client):
        body = client.post("/api/panel", json=record(events=[decl(1, 1, 1, 1)])).get_json()
        belief = body["belief"]
        assert len(belief["pBad"]) == 4
        assert len(belief["panel"]) == 4
        assert set(belief["panel"][0]) == {"pSafe", "pBomb", "onePly", "horizon", "noCards"}

    def test_seat_order_preserved(self, client):
        """The panel is positional: row i is seat i, whatever the numbers say."""
        events = [decl(3, 0, 1, 2), cut(1, "safe")]
        body = client.post("/api/panel", json=record(events=events)).get_json()
        assert body["state"]["revealed"] == [0, 1, 0, 0]  # seat 1, not sorted anywhere
        assert len(body["belief"]["panel"]) == 4

    def test_approx_flag_when_capped(self, client):
        """Round start at N=4: 4 cuts left vs cap 3 -> capped; after one cut -> exact."""
        events = [decl(1, 1, 1, 1)]
        body = client.post("/api/panel", json=record(events=events)).get_json()
        assert body["belief"]["approx"] is True
        assert body["belief"]["maxDepth"] == 3
        events.append(cut(0, "nothing"))
        body = client.post("/api/panel", json=record(events=events)).get_json()
        assert body["belief"]["approx"] is False

    def test_uncertain_deal_exposes_both_counts(self, client):
        body = client.post("/api/panel", json=record(events=[decl(1, 1, 1, 1)])).get_json()
        assert set(body["belief"]["pNumBad"]) == {"1", "2"}  # N=4 official deal

    def test_malformed_json_is_400(self, client):
        resp = client.post("/api/panel", data="nonsense", content_type="application/json")
        assert resp.status_code == 400

    def test_invalid_record_is_422_with_event_index(self, client):
        resp = client.post("/api/panel", json=record(events=[cut(0, "safe")]))
        assert resp.status_code == 422
        assert resp.get_json()["eventIndex"] == 0

    def test_static_page_served(self, client):
        assert client.get("/").status_code == 200


def full_round(n, declarations, results):
    events = [decl(*declarations)]
    events += [cut(p, results[p]) for p in range(n)]
    return events


class TestRoundsAndGameEnd:
    def test_round_rollover_fields(self, client):
        events = full_round(4, [1, 1, 1, 1], ["safe", "nothing", "nothing", "nothing"])
        body = client.post("/api/panel", json=record(events=events)).get_json()
        state = body["state"]
        assert state["round"] == 2
        assert state["handSize"] == 4
        assert state["awaiting"] == "declarations"
        assert body["belief"]["panel"] is None  # between rounds: no live cut stats

    def test_game_over_bomb(self, client):
        events = [decl(1, 1, 1, 1), cut(2, "bomb")]
        body = client.post("/api/panel", json=record(events=events)).get_json()
        assert body["state"]["gameOver"] == {"winner": "bad", "reason": "bomb"}
        assert body["state"]["awaiting"] == "over"
        assert body["belief"] is not None  # final belief still shown

    def test_game_over_wires(self, client):
        events = [decl(1, 1, 1, 1)] + [cut(p, "safe") for p in range(4)]
        body = client.post("/api/panel", json=record(events=events)).get_json()
        assert body["state"]["gameOver"] == {"winner": "good", "reason": "wires"}

    def test_game_over_time(self, client):
        events = []
        for _ in range(4):
            events += full_round(4, [0, 0, 0, 0],
                                 ["nothing", "nothing", "nothing", "nothing"])
        body = client.post("/api/panel", json=record(events=events)).get_json()
        assert body["state"]["gameOver"] == {"winner": "bad", "reason": "time"}

    def test_no_entries_after_game_over(self, client):
        events = [decl(1, 1, 1, 1), cut(2, "bomb"), cut(0, "safe")]
        resp = client.post("/api/panel", json=record(events=events))
        assert resp.status_code == 422
        assert resp.get_json()["eventIndex"] == 2


class TestWarningDiscipline:
    """Jointly impossible entries are legal input: 200 with a warning, never 422, and
    each warning fires once — on the entry that raised it (SC-004, contract inv. 3)."""

    IMPOSSIBLE_DECLS = [5, 5, 5, 5]  # >=2 good players would jointly hold 10 of 4 wires

    def test_impossible_declarations_warn_once(self, client):
        events = [decl(*self.IMPOSSIBLE_DECLS)]
        resp = client.post("/api/panel", json=record(events=events))
        assert resp.status_code == 200
        body = resp.get_json()
        assert [w["code"] for w in body["warnings"]] == ["impossible_declarations"]
        assert body["belief"] is not None  # fallback belief still renders

        # The first cut on a degenerate round also reads as impossible — exactly what
        # the interactive Play loop reports — but each warning fires at most once per
        # round: the second cut is quiet.
        events.append(cut(0, "nothing"))
        body = client.post("/api/panel", json=record(events=events)).get_json()
        assert [w["code"] for w in body["warnings"]] == ["impossible_cut"]
        events.append(cut(1, "nothing"))
        body = client.post("/api/panel", json=record(events=events)).get_json()
        assert body["warnings"] == []

    def test_impossible_cut_warns_and_continues(self, client):
        # One bad guy, no bomb, declarations [3,0,0,0] with 4 wires live: the lone
        # liar holds exactly one wire, so ONE safe cut on a zero-declarer is
        # explicable (they're the bad guy) — but a second, on a different
        # zero-declarer, cannot be.
        events = [decl(3, 0, 0, 0), cut(1, "safe")]
        body = client.post("/api/panel",
                           json=record(bomb=False, override=1, events=events)).get_json()
        assert body["warnings"] == []

        events.append(cut(2, "safe"))
        resp = client.post("/api/panel",
                           json=record(bomb=False, override=1, events=events))
        assert resp.status_code == 200  # impossible input is a warning, never a 422
        body = resp.get_json()
        assert [w["code"] for w in body["warnings"]] == ["impossible_cut"]
        assert body["belief"] is not None  # the assistant keeps rendering (SC-004)
        assert all(0.0 <= p <= 1.0 for p in body["belief"]["pBad"])

    def test_undo_restores_the_pre_mistake_response(self, client):
        """Client undo = truncate the log: the server response for the shorter log is
        exactly the pre-mistake response (full-history undo, FR-011)."""
        good = record(events=[decl(1, 1, 1, 1), cut(0, "safe")])
        before = client.post("/api/panel", json=good).get_json()

        with_mistake = record(events=good["events"] + [cut(3, "nothing")])
        client.post("/api/panel", json=with_mistake)

        undone = client.post("/api/panel", json=good).get_json()
        assert undone == before


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q", "-n0"]))
