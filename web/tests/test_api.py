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


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q", "-n0"]))
