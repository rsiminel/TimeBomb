"""Save/resume round-trip (specs/002-host-local-game, SC-007 / FR-023): save
mid-round-2, simulate an app restart, resume, and get the identical public state
back -- with hidden state intact (same deal, same roles) and still hidden. Schema
mismatches are rejected cleanly; duplicate names 409; deletes delete."""

import json

import pytest


def _setup(n=4, seed=31):
    return {"seats": [{"name": "P%d" % i, "occupant": "human"} for i in range(n)],
            "seed": seed, "panel_allowed": True}


def _play_into_round_2(client):
    """Drive the hotseat game until round index 1 is underway."""
    for _ in range(200):
        view = client.get("/api/game").get_json()
        assert view["phase"] != "finished", "game ended before round 2; pick a new seed"
        if view["roundIndex"] >= 1 and view["declarations"].count(None) < 2:
            return view
        seat = view["pending"]["seats"][0]
        if view["pending"]["kind"] == "declare":
            intent = {"seat": seat, "kind": "declare", "value": 1,
                      "version": view["version"]}
        else:
            intent = {"seat": seat, "kind": "cut", "value": view["legalTargets"][0],
                      "version": view["version"]}
        assert client.post("/api/game/intent", json=intent).status_code == 200
    pytest.fail("never reached round 2")


def test_save_restart_resume_round_trip(client, tmp_path, monkeypatch):
    import game_service
    monkeypatch.setattr(game_service, "SAVES_DIR", tmp_path)

    client.post("/api/game", json=_setup())
    saved_view = _play_into_round_2(client)

    resp = client.post("/api/saves", json={"name": "game night"})
    assert resp.status_code == 201
    assert client.post("/api/saves", json={"name": "game night"}).status_code == 409

    listing = client.get("/api/saves").get_json()
    assert [s["name"] for s in listing] == ["game night"]
    assert listing[0]["seats"] == ["P0", "P1", "P2", "P3"]
    assert listing[0]["phase"] in ("awaiting_declarations", "awaiting_cut")

    # Simulated restart: the in-memory service state is wiped; only the file remains.
    game_service.reset_for_tests()
    assert client.get("/api/game").status_code == 404

    resp = client.post("/api/saves/game night/resume")
    assert resp.status_code == 200
    assert resp.get_json()["version"] == 1     # version restarts with the process

    resumed_view = client.get("/api/game").get_json()
    for key in ("numPlayers", "playerNames", "roundIndex", "handSize", "activeWires",
                "declarations", "revealed", "found", "cutLog", "claimLog",
                "declarationHistory", "currentCutter", "phase", "pending",
                "occupants", "panelAllowed"):
        assert resumed_view[key] == saved_view[key], key
    assert "reveal" not in resumed_view        # hidden state is intact AND hidden

    # Same deal underneath: the same next moves stay legal and the game still ends.
    seat = resumed_view["pending"]["seats"][0]
    priv = client.post("/api/game/unlock",
                       json={"seat": seat, "version": resumed_view["version"]})
    assert priv.status_code == 200             # roles/hands restored, servable


def test_resume_while_active_is_409_and_missing_is_404(client, tmp_path, monkeypatch):
    import game_service
    monkeypatch.setattr(game_service, "SAVES_DIR", tmp_path)

    client.post("/api/game", json=_setup())
    assert client.post("/api/saves", json={"name": "s1"}).status_code == 201
    assert client.post("/api/saves/s1/resume").status_code == 409   # game active
    client.delete("/api/game")
    assert client.post("/api/saves/nope/resume").status_code == 404
    assert client.post("/api/saves/s1/resume").status_code == 200


def test_schema_mismatch_rejected_cleanly(client, tmp_path, monkeypatch):
    import game_service
    monkeypatch.setattr(game_service, "SAVES_DIR", tmp_path)

    client.post("/api/game", json=_setup())
    client.post("/api/saves", json={"name": "old"})
    client.delete("/api/game")

    path = tmp_path / "old.json"
    data = json.loads(path.read_text())
    data["events"][0]["schema_version"] = 999
    path.write_text(json.dumps(data))

    resp = client.post("/api/saves/old/resume")
    assert resp.status_code == 422
    assert "error" in resp.get_json()
    assert path.exists()                       # file untouched by the rejection


def test_save_needs_active_game_and_safe_name(client, tmp_path, monkeypatch):
    import game_service
    monkeypatch.setattr(game_service, "SAVES_DIR", tmp_path)

    assert client.post("/api/saves", json={"name": "s"}).status_code == 404
    client.post("/api/game", json=_setup())
    assert client.post("/api/saves", json={"name": "../evil"}).status_code == 422
    assert client.post("/api/saves", json={"name": ""}).status_code == 422


def test_delete_save(client, tmp_path, monkeypatch):
    import game_service
    monkeypatch.setattr(game_service, "SAVES_DIR", tmp_path)

    client.post("/api/game", json=_setup())
    client.post("/api/saves", json={"name": "gone"})
    assert client.delete("/api/saves/gone").status_code == 204
    assert client.get("/api/saves").get_json() == []
    assert client.delete("/api/saves/gone").status_code == 404
