"""Contract tests for the hosted-game HTTP API (specs/002-host-local-game,
contracts/api.md): POST/GET/DELETE /api/game, /api/game/intent -- happy path, 422 on
an illegal intent, 409 on a version conflict, 409 on a second game."""


def _setup(n=4, occupant="human", seed=1):
  return {"seats": [{"name": "P%d" % i, "occupant": occupant} for i in range(n)],
          "seed": seed}


def test_create_get_delete_lifecycle(client):
  resp = client.post("/api/game", json=_setup())
  assert resp.status_code == 201
  assert resp.get_json()["version"] == 1

  resp = client.get("/api/game")
  assert resp.status_code == 200
  body = resp.get_json()
  assert body["version"] == 1
  assert body["phase"] == "awaiting_declarations"
  assert body["pending"] == {"kind": "declare", "seats": [0, 1, 2, 3]}
  assert body["numPlayers"] == 4

  resp = client.delete("/api/game")
  assert resp.status_code == 204
  assert client.get("/api/game").status_code == 404


def test_second_game_is_409_until_abandoned(client):
  assert client.post("/api/game", json=_setup()).status_code == 201
  resp = client.post("/api/game", json=_setup())
  assert resp.status_code == 409
  assert "error" in resp.get_json()

  client.delete("/api/game")
  assert client.post("/api/game", json=_setup()).status_code == 201


def test_invalid_setup_wiring_is_422(client):
  # Setup *validation* is covered by test_setup_rejects_* in tbgame/tests/
  # test_table_game.py; this only checks the SetupError -> 422 wiring.
  resp = client.post("/api/game", json={"seats": [{"name": "only one", "occupant": "human"}]})
  assert resp.status_code == 422
  assert "error" in resp.get_json()


def test_intent_happy_path(client):
  client.post("/api/game", json=_setup())
  resp = client.post("/api/game/intent",
                     json={"seat": 0, "kind": "declare", "value": 1, "version": 1})
  assert resp.status_code == 200
  body = resp.get_json()
  assert body["version"] == 2
  assert any(e["type"] == "declaration" for e in body["events"])


def test_intent_illegal_is_422_with_engine_reason(client):
  client.post("/api/game", json=_setup())
  resp = client.post("/api/game/intent",
                     json={"seat": 0, "kind": "declare", "value": 99, "version": 1})
  assert resp.status_code == 422
  assert resp.get_json()["error"]   # the engine's own reason, forwarded verbatim


def test_intent_stale_version_is_409(client):
  client.post("/api/game", json=_setup())
  resp = client.post("/api/game/intent",
                     json={"seat": 0, "kind": "declare", "value": 1, "version": 99})
  assert resp.status_code == 409


def test_intent_wrong_seat_or_kind_is_422(client):
  client.post("/api/game", json=_setup())
  resp = client.post("/api/game/intent",
                     json={"seat": 0, "kind": "cut", "value": 1, "version": 1})
  assert resp.status_code == 422


def test_no_active_game_is_404(client):
  assert client.get("/api/game").status_code == 404
  assert client.delete("/api/game").status_code == 404
  resp = client.post("/api/game/intent",
                     json={"seat": 0, "kind": "declare", "value": 1, "version": 1})
  assert resp.status_code == 404
