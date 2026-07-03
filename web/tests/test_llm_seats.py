"""LLM seats with a stubbed agent (specs/002-host-local-game, FR-017/018, T036):
key gating at setup, settings endpoints that never echo the key, a failing agent
pausing the game (`paused_llm`), the retry and substitute recovery paths, and
resuming an LLM-seat save without a key routing through substitution."""

import json
import time

import pytest


@pytest.fixture()
def llm_env(tmp_path, monkeypatch):
    """No env key; instance dir and saves dir isolated to tmp."""
    import game_service
    monkeypatch.delenv(game_service.LLM_KEY_ENV, raising=False)
    monkeypatch.setattr(game_service, "INSTANCE_DIR", tmp_path / "instance")
    monkeypatch.setattr(game_service, "SAVES_DIR", tmp_path / "saves")
    monkeypatch.setattr(game_service, "AI_POLL_INTERVAL_S", 0.02)
    return game_service


def _setup(occupants, seed=3):
    return {"seats": [{"name": "P%d" % i, "occupant": occ}
                      for i, occ in enumerate(occupants)], "seed": seed}


def _wait(check, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = check()
        if value:
            return value
        time.sleep(0.05)
    pytest.fail("timed out waiting")


class _FailingLLM:
    """Stands in for the arena LLM agent: every decision fails."""
    last_error = "stub: model call failed"

    def declare(self, view):
        return None

    def choose_cut(self, view):
        return None

    def claim(self, view):
        return None


class _FlakyLLM(_FailingLLM):
    """Fails until told to behave, then declares 1 / cuts the first legal seat."""
    def __init__(self):
        self.works = False

    def declare(self, view):
        return 1 if self.works else None

    def choose_cut(self, view):
        from tbgame.state import legal_targets
        return legal_targets(view.public, view.private.my_index)[0] if self.works else None


def test_llm_seats_rejected_at_setup_without_key(client, llm_env):
    resp = client.post("/api/game",
                       json=_setup(["human", "llm:test", "human", "human"]))
    assert resp.status_code == 422
    assert "key" in resp.get_json()["error"]


def test_settings_endpoint_env_wins_and_never_echoes(client, llm_env, monkeypatch):
    assert client.get("/api/settings/llm").get_json() == {"configured": False,
                                                          "source": None}
    assert client.put("/api/settings/llm", json={"api_key": "sk-file-secret"}).status_code == 204
    body = client.get("/api/settings/llm").get_json()
    assert body == {"configured": True, "source": "file"}
    assert "sk-file-secret" not in json.dumps(body)

    monkeypatch.setenv(llm_env.LLM_KEY_ENV, "sk-env-secret")
    assert client.get("/api/settings/llm").get_json()["source"] == "env"
    assert client.put("/api/settings/llm", json={"api_key": "x"}).status_code == 409


def _start_llm_game(client, llm_env, monkeypatch, agent):
    client.put("/api/settings/llm", json={"api_key": "sk-test"})
    monkeypatch.setattr(llm_env, "_make_llm_agent", lambda occupant: agent)
    resp = client.post("/api/game",
                       json=_setup(["human", "llm:test", "human", "human"]))
    assert resp.status_code == 201, resp.get_json()


def test_failing_llm_pauses_and_substitute_recovers(client, llm_env, monkeypatch):
    _start_llm_game(client, llm_env, monkeypatch, _FailingLLM())

    paused = _wait(lambda: client.get("/api/game").get_json()["pausedLlm"])
    assert paused["seat"] == 1
    assert paused["error"] == "stub: model call failed"

    # Wrong-seat and bad-action recoveries are refused.
    assert client.post("/api/game/llm-recover",
                       json={"seat": 2, "action": "retry"}).status_code == 422
    assert client.post("/api/game/llm-recover",
                       json={"seat": 1, "action": "nope"}).status_code == 422

    resp = client.post("/api/game/llm-recover", json={"seat": 1, "action": "substitute"})
    assert resp.status_code == 200
    # The solver bot takes over: seat 1's declaration arrives and no pause remains.
    view = _wait(lambda: (v := client.get("/api/game").get_json())
                 and v["declarations"][1] is not None and v)
    assert view["pausedLlm"] is None


def test_failing_llm_retry_path(client, llm_env, monkeypatch):
    agent = _FlakyLLM()
    _start_llm_game(client, llm_env, monkeypatch, agent)

    _wait(lambda: client.get("/api/game").get_json()["pausedLlm"])
    agent.works = True             # the transient failure clears
    resp = client.post("/api/game/llm-recover", json={"seat": 1, "action": "retry"})
    assert resp.status_code == 200
    view = _wait(lambda: (v := client.get("/api/game").get_json())
                 and v["declarations"][1] is not None and v)
    assert view["pausedLlm"] is None
    assert view["declarations"][1] == 1     # the same (recovered) agent decided


def test_resume_llm_save_without_key_routes_to_substitution(client, llm_env, monkeypatch):
    real_factory = llm_env._make_llm_agent
    _start_llm_game(client, llm_env, monkeypatch, _FailingLLM())
    assert client.post("/api/saves", json={"name": "llmgame"}).status_code == 201
    client.delete("/api/game")

    # Restart with no key at all: resume succeeds, then pauses on the LLM seat.
    (llm_env.INSTANCE_DIR / "settings.json").unlink()
    monkeypatch.setattr(llm_env, "_make_llm_agent", real_factory)
    assert client.post("/api/saves/llmgame/resume").status_code == 200

    paused = _wait(lambda: client.get("/api/game").get_json()["pausedLlm"])
    assert paused["seat"] == 1
    assert "key" in paused["error"]

    resp = client.post("/api/game/llm-recover", json={"seat": 1, "action": "substitute"})
    assert resp.status_code == 200
    view = _wait(lambda: (v := client.get("/api/game").get_json())
                 and v["declarations"][1] is not None and v)
    assert view["pausedLlm"] is None
