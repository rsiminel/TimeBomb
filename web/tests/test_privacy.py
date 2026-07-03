"""Hotseat privacy (specs/002-host-local-game, SC-003 / FR-008): script an all-human
game through the HTTP API, record every response body, and assert none of them ever
carries hidden information -- roles, hands, bomb possession, declaration truth --
outside the one PrivateView the unlock endpoint returns for the seat that asked.
Also: unlocking seat A and then asking for seat B without re-locking is refused."""

import pytest


# Keys that only ever belong server-side during play (the deal, declaration truth)
# or inside an unlock response (the my* trio). `reveal` is the post-game payload.
HIDDEN_KEYS = {"wires", "bombs", "true_wires", "roles", "myRole", "myWires",
               "iHoldBomb", "my_role", "my_wires", "i_hold_bomb", "reveal"}


def _hidden_keys_in(body, allowed=frozenset()):
    """Recursively collect any HIDDEN_KEYS present in a JSON body."""
    found = set()
    if isinstance(body, dict):
        for key, value in body.items():
            if key in HIDDEN_KEYS and key not in allowed:
                found.add(key)
            found |= _hidden_keys_in(value, allowed)
    elif isinstance(body, list):
        for item in body:
            found |= _hidden_keys_in(item, allowed)
    return found


def _setup(n=4, seed=11):
    return {"seats": [{"name": "P%d" % i, "occupant": "human"} for i in range(n)],
            "seed": seed}


UNLOCK_ALLOWED = frozenset({"myRole", "myWires", "iHoldBomb"})


def test_unlock_is_single_seat_at_a_time(client):
    client.post("/api/game", json=_setup())
    assert client.post("/api/game/unlock", json={"seat": 0, "version": 1}).status_code == 200
    resp = client.post("/api/game/unlock", json={"seat": 1, "version": 1})
    assert resp.status_code == 403          # seat 0 is mid-action; B's view is refused
    client.post("/api/game/lock")
    assert client.post("/api/game/unlock", json={"seat": 1, "version": 1}).status_code == 200


def test_no_response_ever_leaks_hidden_information(client):
    """Play a full 4-human hotseat game; every pre-finish response body must be free
    of hidden fields. The seat's own unlock response may hold only its my* trio."""
    assert client.post("/api/game", json=_setup(seed=11)).status_code == 201

    leaks = []

    def check(what, resp, allowed=frozenset()):
        body = resp.get_json(silent=True)
        bad = _hidden_keys_in(body, allowed)
        if bad:
            leaks.append((what, sorted(bad)))
        return body

    rounds_seen = set()
    for _ in range(200):                     # bound: a game is far shorter than this
        resp = client.get("/api/game")
        view = resp.get_json()
        if view["phase"] == "finished":
            break                            # reveal is public by design now (FR-025)
        check("GET /api/game", resp)
        rounds_seen.add(view["roundIndex"])
        seat = view["pending"]["seats"][0]
        unlock = client.post("/api/game/unlock",
                             json={"seat": seat, "version": view["version"]})
        priv = check("unlock seat %d" % seat, unlock, allowed=UNLOCK_ALLOWED)
        assert unlock.status_code == 200 and priv["myIndex"] == seat

        if view["pending"]["kind"] == "declare":
            intent = {"seat": seat, "kind": "declare", "value": priv["myWires"],
                      "version": view["version"]}
        else:
            intent = {"seat": seat, "kind": "cut", "value": view["legalTargets"][0],
                      "version": view["version"]}
        resp = client.post("/api/game/intent", json=intent)
        assert resp.status_code == 200, resp.get_json()
        # A game-ending cut legitimately returns game_end (roles are public the
        # moment the game is over); everything before that stays hidden.
        events = resp.get_json()["events"]
        if not any(e["type"] == "game_end" for e in events):
            check("intent %s by seat %d" % (view["pending"]["kind"], seat), resp)
    else:
        pytest.fail("game never finished")

    # The leak surface under test includes the round-crossing cut (the next round's
    # deal rides its response unless redacted), so the game must span rounds.
    assert len(rounds_seen) >= 2, "seed must yield a multi-round game to exercise this"
    assert not leaks, "hidden fields leaked: %s" % leaks
