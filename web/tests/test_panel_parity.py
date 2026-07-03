"""Panel parity (specs/002-host-local-game, SC-008 / FR-019..021): the hosted game's
opt-in panel must be numerically identical to what the v1 assistant would show for
the same public events. The test scripts a hotseat game, mirrors every public move
into a hand-built v1 GameRecord, and compares `GET /api/game/panel` against
`POST /api/panel` after each accepted intent. Also: 403 when the game did not opt in."""


def _setup(n=4, seed=23, panel_allowed=True):
    return {"seats": [{"name": "P%d" % i, "occupant": "human"} for i in range(n)],
            "seed": seed, "panel_allowed": panel_allowed}


def test_panel_403_when_not_opted_in(client):
    client.post("/api/game", json=_setup(panel_allowed=False))
    resp = client.get("/api/game/panel")
    assert resp.status_code == 403
    assert "error" in resp.get_json()


def test_panel_matches_v1_at_every_step(client):
    n = 4
    assert client.post("/api/game", json=_setup(n=n)).status_code == 201

    # The v1 record, grown move-by-move alongside the hosted game. Declarations
    # enter it only once the round's full set is in (matching the public timeline).
    record = {"setup": {"players": ["P%d" % i for i in range(n)], "bomb": True,
                        "numBadOverride": None},
              "events": []}
    round_decls = {}

    def compare(step):
        hosted = client.get("/api/game/panel")
        assert hosted.status_code == 200, hosted.get_json()
        v1 = client.post("/api/panel", json=record)
        assert v1.status_code == 200, v1.get_json()
        assert hosted.get_json()["belief"] == v1.get_json()["belief"], step
        assert hosted.get_json()["state"] == v1.get_json()["state"], step

    compare("before any move")
    steps = 0
    for _ in range(200):
        view = client.get("/api/game").get_json()
        if view["phase"] == "finished":
            break
        seat = view["pending"]["seats"][0]
        priv = client.post("/api/game/unlock",
                           json={"seat": seat, "version": view["version"]}).get_json()
        if view["pending"]["kind"] == "declare":
            value = min(priv["myWires"] + (seat % 2), view["handSize"])  # some lies
            intent = {"seat": seat, "kind": "declare", "value": value,
                      "version": view["version"]}
            round_decls[seat] = value
        else:
            target = view["legalTargets"][0]
            intent = {"seat": seat, "kind": "cut", "value": target,
                      "version": view["version"]}
        resp = client.post("/api/game/intent", json=intent)
        assert resp.status_code == 200, resp.get_json()

        # Mirror the accepted move into the v1 record.
        if view["pending"]["kind"] == "declare":
            if len(round_decls) == n:
                record["events"].append(
                    {"type": "declarations",
                     "values": [round_decls[i] for i in range(n)]})
                round_decls = {}
        else:
            result = [e for e in resp.get_json()["events"] if e["type"] == "cut"][0]["result"]
            record["events"].append(
                {"type": "cut", "player": target,
                 "result": {"wire": "safe", "dud": "nothing", "bomb": "bomb"}[result]})

        compare("after step %d (%s by seat %d)" % (steps, view["pending"]["kind"], seat))
        steps += 1
    else:
        assert False, "game never finished"
    assert steps > n, "game too short to exercise parity"
