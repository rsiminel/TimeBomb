# Contract: web HTTP API (002)

Transport-only layer over `tbgame` (contracts/engine.md) and `web/replay.py`. All
bodies JSON. Errors: `{error: str}` with 4xx status; engine rejection reasons are
forwarded verbatim. The v1 endpoint `POST /api/panel` is unchanged.

## Pages

| Route | Serves |
|---|---|
| `GET /` | home page (two doors) |
| `GET /assistant` | v1 assistant page, moved verbatim (its own JS/CSS under `/assistant/…`) |
| `GET /play` | game page |

## Game lifecycle

| Endpoint | Body → Response | Notes |
|---|---|---|
| `POST /api/game` | GameSetup → `201 {version}` | `409` if a game is active (body says so; client offers abandon). `422` on invalid setup. |
| `DELETE /api/game` | — → `204` | abandon the active game (client confirms first). |
| `GET /api/game` | — → TableView | the shared surface; includes `version`, `phase`, `pending`, `thinking`, `panel_allowed`, `unlocked_seat`, and (when finished or exhibition) the reveal payload. `404` if no active game. |

Polling contract: the client polls `GET /api/game` (~1 s) whenever `thinking` is
non-empty or after submitting; all state changes are observable through `version`.

## Hotseat privacy (FR-008/009)

| Endpoint | Body → Response | Notes |
|---|---|---|
| `POST /api/game/unlock` | `{seat, version}` → `200 PrivateView` | pass-the-device ack: unlocks exactly that human seat's private view. `409` on version mismatch; `403` if the seat is not human or another seat is mid-action. |
| `POST /api/game/lock` | `{}` → `204` | re-locks (also implicit on any accepted intent and on page load). |

`GET /api/game` never contains private fields. `PrivateView` is returned only by
`unlock`, only for one seat at a time. In human games no response ever carries
another seat's role/hand/bomb (SC-003 is tested against recorded traffic).

## Intents

| Endpoint | Body → Response |
|---|---|
| `POST /api/game/intent` | `{seat, kind: "declare"\|"cut", value, claim?, version}` → `200 {version, events}` |

`422` with the engine's reason on `IllegalIntent`; `409` on version mismatch (stale
tab). Claims ride the intent (clarification Q3). Accepted intents re-lock privacy.

## Panel (FR-019/020/021)

| Endpoint | Response |
|---|---|
| `GET /api/game/panel` | `200` — `replay.py` panel output, verbatim, for the game's public record. `403` if the game did not opt in. |

## Saves (FR-023)

| Endpoint | Body → Response |
|---|---|
| `GET /api/saves` | `200 [{name, created, seats, phase}]` |
| `POST /api/saves` | `{name}` → `201` (`409` duplicate name, `404` no active game) |
| `POST /api/saves/<name>/resume` | → `200 {version}` (`409` if a game is active; `422 {error}` on schema mismatch) |
| `DELETE /api/saves/<name>` | → `204` (client confirms first) |

## LLM configuration (FR-017)

| Endpoint | Body → Response |
|---|---|
| `GET /api/settings/llm` | `200 {configured: bool, source: "env"\|"file"\|null}` — never the key itself |
| `PUT /api/settings/llm` | `{api_key}` → `204`; stored server-side; `409` if env var is set (env wins) |
| `POST /api/game/llm-recover` | `{seat, action: "retry"\|"substitute"}` → `200 {version}` — resolves a paused LLM seat |

## Replay (FR-025)

Replay is client-side over data already in the finished game's reveal payload
(truth-annotated event list from `game.reveal()`); no additional endpoint. Resumed
*finished* saves open directly in reveal/replay.
