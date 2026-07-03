"""Contract tests for tbgame.engine.TableGame (specs/002-host-local-game).

Covers the state transitions of data-model.md: IllegalIntent on every bad intent
shape (wrong seat/kind/range/claim), deal determinism under a fixed seed, from_events
round-trip equality, and reveal only when finished. Independent of TableGame's
internals (T005) -- exercises only the public contract of contracts/engine.md.
"""

import pytest

from tbgame.engine import TableGame, SetupError, IllegalIntent, SchemaError
from tbgame.state import SCHEMA_VERSION


def _setup(n=4, seed=1, role_deal="official", panel_allowed=False, occupant="solver_bot"):
  return {
      "seats": [{"name": "P%d" % i, "occupant": occupant} for i in range(n)],
      "role_deal": role_deal,
      "seed": seed,
      "panel_allowed": panel_allowed,
  }


def _declare_all(game, value=1):
  events = []
  for seat in list(game.pending.seats):
    events += game.submit({"seat": seat, "kind": "declare", "value": value})
  return events


def _play_to_finish(game, declare_value=1):
  """Drive a game to completion with a fixed declaration and always-lowest-legal cuts."""
  while not game.finished:
    pending = game.pending
    if pending.kind == "declare":
      _declare_all(game, declare_value)
    else:
      seat = pending.seats[0]
      legal = _legal(game, seat)
      game.submit({"seat": seat, "kind": "cut", "value": min(legal)})
  return game


def _legal(game, seat):
  pub = game.public_state()
  return [j for j in range(pub.num_players) if j != seat and pub.revealed[j] < pub.hand_size]


# ---------------------------------------------------------------------------
# Setup validation
# ---------------------------------------------------------------------------

def test_setup_rejects_bad_seat_count():
  with pytest.raises(SetupError):
    TableGame(_setup(n=3))
  with pytest.raises(SetupError):
    TableGame(_setup(n=9))


def test_setup_rejects_duplicate_names():
  setup = _setup(n=4)
  setup["seats"][1]["name"] = setup["seats"][0]["name"]
  with pytest.raises(SetupError):
    TableGame(setup)


def test_setup_rejects_bad_occupant():
  setup = _setup(n=4)
  setup["seats"][0]["occupant"] = "ghost"
  with pytest.raises(SetupError):
    TableGame(setup)


def test_setup_accepts_llm_occupant_tag():
  setup = _setup(n=4)
  setup["seats"][0]["occupant"] = "llm:claude-sonnet-5"
  game = TableGame(setup)
  assert game.occupants[0] == "llm:claude-sonnet-5"


def test_setup_rejects_infeasible_role_deal_override():
  with pytest.raises(SetupError):
    TableGame(_setup(n=4, role_deal=0))
  with pytest.raises(SetupError):
    TableGame(_setup(n=4, role_deal=4))


def test_setup_accepts_feasible_role_deal_override():
  game = TableGame(_setup(n=4, role_deal=2, occupant="solver_bot"))
  assert sum(game.reveal()["roles"]) == 2


def test_exhibition_mode_when_all_bots():
  game = TableGame(_setup(n=4, occupant="solver_bot"))
  # No human seats -> reveal() works immediately, before any play.
  payload = game.reveal()
  assert "roles" in payload and len(payload["roles"]) == 4


# ---------------------------------------------------------------------------
# Deal determinism (same seed -> same deal)
# ---------------------------------------------------------------------------

def test_same_seed_same_deal():
  a = TableGame(_setup(seed=42))
  b = TableGame(_setup(seed=42))
  assert a.reveal()["roles"] == b.reveal()["roles"]
  round_start_a = [e for e in a.events if e["type"] == "round_start"][0]
  round_start_b = [e for e in b.events if e["type"] == "round_start"][0]
  assert round_start_a["wires"] == round_start_b["wires"]
  assert round_start_a["bombs"] == round_start_b["bombs"]


def test_different_seed_can_differ():
  # Not a hard guarantee for any single pair, but across many seeds at least one deal
  # must differ, else the seed isn't doing anything.
  deals = set()
  for seed in range(20):
    g = TableGame(_setup(seed=seed))
    rs = [e for e in g.events if e["type"] == "round_start"][0]
    deals.add((tuple(rs["wires"]), tuple(rs["bombs"]), tuple(g.reveal()["roles"])))
  assert len(deals) > 1


# ---------------------------------------------------------------------------
# Pending / IllegalIntent
# ---------------------------------------------------------------------------

def test_pending_starts_as_declare_for_all_seats():
  game = TableGame(_setup())
  assert game.pending.kind == "declare"
  assert sorted(game.pending.seats) == [0, 1, 2, 3]


def test_illegal_intent_wrong_seat():
  game = TableGame(_setup())
  legal_seat = game.pending.seats[0]
  # No seat index equal to num_players is ever pending; use one that hasn't gone yet
  # but craft a bad intent using a seat NOT in pending after declaring it once.
  game.submit({"seat": legal_seat, "kind": "declare", "value": 1})
  with pytest.raises(IllegalIntent):
    game.submit({"seat": legal_seat, "kind": "declare", "value": 1})  # already declared


def test_illegal_intent_wrong_kind():
  game = TableGame(_setup())
  seat = game.pending.seats[0]
  with pytest.raises(IllegalIntent):
    game.submit({"seat": seat, "kind": "cut", "value": 1})  # declare phase, not cut


def test_illegal_intent_declare_out_of_range():
  game = TableGame(_setup())
  seat = game.pending.seats[0]
  with pytest.raises(IllegalIntent):
    game.submit({"seat": seat, "kind": "declare", "value": 99})
  with pytest.raises(IllegalIntent):
    game.submit({"seat": seat, "kind": "declare", "value": -1})


def test_illegal_intent_cut_illegal_target():
  game = TableGame(_setup())
  _declare_all(game)
  cutter = game.pending.seats[0]
  with pytest.raises(IllegalIntent):
    game.submit({"seat": cutter, "kind": "cut", "value": cutter})  # can't cut self
  with pytest.raises(IllegalIntent):
    game.submit({"seat": cutter, "kind": "cut", "value": 99})


def test_illegal_intent_bad_claim_shapes():
  game = TableGame(_setup())
  seat = game.pending.seats[0]
  with pytest.raises(IllegalIntent):
    game.submit({"seat": seat, "kind": "declare", "value": 1,
                "claim": {"kind": "distrust", "target": seat}})  # target == speaker
  with pytest.raises(IllegalIntent):
    game.submit({"seat": seat, "kind": "declare", "value": 1,
                "claim": {"kind": "self_honest", "target": 1}})  # self_honest needs no target
  with pytest.raises(IllegalIntent):
    game.submit({"seat": seat, "kind": "declare", "value": 1,
                "claim": {"kind": "nonsense", "target": None}})


def test_valid_claim_rides_declare_and_appears_in_claim_log():
  game = TableGame(_setup())
  seat = game.pending.seats[0]
  other = 1 if seat != 1 else 2
  events = game.submit({"seat": seat, "kind": "declare", "value": 1,
                        "claim": {"kind": "distrust", "target": other}})
  kinds = [e["type"] for e in events]
  assert "declaration" in kinds and "claim" in kinds
  pub = game.public_state()
  assert pub.claim_log and pub.claim_log[-1] == {"round": 0, "speaker": seat,
                                                  "kind": "distrust", "target": other}


def test_illegal_intent_never_applies_partially():
  game = TableGame(_setup())
  seat = game.pending.seats[0]
  before = len(game.events)
  with pytest.raises(IllegalIntent):
    game.submit({"seat": seat, "kind": "declare", "value": 1,
                "claim": {"kind": "distrust", "target": seat}})
  assert len(game.events) == before
  assert seat in game.pending.seats   # declaration was not applied either


# ---------------------------------------------------------------------------
# Round / phase transitions
# ---------------------------------------------------------------------------

def test_phase_advances_declare_to_cut_to_next_round():
  game = TableGame(_setup())
  assert game.public_state().phase == "awaiting_declarations"
  _declare_all(game)
  assert game.public_state().phase == "awaiting_cut"
  assert game.pending.kind == "cut" and len(game.pending.seats) == 1


def test_game_terminates_and_outcome_is_consistent():
  for seed in range(15):
    game = _play_to_finish(TableGame(_setup(seed=seed)))
    assert game.finished
    assert isinstance(game.outcome["good_guys_won"], bool)
    assert game.outcome["reason"] in ("all wires cut", "bomb detonated", "out of time")


# ---------------------------------------------------------------------------
# from_events round-trip
# ---------------------------------------------------------------------------

def test_from_events_round_trip_mid_game():
  game = TableGame(_setup(seed=7))
  _declare_all(game, value=1)
  seat = game.pending.seats[0]
  game.submit({"seat": seat, "kind": "cut", "value": min(_legal(game, seat))})

  rebuilt = TableGame.from_events(game.events)
  assert rebuilt.events == game.events
  assert rebuilt.pending.kind == game.pending.kind
  assert rebuilt.pending.seats == game.pending.seats
  assert rebuilt.public_state().active_wires == game.public_state().active_wires
  assert rebuilt.public_state().revealed == game.public_state().revealed
  for seat in range(4):
    assert rebuilt.private_view(seat) == game.private_view(seat)


def test_from_events_round_trip_finished_game():
  game = _play_to_finish(TableGame(_setup(seed=3)))
  rebuilt = TableGame.from_events(game.events)
  assert rebuilt.finished
  assert rebuilt.outcome == game.outcome
  assert rebuilt.reveal() == game.reveal()


def test_from_events_rejects_schema_mismatch():
  game = TableGame(_setup())
  bad_events = [dict(game.events[0], schema_version=SCHEMA_VERSION + 1)] + game.events[1:]
  with pytest.raises(SchemaError):
    TableGame.from_events(bad_events)


def test_from_events_rejects_missing_game_start():
  with pytest.raises(SchemaError):
    TableGame.from_events([])
  with pytest.raises(SchemaError):
    TableGame.from_events([{"type": "round_start"}])


# ---------------------------------------------------------------------------
# reveal()
# ---------------------------------------------------------------------------

def test_reveal_raises_before_finish_for_human_games():
  game = TableGame(_setup(occupant="human"))
  with pytest.raises(RuntimeError):
    game.reveal()


def test_reveal_after_finish_annotates_lies():
  game = TableGame(_setup(occupant="human", seed=9))
  # Declare a value guaranteed to sometimes mismatch true_wires for at least one seat.
  while not game.finished:
    pending = game.pending
    if pending.kind == "declare":
      for seat in list(pending.seats):
        game.submit({"seat": seat, "kind": "declare", "value": 0})
    else:
      seat = pending.seats[0]
      game.submit({"seat": seat, "kind": "cut", "value": min(_legal(game, seat))})
  payload = game.reveal()
  decls = [e for e in payload["events"] if e["type"] == "declaration"]
  assert decls and all("lie" in e for e in decls)
  assert any(e["lie"] for e in decls) or all(e["declared"] == e["true_wires"] for e in decls)


if __name__ == "__main__":
  raise SystemExit(pytest.main([__file__, "-q"]))
