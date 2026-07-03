"""Self-play gate for SolverBot (specs/002-host-local-game SC-002/SC-005).

100 seeded all-bot games must run to completion with zero illegal intents (any
IllegalIntent raised by ``game.submit`` fails the test), a winner consistent with the
engine's own judgment, an identical dealt hand under a repeated seed (bot PLAY may
still differ -- bots keep their own unseeded RNG, research R2), and every solver-bot
decision within the 2 s latency budget (SC-002).
"""
import time

import pytest

from tbgame.engine import TableGame
from tbgame.state import AgentView
from tbgame.agents.solver import SolverBot

N_SEATS = 5
N_GAMES = 100
DECISION_BUDGET_S = 2.0


def _setup(seed, n=N_SEATS):
  return {"seats": [{"name": "P%d" % i, "occupant": "solver_bot"} for i in range(n)],
          "seed": seed}


def _agent_view(game, seat):
  return AgentView(public=game.public_state(), private=game.private_view(seat))


def _play(game, agents, timings):
  while not game.finished:
    pending = game.pending
    if pending.kind == "declare":
      for seat in list(pending.seats):
        view = _agent_view(game, seat)
        t0 = time.monotonic()
        value = agents[seat].declare(view)
        claim = agents[seat].claim(view)
        timings.append(time.monotonic() - t0)
        game.submit({"seat": seat, "kind": "declare", "value": value, "claim": claim})
    else:
      seat = pending.seats[0]
      view = _agent_view(game, seat)
      t0 = time.monotonic()
      target = agents[seat].choose_cut(view)
      claim = agents[seat].claim(view)
      timings.append(time.monotonic() - t0)
      game.submit({"seat": seat, "kind": "cut", "value": target, "claim": claim})
  return game


@pytest.mark.slow
def test_self_play_beats_the_clock_and_never_breaks_rules():
  timings = []
  for seed in range(N_GAMES):
    game = TableGame(_setup(seed))
    agents = [SolverBot() for _ in range(N_SEATS)]
    _play(game, agents, timings)   # an IllegalIntent here fails the test
    assert game.finished
    assert isinstance(game.outcome["good_guys_won"], bool)
    assert game.outcome["reason"] in ("all wires cut", "bomb detonated", "out of time")
    if game.outcome["reason"] == "bomb detonated":
      assert not game.outcome["good_guys_won"]
    elif game.outcome["reason"] == "all wires cut":
      assert game.outcome["good_guys_won"]
  assert timings and max(timings) < DECISION_BUDGET_S


def test_same_seed_same_deal_regardless_of_bot_play():
  def _deal(seed):
    game = TableGame(_setup(seed))
    rs = next(e for e in game.events if e["type"] == "round_start")
    return game.reveal()["roles"], rs["wires"], rs["bombs"]
  assert _deal(777) == _deal(777)


if __name__ == "__main__":
  raise SystemExit(pytest.main([__file__, "-q", "-m", "slow"]))
