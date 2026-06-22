"""Engine correctness for the sim/ arena -- the M0 exit criterion (SPEC.md §9).

This suite is about *game rules and the information firewall*, not probability math (that
is covered against the math.comb oracle elsewhere). It runs entirely on deterministic stub
agents -- no LLM, no network -- so it is fast and reproducible. Paths are set by the
repo conftest.py.
"""

import random

import pytest

from engine import (Engine, WIRE, BLANK, BOMB,
                    _validate_declaration, _validate_target)
from state import (GroundTruth, PublicState, PrivateView, AgentView, legal_targets,
                   render_agent)
from base import Agent
from llm import LLMAgent


# ---------------------------------------------------------------------------
# Deterministic test agents (no RNG, no model) -> seeded games are reproducible
# ---------------------------------------------------------------------------

class StubAgent(Agent):
  """Declares a fixed count; always cuts the lowest-index legal target."""

  def __init__(self, decl=1):
    self.decl = decl
    self.last_reasoning = "stub"

  def declare(self, view):
    return min(self.decl, view.public.hand_size)

  def choose_cut(self, view):
    return min(legal_targets(view.public, view.private.my_index))


class MalAgent(Agent):
  """Returns garbage to exercise the engine's validation/fallback."""

  def declare(self, view):
    return None

  def choose_cut(self, view):
    return 999


def _stub_pub(**over):
  base = dict(num_players=4, num_bad_prior={1: 1.0}, num_bom=1,
              player_names=["A", "B", "C", "D"], round_index=0, hand_size=5,
              round_start_active=5, active_wires=5,
              declarations=[None] * 4, revealed=[0] * 4, found=[0] * 4)
  base.update(over)
  return PublicState(**base)


def _events(log, kind):
  return [e for e in log.events if e["type"] == kind]


# ---------------------------------------------------------------------------
# Information firewall
# ---------------------------------------------------------------------------

def test_agent_view_carries_no_ground_truth():
  eng = Engine(num_players=4)
  gt = GroundTruth(roles=[0, 1, 0, 0], wires=[1, 0, 2, 1], bombs=[0, 0, 0, 1], num_bad=1)
  view = eng._build_view(gt, _stub_pub(), 0)

  # The view is exactly {public, private, assistant_panel} -- nothing else.
  assert set(vars(view)) == {"public", "private", "assistant_panel"}
  leaky = {"roles", "wires", "bombs"}
  assert not (set(vars(view.public)) & leaky)
  assert not (set(vars(view.private)) & leaky)
  # Player 0 knows only its own role, and learns nothing about others' roles/hands.
  assert view.private.my_index == 0
  assert view.private.my_role == gt.roles[0]
  assert view.private.my_wires == gt.wires[0]


def test_private_view_is_per_player():
  eng = Engine(num_players=4)
  gt = GroundTruth(roles=[0, 1, 0, 0], wires=[1, 0, 2, 1], bombs=[0, 0, 0, 1], num_bad=1)
  for i in range(4):
    v = eng._build_view(gt, _stub_pub(), i)
    assert v.private.my_role == gt.roles[i]
    assert v.private.i_hold_bomb == bool(gt.bombs[i])


# ---------------------------------------------------------------------------
# Legal targets + validation/fallback
# ---------------------------------------------------------------------------

def test_legal_targets_excludes_self_and_fully_revealed():
  pub = _stub_pub(revealed=[0, 5, 1, 0])   # player 1 fully revealed (hand_size 5)
  assert legal_targets(pub, 0) == [2, 3]   # not self (0), not the exhausted hand (1)
  assert legal_targets(pub, 2) == [0, 3]


def test_validate_declaration_clamps_to_fallback():
  assert _validate_declaration(3, 5, fallback=0) == 3
  assert _validate_declaration(9, 5, fallback=2) == 2     # out of range -> fallback
  assert _validate_declaration(-1, 5, fallback=2) == 2
  assert _validate_declaration(None, 5, fallback=2) == 2  # non-int -> fallback


def test_validate_target_falls_back_to_legal():
  random.seed(0)
  assert _validate_target(2, [1, 2, 3]) == 2
  assert _validate_target(7, [1, 2, 3]) in (1, 2, 3)      # illegal -> some legal target
  assert _validate_target(None, [1, 2, 3]) in (1, 2, 3)


# ---------------------------------------------------------------------------
# Cut resolution against the hidden deal
# ---------------------------------------------------------------------------

def test_resolve_cut_outcomes():
  eng = Engine(num_players=4)
  pub = _stub_pub()
  all_wires = GroundTruth(roles=[0] * 4, wires=[5, 0, 0, 0], bombs=[0] * 4, num_bad=0)
  no_wires = GroundTruth(roles=[0] * 4, wires=[0, 0, 0, 0], bombs=[0] * 4, num_bad=0)
  for s in range(50):
    random.seed(s)
    assert eng._resolve_cut(all_wires, pub, 0) == WIRE   # every card is a wire
    assert eng._resolve_cut(no_wires, pub, 1) == BLANK   # no wire, no bomb


def test_resolve_cut_bomb_is_the_last_facedown_slot():
  eng = Engine(num_players=4)
  bomb_gt = GroundTruth(roles=[0] * 4, wires=[0, 0, 0, 0], bombs=[0, 0, 1, 0], num_bad=0)
  pub = _stub_pub(revealed=[0, 0, 4, 0])   # player 2 has one face-down card left
  for s in range(20):
    random.seed(s)
    assert eng._resolve_cut(bomb_gt, pub, 2) == BOMB     # only card left is the bomb


# ---------------------------------------------------------------------------
# Full-game flow, outcomes, and invariants
# ---------------------------------------------------------------------------

def _play(seed, agent=StubAgent):
  return Engine(num_players=4).play_game([agent() for _ in range(4)], seed=seed)


def test_game_terminates_with_consistent_outcome():
  for seed in range(40):
    outcome, log = _play(seed)
    assert isinstance(outcome["good_guys_won"], bool)
    assert outcome["reason"] in ("all wires cut", "bomb detonated", "out of time")
    cuts = _events(log, "cut")
    wires_cut = sum(c["result"] == WIRE for c in cuts)
    bombs_cut = [c for c in cuts if c["result"] == BOMB]

    if outcome["reason"] == "all wires cut":
      assert outcome["good_guys_won"]
      assert wires_cut == 4                     # all N active wires found
      assert log.events[-2]["type"] == "cut"    # ended on the wire-finding cut
    elif outcome["reason"] == "bomb detonated":
      assert not outcome["good_guys_won"]
      assert len(bombs_cut) == 1 and cuts[-1]["result"] == BOMB
    else:  # out of time
      assert not outcome["good_guys_won"]
      assert not bombs_cut and wires_cut < 4
      assert len(_events(log, "round_start")) == 4   # played all four rounds (H 5->2)


def test_seed_is_reproducible():
  out_a, log_a = _play(123)
  out_b, log_b = _play(123)
  assert out_a == out_b
  results_a = [(e["cutter"], e["target"], e["result"]) for e in _events(log_a, "cut")]
  results_b = [(e["cutter"], e["target"], e["result"]) for e in _events(log_b, "cut")]
  assert results_a == results_b


def test_every_logged_cut_was_legal():
  # Replay the public revealed-counts and confirm each cut targeted a legal player.
  for seed in range(40):
    _, log = _play(seed)
    revealed = [0, 0, 0, 0]
    cur_round = None
    for e in log.events:
      if e["type"] == "round_start":
        revealed = [0, 0, 0, 0]
        cur_round = e["round"]
      elif e["type"] == "cut":
        c, t = e["cutter"], e["target"]
        assert t != c                                  # never cut yourself
        assert revealed[t] < 5 - (cur_round or 0)      # had a face-down card
        revealed[t] += 1


def test_declarations_are_in_range():
  for seed in range(20):
    _, log = _play(seed)
    for e in _events(log, "declaration"):
      hand_size = 5 - e["round"]
      assert 0 <= e["declared"] <= hand_size


def test_malformed_agent_never_crashes():
  # Mixed roster of garbage-returning agents: the game must still complete legally.
  agents = [MalAgent(), StubAgent(), MalAgent(), StubAgent()]
  outcome, log = Engine(num_players=4).play_game(agents, seed=7)
  assert outcome["reason"] in ("all wires cut", "bomb detonated", "out of time")
  for e in _events(log, "cut"):
    assert e["target"] != e["cutter"]
  for e in _events(log, "declaration"):
    assert 0 <= e["declared"] <= 5 - e["round"]


def test_log_starts_and_ends_well_formed():
  _, log = _play(1)
  assert log.events[0]["type"] == "game_start"
  assert log.events[0]["player_names"] == ["Alice", "Bob", "Clara", "Darryl"]
  assert log.events[-1]["type"] == "game_end"


# ---------------------------------------------------------------------------
# Coherent agent context (regression for the stale-cut_log bug)
# ---------------------------------------------------------------------------

class CoherenceAgent(StubAgent):
  """Asserts, every time it is asked to cut, that the public cut log it is shown agrees
  with the face-down counts: the number of cuts recorded this round must equal the number
  of cards revealed this round. The old bug refreshed cut_log only at round end, so a
  mid-round view showed changed face-down counts beside an empty 'cuts this round'."""

  def choose_cut(self, view):
    pub = view.public
    this_round = [c for c in pub.cut_log if c["round"] == pub.round_index]
    assert len(this_round) == sum(pub.revealed), (
        "cut_log shows %d cuts this round but %d cards are revealed"
        % (len(this_round), sum(pub.revealed)))
    return super().choose_cut(view)


def test_view_cut_log_stays_coherent_with_revealed():
  for seed in range(30):
    Engine(num_players=4).play_game([CoherenceAgent() for _ in range(4)], seed=seed)


def test_render_has_rules_and_full_history():
  # A hand-built mid-round-2 cut view: one past round + one cut already made this round.
  pub = PublicState(
      num_players=4, num_bad_prior={1: 1.0}, num_bom=1, player_names=["A", "B", "C", "D"],
      round_index=1, hand_size=4, round_start_active=3, active_wires=3,
      declarations=[1, 1, 0, 2], revealed=[0, 0, 1, 0], found=[0, 0, 0, 0],
      declaration_history=[[2, 0, 1, 1]],
      cut_log=[{"round": 0, "cutter": 0, "target": 1, "result": BLANK},
               {"round": 1, "cutter": 0, "target": 2, "result": BLANK}])
  view = AgentView(public=pub, private=PrivateView(1, 0, 1, False))
  text = render_agent(view, "cut")

  assert "TIME BOMB" in text and "Bad guys do NOT 'protect' the bomb" in text  # rules primer
  assert "Round 1 (hand size 5):" in text          # the past round, with its hand size
  assert "THIS ROUND — Round 2 (hand size 4):" in text
  assert "Declared wire counts — A=2" in text       # past-round declarations are shown
  assert "1. A cut C → a dud" in text               # this round's in-progress cut is shown
  assert "none yet" not in text                     # ... so it is NOT called empty (the bug)


def test_llm_agent_accumulates_private_memory():
  a = LLMAgent()
  assert a._memory_block() == ""                    # empty before any decision
  a.last_reasoning = "blend in as a normal player"
  a._remember(0, "declared 1")
  a.last_reasoning = "cut the loud over-declarer"
  a._remember(0, "cut C (Player 2)")
  block = a._memory_block()
  assert "Round 1" in block
  assert "declared 1" in block and "cut C (Player 2)" in block
  assert "blend in as a normal player" in block     # past reasoning is carried forward


if __name__ == "__main__":
  raise SystemExit(pytest.main([__file__, "-q"]))
