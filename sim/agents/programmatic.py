"""Sandbox driver agents -- NOT statistical controls (model-faithful programmatic play
already lives in ``General.PlayAuto``). Their job is to exercise and verify the engine
before any LLM is attached (SPEC.md Principle 3): ``RandomAgent`` for batch smoke-testing,
``HumanAgent`` so you can play a full game by hand through the exact same interface and
renderer the LLM will use.
"""

import random

from base import Agent
from state import legal_targets, render_agent


class RandomAgent(Agent):
  name = "random"

  def declare(self, view):
    return random.randint(0, view.public.hand_size)

  def choose_cut(self, view):
    return random.choice(legal_targets(view.public, view.private.my_index))


class HumanAgent(Agent):
  name = "human"

  def declare(self, view):
    print("\n" + render_agent(view, "declare"))
    return _prompt_int("Your declaration: ", 0, view.public.hand_size)

  def discuss(self, view):
    print("\n" + render_agent(view, "discuss"))
    return input("Say to the table (blank = stay silent): ").strip() or None

  def choose_cut(self, view):
    print("\n" + render_agent(view, "cut"))
    legal = legal_targets(view.public, view.private.my_index)
    while True:
      t = _prompt_int("Player to cut: ", 0, view.public.num_players - 1)
      if t in legal:
        return t
      print("  Not a legal target. Choose from %s." % legal)


def _prompt_int(msg, lo, hi):
  while True:
    try:
      v = int(input(msg))
    except ValueError:
      print("  Enter a whole number.")
      continue
    if lo <= v <= hi:
      return v
    print("  Enter a number between %d and %d." % (lo, hi))
