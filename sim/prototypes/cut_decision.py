"""M1 vertical slice: ONE cut decision, by ONE LLM agent, from ONE hand-built game state.

The smallest thing that exercises the only two hard parts of an LLM agent (SPEC.md §6):
  1. state -> text   (the renderer in state.py)
  2. text -> validated action   (JSON output + validation + fallback)

No game loop, no multi-agent, no logging yet. Run it, read the reasoning, iterate on the
prompt until the agent cuts sensibly. Starting defaults: stateless (the full observable
history is in the prompt), structured (JSON) output, retry-once-then-fallback.

The model runs via **headless Claude Code** (`claude -p`), which authenticates with your
existing Claude Code login -- no ANTHROPIC_API_KEY. Each call is a fresh, isolated
subprocess, which *is* our stateless-agent design (one process per decision = naturally
isolated per-agent context). We replace the default system prompt with a short persona
(cheaper, faster, and keeps Claude Code's coding identity out of the player) and run from
a neutral cwd so the repo's CLAUDE.md is not auto-injected.

  $ python sim/prototypes/cut_decision.py          # live, if the claude CLI is present
  $ python sim/prototypes/cut_decision.py --dry     # just print the prompt that would be sent
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

# -- path bootstrap (repo convention; packaging is a separate TODO) -----------
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in ("timebomb", "sim", os.path.join("sim", "agents")):
  sys.path.insert(0, os.path.join(_ROOT, _p))

from state import (GroundTruth, PublicState, PrivateView, AgentView,
                   legal_targets, render_agent)

MODEL = "claude-haiku-4-5"   # fast model for play; reserve opus for deep-dives

SYSTEM = ("You are an expert, strategic Time Bomb player. Play to win for your secret "
          "team, reading the public declarations and cut history for tells. You think "
          "privately, then commit to one move.")

OUTPUT_INSTRUCTION = (
    "Respond with ONLY a JSON object and nothing else:\n"
    '{"reasoning": "<your private thinking, not shown to anyone>", '
    '"target": <the player index you cut>}')

# A neutral working dir so the repo's CLAUDE.md is not auto-discovered into the agent.
_NEUTRAL_CWD = tempfile.mkdtemp(prefix="tb_agent_")


def example_view():
  """A hand-built mid-round snapshot: round 2, 4 players, it is the BAD guy's turn to cut.
  He holds no bomb and doesn't know the bomb sits in Player 3's hand -- a clean spot to see
  whether the model plays its role (stall / steer toward an unknown hand vs. find wires)."""
  N = 4
  roles = [0, 0, 1, 0]            # Player 2 (Clara) is the lone bad guy
  wires = [1, 0, 1, 1]            # truth this round (sums to round_start_active = 3)
  bombs = [0, 0, 0, 1]            # Player 3 (Darryl) holds the bomb this round
  gt = GroundTruth(roles=roles, wires=wires, bombs=bombs, num_bad=1, seed=None)

  pub = PublicState(
      num_players=N, num_bad_prior={1: 2 / 5, 2: 3 / 5}, num_bom=1,
      player_names=["Alice", "Bob", "Clara", "Darryl"],
      round_index=1, hand_size=4, round_start_active=3, active_wires=2,
      declarations=[2, 0, 1, 1],          # Player 0 over-declared; Player 2 (you) said 1
      revealed=[0, 1, 1, 0], found=[0, 0, 1, 0],
      cut_log=[
          {"round": 1, "cutter": 0, "target": 1, "result": "dud"},
          {"round": 1, "cutter": 1, "target": 2, "result": "wire"},
      ],
      declaration_history=[[1, 1, 1, 0]],  # round 1's declarations
      current_cutter=2)

  me = 2
  private = PrivateView(my_index=me, my_role=roles[me],
                        my_wires=wires[me], i_hold_bomb=bool(bombs[me]))
  return AgentView(public=pub.snapshot(), private=private)


def decide_cut(view, dry=False):
  """state -> text -> validated action. Returns (target, reasoning, source)."""
  prompt = render_agent(view, "cut") + "\n\n" + OUTPUT_INSTRUCTION
  legal = legal_targets(view.public, view.private.my_index)

  if dry or shutil.which("claude") is None:
    print("=== SYSTEM (replaces Claude Code default) ===\n" + SYSTEM)
    print("\n=== USER (rendered state) ===\n" + prompt)
    return random.choice(legal), "(dry run: no model called)", "fallback"

  target, reasoning = _call_model(prompt)            # one attempt
  if target not in legal:                            # retry once, then fall back
    target, reasoning = _call_model(
        prompt + "\n\nYour previous choice was illegal. Legal targets are %s." % legal)
  if target not in legal:
    return random.choice(legal), reasoning, "fallback"
  return target, reasoning, "model"


def _call_model(prompt):
  """One headless `claude -p` call. Returns (target, reasoning); target is -1 on failure
  so the caller can retry / fall back."""
  try:
    proc = subprocess.run(
        ["claude", "-p", prompt,
         "--output-format", "json",
         "--system-prompt", SYSTEM,
         "--model", MODEL],
        cwd=_NEUTRAL_CWD, capture_output=True, text=True, timeout=180)
  except (subprocess.TimeoutExpired, OSError) as e:
    return -1, "(call failed: %s)" % e
  if proc.returncode != 0:
    return -1, "(claude exited %d: %s)" % (proc.returncode, proc.stderr[:200])
  try:
    result_text = json.loads(proc.stdout)["result"]   # unwrap the CLI envelope
  except (ValueError, KeyError):
    return -1, "(could not parse CLI envelope)"
  return _extract_action(result_text)


def _extract_action(text):
  """Pull {"target", "reasoning"} out of the model's reply, tolerating code fences and
  surrounding prose."""
  try:
    s, e = text.find("{"), text.rfind("}")
    obj = json.loads(text[s:e + 1])
    return int(obj["target"]), obj.get("reasoning", "")
  except (ValueError, KeyError):
    return -1, "(could not parse action JSON from: %r)" % text[:200]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--dry", action="store_true", help="print the prompt, do not call the model")
  args = ap.parse_args()

  view = example_view()
  target, reasoning, source = decide_cut(view, dry=args.dry)
  names = view.public.player_names
  print("\n--- DECISION (%s) ---" % source)
  print("Reasoning: %s" % reasoning)
  print("Cut: Player %d (%s)" % (target, names[target]))


if __name__ == "__main__":
  main()
