"""Play ONE full game and write its logs into a per-run directory.

One invocation = one game (running several means several invocations). This keeps every
run self-contained: there is no run-level index to maintain, so the game's own files are
the whole record.

Each invocation writes to ``<out>/<label>/``:
  * ``<label>.jsonl`` -- the game event log (the canonical machine record; see sim/LOGS.md).
  * ``<label>.md`` -- the game transcript (the human read), with a ``**Run:**`` header line
    carrying token/cost usage and any ``--notes`` (the durable home for what the retired
    manifest used to hold).
  * ``<label>.<Name>.md`` -- one per LLM player: that seat's full raw `claude -p` session
    (persona + every turn's prompt and reply, verbatim), the game from its perspective.

  $ python sim/run.py --seed 0
  $ python sim/run.py --players 6 --model claude-fable-5 --label fable-probe
  $ python sim/run.py --agent random            # instant smoke run

Speed: within the game the declaration phase runs in parallel (engine.py); the cut phase is
causally sequential.
"""

import argparse
import datetime
import os
import re
import sys

# -- path bootstrap (repo convention; packaging is a separate TODO) -----------
# "" = the repo root itself, needed since the shims import the promoted `tbgame` package.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("", "timebomb", "sim", os.path.join("sim", "agents")):
  sys.path.insert(0, os.path.join(_ROOT, _p))

from engine import Engine
from transcript import write_markdown, write_agent_session
from llm import LLMAgent
from programmatic import RandomAgent

AGENTS = {"llm": LLMAgent, "random": RandomAgent}


def _sum_usage(agents):
  """Sum every agent's token tally for the game (0 for non-LLM agents)."""
  total = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
           "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "cost_usd": 0.0}
  for a in agents:
    for k, v in (getattr(a, "usage", None) or {}).items():
      total[k] += v
  return total


def _slug(name):
  """Filesystem-safe fragment of a player name for the per-agent transcript filename."""
  return re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("_") or "player"


def _write_agent_sessions(agents, log, base):
  """Write each LLM player's raw session transcript to ``<base>.<Name>.md``."""
  start = log.events[0]
  names, roles = start["player_names"], start["roles"]
  for i, agent in enumerate(agents):
    turns = getattr(agent, "transcript", None)
    if not turns:                                # non-LLM agents keep no session
      continue
    role = "BAD" if roles[i] else "good"
    path = "%s.%s.md" % (base, _slug(names[i]))
    write_agent_session(names[i], role, agent.describe(), turns, path)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--players", type=int, default=6)   # standard Time Bomb config: 6p, 2 bad, 1 bomb
  ap.add_argument("--agent", choices=AGENTS, default="llm")
  ap.add_argument("--notes", default=None, help="free-text note recorded in the transcript header")
  ap.add_argument("--out", default=os.path.join(_ROOT, "sim", "logs"))
  ap.add_argument("--label", default=None, help="run directory name (default: <time>_<agent>)")
  ap.add_argument("--seed", type=int, default=None, help="RNG seed (deal + cut resolution)")
  ap.add_argument("--model", default=None, help="LLM model id (default: agent's own default)")
  ap.add_argument("--thinking-tokens", type=int, default=0,
                  help="extended-thinking budget: 0 off (fast), >0 cap, <0 Claude Code default (on)")
  ap.add_argument("--timeout", type=int, default=None, help="per-call timeout seconds")
  args = ap.parse_args()

  agent_kwargs = {"thinking_tokens": args.thinking_tokens}
  if args.model:
    agent_kwargs["model"] = args.model
  if args.timeout:
    agent_kwargs["timeout"] = args.timeout
  kwargs = agent_kwargs if args.agent == "llm" else {}

  now = datetime.datetime.now()
  label = args.label or "%s_%s" % (now.strftime("%Y%m%d-%H%M"), args.agent)
  run_dir = os.path.join(args.out, label)
  os.makedirs(run_dir, exist_ok=True)

  agents = [AGENTS[args.agent](**kwargs) for _ in range(args.players)]
  outcome, log = Engine(num_players=args.players).play_game(agents, seed=args.seed)

  usage = _sum_usage(agents)
  base = os.path.join(run_dir, label)
  log.to_jsonl(base + ".jsonl")
  write_markdown(log, base + ".md", run_meta={"usage": usage, "notes": args.notes})
  _write_agent_sessions(agents, log, base)

  print("  %s (%s)  ·  %d calls, %d in / %d out tok, $%.4f"
        % ("good win" if outcome["good_guys_won"] else "bad win", outcome["reason"],
           usage["calls"], usage["input_tokens"], usage["output_tokens"], usage["cost_usd"]))
  print("run '%s' (seed %s) -> %s" % (label, args.seed, run_dir))


if __name__ == "__main__":
  main()
