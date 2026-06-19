"""Play full games and write logs grouped into a per-run directory.

Each invocation writes to ``<out>/<label>/``:
  * ``g000_s0_bad.md`` / ``.jsonl`` -- one transcript + event-log pair per game, the
    filename carrying the game index, seed, and outcome so a run is skimmable at a glance.
  * ``manifest.json`` -- the run-level record: agent config, parameters, and every game's
    seed + outcome. The entry point for later analysis (see sim/analysis/).

  $ python sim/run.py --games 1 --seed 0
  $ python sim/run.py --games 8 --concurrency 4 --label haiku-nothink-baseline
  $ python sim/run.py --games 5 --agent random            # instant smoke run

Speed: within a game the declaration phase runs in parallel (engine.py); the cut phase is
causally sequential. ``--concurrency`` runs whole games in separate *processes* -- required
because the backend deal uses the global ``random`` module.
"""

import argparse
import datetime
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# -- path bootstrap (repo convention; packaging is a separate TODO) -----------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("timebomb", "sim", os.path.join("sim", "agents")):
  sys.path.insert(0, os.path.join(_ROOT, _p))

from engine import Engine
from transcript import write_markdown
from llm import LLMAgent
from programmatic import RandomAgent

AGENTS = {"llm": LLMAgent, "random": RandomAgent}


def play_one(idx, players, agent_name, run_dir, seed):
  """Play and log one game. Top-level (picklable) so it can run in a worker process.
  Returns a manifest entry dict."""
  agents = [AGENTS[agent_name]() for _ in range(players)]
  outcome, log = Engine(num_players=players).play_game(agents, seed=seed)

  verdict = "good" if outcome["good_guys_won"] else "bad"
  stem = "g%03d_s%s_%s" % (idx, "rand" if seed is None else seed, verdict)
  base = os.path.join(run_dir, stem)
  log.to_jsonl(base + ".jsonl")
  write_markdown(log, base + ".md")
  return {"idx": idx, "seed": seed, "good_guys_won": outcome["good_guys_won"],
          "reason": outcome["reason"], "file": stem}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--games", type=int, default=1)
  ap.add_argument("--players", type=int, default=4)
  ap.add_argument("--agent", choices=AGENTS, default="llm")
  ap.add_argument("--out", default=os.path.join(_ROOT, "sim", "logs"))
  ap.add_argument("--label", default=None, help="run directory name (default: <time>_<agent>)")
  ap.add_argument("--seed", type=int, default=None, help="base seed; game g uses seed+g")
  ap.add_argument("--concurrency", type=int, default=1, help="games to run in parallel")
  args = ap.parse_args()

  now = datetime.datetime.now()
  label = args.label or "%s_%s" % (now.strftime("%Y%m%d-%H%M"), args.agent)
  run_dir = os.path.join(args.out, label)
  os.makedirs(run_dir, exist_ok=True)

  def seed_for(g):
    return None if args.seed is None else args.seed + g

  jobs = [(g, args.players, args.agent, run_dir, seed_for(g)) for g in range(args.games)]
  if args.concurrency > 1:
    with ProcessPoolExecutor(max_workers=args.concurrency) as ex:
      results = [f.result() for f in as_completed([ex.submit(play_one, *j) for j in jobs])]
  else:
    results = [play_one(*j) for j in jobs]
  results.sort(key=lambda r: r["idx"])

  manifest = {
      "label": label,
      "created": now.isoformat(timespec="seconds"),
      "params": {"players": args.players, "agent": args.agent,
                 "games": args.games, "base_seed": args.seed},
      "agent_config": AGENTS[args.agent]().describe(),
      "games": results,
  }
  with open(os.path.join(run_dir, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

  wins = sum(r["good_guys_won"] for r in results)
  for r in results:
    print("  g%03d s%s: %s (%s)" % (r["idx"], r["seed"],
                                    "good win" if r["good_guys_won"] else "bad win", r["reason"]))
  print("run '%s': %d games, good-guy wins %d/%d -> %s"
        % (label, len(results), wins, len(results), run_dir))


if __name__ == "__main__":
  main()
