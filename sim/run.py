"""Play full games and write one log pair per game: a machine-readable ``.jsonl`` event
log (the replay / analysis source, SPEC.md §8) and a human-readable ``.md`` transcript.

  $ python sim/run.py --games 1 --players 4 --out sim/logs            # all-LLM game
  $ python sim/run.py --games 8 --concurrency 4                       # 8 games, 4 at once
  $ python sim/run.py --games 5 --agent random --seed 0               # instant smoke run

Speed: within a game the declaration phase runs in parallel (see ``engine.py``); the cut
phase is causally sequential and cannot be. ``--concurrency`` runs whole games in separate
*processes* -- required because the backend deal uses the global ``random`` module, so
games must not share an interpreter's RNG state.
"""

import argparse
import datetime
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


def play_one(idx, players, agent_name, out_dir, seed):
  """Play and log one game. Top-level (picklable) so it can run in a worker process.
  Returns a one-line summary string."""
  agents = [AGENTS[agent_name]() for _ in range(players)]
  outcome, log = Engine(num_players=players).play_game(agents, seed=seed)

  stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  base = os.path.join(out_dir, "game_%s_%s_s%s_g%d" % (stamp, agent_name, seed, idx))
  log.to_jsonl(base + ".jsonl")
  write_markdown(log, base + ".md")
  verdict = "good win" if outcome["good_guys_won"] else "bad win"
  return "game %d: %s (%s) -> %s.md" % (idx, verdict, outcome["reason"], base)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--games", type=int, default=1)
  ap.add_argument("--players", type=int, default=4)
  ap.add_argument("--agent", choices=AGENTS, default="llm")
  ap.add_argument("--out", default=os.path.join(_ROOT, "sim", "logs"))
  ap.add_argument("--seed", type=int, default=None, help="base seed; game g uses seed+g")
  ap.add_argument("--concurrency", type=int, default=1, help="games to run in parallel")
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)

  def seed_for(g):
    return None if args.seed is None else args.seed + g

  jobs = [(g, args.players, args.agent, args.out, seed_for(g)) for g in range(args.games)]
  if args.concurrency > 1:
    with ProcessPoolExecutor(max_workers=args.concurrency) as ex:
      futs = [ex.submit(play_one, *job) for job in jobs]
      for f in as_completed(futs):
        print(f.result())
  else:
    for job in jobs:
      print(play_one(*job))


if __name__ == "__main__":
  main()
