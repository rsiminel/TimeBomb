"""Play full games and write logs grouped into a per-run directory.

Each invocation writes to ``<out>/<label>/``:
  * ``<label>-0.md`` / ``.jsonl`` -- one transcript + event-log pair per game, named after
    the run so files from different runs stay distinguishable side by side.
  * ``manifest.json`` -- the run-level record: agent config, parameters, and every game's
    seed + outcome. The entry point for later analysis (see sim/analysis/).
  * ``run.log`` -- the run's console output (stdout is teed here).

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
# "" = the repo root itself, needed since the shims import the promoted `tbgame` package.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("", "timebomb", "sim", os.path.join("sim", "agents")):
  sys.path.insert(0, os.path.join(_ROOT, _p))

from engine import Engine
from transcript import write_markdown
from llm import LLMAgent
from programmatic import RandomAgent
from state import SCHEMA_VERSION

AGENTS = {"llm": LLMAgent, "random": RandomAgent}


def play_one(idx, players, agent_name, run_dir, seed, agent_kwargs):
  """Play and log one game. Top-level (picklable) so it can run in a worker process.
  Returns a manifest entry dict."""
  kwargs = agent_kwargs if agent_name == "llm" else {}
  agents = [AGENTS[agent_name](**kwargs) for _ in range(players)]
  outcome, log = Engine(num_players=players).play_game(agents, seed=seed)

  # Both game files share the stem <label>-<game index>, so files from different runs stay
  # distinguishable when several are open at once; seed and outcome live in the manifest.
  stem = "%s-%d" % (os.path.basename(run_dir), idx)
  base = os.path.join(run_dir, stem)
  log.to_jsonl(base + ".jsonl")
  write_markdown(log, base + ".md")
  return {"idx": idx, "seed": seed, "good_guys_won": outcome["good_guys_won"],
          "reason": outcome["reason"], "file": stem, "usage": _sum_usage(agents)}


def _sum_usage(agents):
  """Sum every agent's token tally for one game (0 for non-LLM agents)."""
  return _sum_usage_dicts(getattr(a, "usage", None) for a in agents)


def _sum_usage_dicts(dicts):
  """Sum a sequence of usage dicts (skipping ``None``) into a fresh total."""
  total = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
           "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "cost_usd": 0.0}
  for d in dicts:
    for k, v in (d or {}).items():
      total[k] += v
  return total


class _Tee:
  """Mirror everything printed to ``<run_dir>/run.log`` so a run's console record travels
  with its logs (no shell redirection needed for background runs)."""

  def __init__(self, stream, path):
    self._s, self._f = stream, open(path, "a", buffering=1)   # line-buffered: tail-able live

  def write(self, data):
    self._s.write(data)
    self._f.write(data)

  def flush(self):
    self._s.flush()
    self._f.flush()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--games", type=int, default=1)
  ap.add_argument("--players", type=int, default=6)   # standard Time Bomb config: 6p, 2 bad, 1 bomb
  ap.add_argument("--agent", choices=AGENTS, default="llm")
  ap.add_argument("--notes", default=None, help="free-text note recorded in the run manifest")
  ap.add_argument("--out", default=os.path.join(_ROOT, "sim", "logs"))
  ap.add_argument("--label", default=None, help="run directory name (default: <time>_<agent>)")
  ap.add_argument("--seed", type=int, default=None, help="base seed; game g uses seed+g")
  ap.add_argument("--concurrency", type=int, default=1, help="games to run in parallel")
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

  now = datetime.datetime.now()
  label = args.label or "%s_%s" % (now.strftime("%Y%m%d-%H%M"), args.agent)
  run_dir = os.path.join(args.out, label)
  os.makedirs(run_dir, exist_ok=True)
  sys.stdout = _Tee(sys.stdout, os.path.join(run_dir, "run.log"))

  def seed_for(g):
    return None if args.seed is None else args.seed + g

  jobs = [(g, args.players, args.agent, run_dir, seed_for(g), agent_kwargs)
          for g in range(args.games)]
  if args.concurrency > 1:
    with ProcessPoolExecutor(max_workers=args.concurrency) as ex:
      results = [f.result() for f in as_completed([ex.submit(play_one, *j) for j in jobs])]
  else:
    results = [play_one(*j) for j in jobs]
  results.sort(key=lambda r: r["idx"])

  run_usage = _sum_usage_dicts(r.get("usage") for r in results)
  manifest = {
      "schema_version": SCHEMA_VERSION,
      "label": label,
      "created": now.isoformat(timespec="seconds"),
      "notes": args.notes,
      "params": {"players": args.players, "agent": args.agent,
                 "games": args.games, "base_seed": args.seed},
      "agent_config": AGENTS[args.agent](**(agent_kwargs if args.agent == "llm" else {})).describe(),
      "usage": run_usage,
      "games": results,
  }
  with open(os.path.join(run_dir, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

  wins = sum(r["good_guys_won"] for r in results)
  for r in results:
    u = r.get("usage") or {}
    print("  g%03d s%s: %s (%s)  ·  %d calls, %d in / %d out tok, $%.4f"
          % (r["idx"], r["seed"], "good win" if r["good_guys_won"] else "bad win",
             r["reason"], u.get("calls", 0), u.get("input_tokens", 0),
             u.get("output_tokens", 0), u.get("cost_usd", 0.0)))
  print("run '%s': %d games, good-guy wins %d/%d -> %s"
        % (label, len(results), wins, len(results), run_dir))
  ng = max(len(results), 1)
  print("tokens: %d in + %d out = %d total over %d calls (%d games)  ·  est. cost $%.4f"
        % (run_usage["input_tokens"], run_usage["output_tokens"],
           run_usage["input_tokens"] + run_usage["output_tokens"],
           run_usage["calls"], len(results), run_usage["cost_usd"]))
  print("per-game mean: %.0f calls, %.0f total tokens, $%.4f"
        % (run_usage["calls"] / ng,
           (run_usage["input_tokens"] + run_usage["output_tokens"]) / ng,
           run_usage["cost_usd"] / ng))


if __name__ == "__main__":
  main()
