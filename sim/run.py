"""Play ONE full game and write its logs into a per-run directory.

One invocation = one game (running several means several invocations). This keeps every
run self-contained: there is no run-level index to maintain, so the game's own files are
the whole record.

Each invocation writes to ``<out>/<label>/``:
  * ``<label>.jsonl`` -- the game event log (the canonical machine record; see sim/LOGS.md),
    flushed incrementally as the game plays -- a killed process never loses progress.
  * ``<label>.sessions.json`` -- per-seat agent state (LLM session id, narration cursor,
    ...), rewritten atomically after every event, so a resumed LLM seat continues its
    real `claude -p` session instead of restarting it.
  * ``<label>.md`` -- the game transcript (the human read), with a ``**Run:**`` header line
    carrying token/cost usage and any ``--notes`` (the durable home for what the retired
    manifest used to hold). Only written once the game actually finishes.
  * ``<label>.<Name>.md`` -- one per LLM player: that seat's full raw `claude -p` session
    (persona + every turn's prompt and reply, verbatim), the game from its perspective.

  $ python sim/run.py --seed 0
  $ python sim/run.py --players 6 --model claude-fable-5 --label fable-probe
  $ python sim/run.py --agent random            # instant smoke run

By default the engine halts the instant one LLM decision exhausts its retries (instead
of silently falling back to a legal move), so there's an exact point to resume from:

  $ python sim/run.py --resume sim/logs/fable-probe

Resume reconstructs the roster and the Engine's settings from the log itself -- no need
to repeat ``--players``/``--model``/etc. ``--halt-on-failure`` off restores the old
behavior (a flaky call never stops the game, and there is nothing to resume).

Speed: within the game the declaration phase runs in parallel (engine.py); the cut phase is
causally sequential.
"""

import argparse
import datetime
import json
import os
import re
import sys

# -- path bootstrap (repo convention; packaging is a separate TODO) -----------
# "" = the repo root itself, needed since the shims import the promoted `tbgame` package.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("", "timebomb", "sim", os.path.join("sim", "agents")):
  sys.path.insert(0, os.path.join(_ROOT, _p))

from engine import Engine, EngineHalted
from state import EventLog
from assistant import PanelAssistant
from transcript import write_markdown, write_agent_session
from llm import LLMAgent
from programmatic import RandomAgent

AGENTS = {"llm": LLMAgent, "random": RandomAgent}
AGENTS_BY_TYPE = {cls.__name__: cls for cls in AGENTS.values()}


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


# ---------------------------------------------------------------------------
# Incremental persistence -- so a killed/interrupted run is always resumable
# ---------------------------------------------------------------------------

def _write_sidecar(path, agents):
  """Atomic rewrite (tmp file + rename) so a crash mid-write can't corrupt it -- this is
  exactly the file a resume depends on being intact."""
  tmp = path + ".tmp"
  with open(tmp, "w") as f:
    json.dump({"agents": [a.to_state() for a in agents]}, f)
  os.replace(tmp, path)


def _make_sink(base, agents, truncate):
  """One callback, wired into the game's ``EventLog``, that fires after every committed
  event: append the line to the on-disk jsonl (flushed) and rewrite the sessions sidecar
  from the agents' current state. Used for every run, not just resumed ones -- so every
  run is resumable, not an opt-in mode. ``truncate`` starts the jsonl clean (a fresh run,
  even one reusing a stale ``--label``); a resume must append after the loaded events
  instead, never truncate."""
  jf = open(base + ".jsonl", "w" if truncate else "a")

  def sink(event):
    jf.write(json.dumps(event, default=str) + "\n")
    jf.flush()
    _write_sidecar(base + ".sessions.json", agents)
  return sink


def _load_run(run_dir):
  """Load a prior (possibly partial) run's event log + agent sidecar for ``--resume``."""
  label = os.path.basename(os.path.normpath(run_dir))
  base = os.path.join(run_dir, label)
  with open(base + ".jsonl") as f:
    events = [json.loads(line) for line in f if line.strip()]
  sidecar_path = base + ".sessions.json"
  states = []
  if os.path.exists(sidecar_path):
    with open(sidecar_path) as f:
      states = json.load(f)["agents"]
  return base, events, states


def _agent_cwd(run_dir, i):
  """A per-seat directory, stable for the life of a run (fresh or resumed), that an LLM
  seat's `claude -p` calls run from. Claude Code looks up a resumed session (--resume
  <id>) relative to the cwd it was opened from -- a directory that's re-randomized every
  process (the module's own default) can never find that session again later, which
  defeats resume for exactly the thing it's supposed to continue. Never touched besides
  this -- no CLAUDE.md ever lands here, same guarantee the old random tempdir gave."""
  return os.path.join(run_dir, ".sessions", "seat%d" % i)


def _rebuild_agents(run_dir, events, states, timeout=None):
  """Re-instantiate each seat's agent from its logged ``describe()`` descriptor (model,
  thinking budget, retries, ...) and restore its saved state (LLM session id, narration
  cursor, ...) so it continues its real session rather than starting over."""
  agents = []
  for i, d in enumerate(events[0]["agents"]):
    cls = AGENTS_BY_TYPE[d["type"]]
    kwargs = {}
    if cls is LLMAgent:
      kwargs = dict(model=d["model"], thinking_tokens=d["thinking_tokens"],
                    timeout=(timeout or d["timeout"]), retries=d["retries"],
                    cwd=_agent_cwd(run_dir, i))
    agent = cls(**kwargs)
    if i < len(states):
      agent.load_state(states[i])
    agents.append(agent)
  return agents


def _rebuild_engine(events, halt_on_failure):
  start = events[0]
  assistant = PanelAssistant() if start["panel_for"] else None
  return Engine(num_players=start["num_players"], player_names=start["player_names"],
               panel_for=start["panel_for"], assistant=assistant,
               talk_between_cuts=start["talk_between_cuts"], talk_top_k=start["talk_top_k"],
               halt_on_failure=halt_on_failure)


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
  ap.add_argument("--talk-between-cuts", action=argparse.BooleanOptionalAction, default=True,
                  help="a full discussion pass before every cut, ending with the cutter "
                       "(--no-talk-between-cuts: one pass per round, after declarations)")
  ap.add_argument("--talk-top-k", type=int, default=2,
                  help="discuss calls per pass: only the k highest urgency bidders speak; "
                       "a skipped player costs no call (negative: everyone is called)")
  ap.add_argument("--assistant", action=argparse.BooleanOptionalAction, default=None,
                  help="show every player the public-info stats readout "
                       "(default: on for llm agents, off for programmatic ones)")
  ap.add_argument("--halt-on-failure", action=argparse.BooleanOptionalAction, default=True,
                  help="stop the game the instant one decision exhausts its retries, so "
                       "--resume has an exact point to continue from (--no-halt-on-failure: "
                       "the old behavior -- a flaky call silently falls back and play continues)")
  ap.add_argument("--resume", default=None, metavar="RUN_DIR",
                  help="continue a prior (halted or interrupted) run from its saved log + "
                       "sessions sidecar; roster and Engine settings come from the log, not "
                       "from --players/--agent/--model/etc. (--timeout may still override)")
  args = ap.parse_args()

  if args.resume:
    run_dir = args.resume
    base, events, states = _load_run(run_dir)
    label = os.path.basename(base)
    agents = _rebuild_agents(run_dir, events, states, timeout=args.timeout)
    eng = _rebuild_engine(events, args.halt_on_failure)
    log = EventLog(events=events, sink=_make_sink(base, agents, truncate=False))
    seed = events[0]["seed"]
  else:
    now = datetime.datetime.now()
    label = args.label or "%s_%s" % (now.strftime("%Y%m%d-%H%M"), args.agent)
    run_dir = os.path.join(args.out, label)
    os.makedirs(run_dir, exist_ok=True)
    base = os.path.join(run_dir, label)

    agent_kwargs = {"thinking_tokens": args.thinking_tokens}
    if args.model:
      agent_kwargs["model"] = args.model
    if args.timeout:
      agent_kwargs["timeout"] = args.timeout
    if args.agent == "llm":
      agents = [AGENTS[args.agent](cwd=_agent_cwd(run_dir, i), **agent_kwargs)
               for i in range(args.players)]
    else:
      agents = [AGENTS[args.agent]() for _ in range(args.players)]
    use_panel = args.assistant if args.assistant is not None else args.agent == "llm"
    eng_kwargs = dict(num_players=args.players, talk_between_cuts=args.talk_between_cuts,
                      talk_top_k=args.talk_top_k if args.talk_top_k >= 0 else None,
                      halt_on_failure=args.halt_on_failure)
    if use_panel:
      eng_kwargs.update(assistant=PanelAssistant(), panel_for=range(args.players))
    eng = Engine(**eng_kwargs)
    log = EventLog(sink=_make_sink(base, agents, truncate=True))
    seed = args.seed

  try:
    outcome, log = eng.play_game(agents, seed=seed, log=log)
  except EngineHalted as exc:
    print("  halted: %s" % exc)
    print("  progress saved -- resume with: python sim/run.py --resume %s" % run_dir)
    sys.exit(1)

  usage = _sum_usage(agents)
  write_markdown(log, base + ".md", run_meta={"usage": usage, "notes": args.notes})
  _write_agent_sessions(agents, log, base)

  print("  %s (%s)  ·  %d calls, %d in / %d out tok, $%.4f"
        % ("good win" if outcome["good_guys_won"] else "bad win", outcome["reason"],
           usage["calls"], usage["input_tokens"], usage["output_tokens"], usage["cost_usd"]))
  print("run '%s' (seed %s) -> %s" % (label, seed, run_dir))


if __name__ == "__main__":
  main()
