"""M3 analysis: read game event logs and summarise *emergent* play (SPEC.md §9).

Pure log reading -- no model calls, so it is free to run and re-run as new questions come
up (SPEC.md §8). Point it at a run directory or a single ``.jsonl``:

  $ python sim/analysis/analyze.py sim/logs/20260619-1545_llm
  $ python sim/analysis/analyze.py sim/logs --csv sim/analysis/out

It reports three things, each cross-referenced to the hidden ground truth in the log:
  1. Outcomes -- good-guy win rate, overall and by true bad count.
  2. Bluffing -- how far declarations stray from the truth (declared - true), split by the
     hidden role, with good-guys-holding-the-bomb broken out (the deferred strategic-lie
     case of model.md §3.6).
  3. Cut behaviour -- what cuts by each role actually reveal (are bad guys wasting cuts?).

Assistant *calibration* under this emergent play (does P(bad) stay honest when agents lie
off-model?) is the natural next analysis; it needs a General.py replay over each log's
public events and is left as a follow-up.
"""

import argparse
import glob
import json
import os
from collections import Counter, defaultdict

ROLE_GROUPS = ("good_nobomb", "good_bomb", "bad")


def load_games(path):
  if os.path.isdir(path):
    files = sorted(glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True))
  else:
    files = [path]
  for f in files:
    with open(f) as fh:
      events = [json.loads(line) for line in fh if line.strip()]
    if events and events[0]["type"] == "game_start":
      yield f, events


def collect(path):
  games = good = 0
  by_numbad = defaultdict(lambda: [0, 0])          # num_bad -> [games, good_wins]
  decl = {g: [] for g in ROLE_GROUPS}              # group -> list of (declared - true)
  cuts = {0: Counter(), 1: Counter()}              # cutter role -> result counts
  decl_rows, cut_rows = [], []                     # for optional CSV export

  for _f, events in load_games(path):
    gs = events[0]
    roles = gs["roles"]
    end = next((e for e in events if e["type"] == "game_end"), None)
    if end is None:
      continue
    bombs = {e["round"]: e["bombs"] for e in events if e["type"] == "round_start"}
    games += 1
    good += end["good_guys_won"]
    nb = gs.get("num_bad")
    by_numbad[nb][0] += 1
    by_numbad[nb][1] += end["good_guys_won"]

    for e in events:
      if e["type"] == "declaration":
        role = roles[e["player"]]
        held = bombs.get(e["round"], [0] * len(roles))[e["player"]]
        delta = e["declared"] - e["true_wires"]
        group = "bad" if role else ("good_bomb" if held else "good_nobomb")
        decl[group].append(delta)
        decl_rows.append((role, int(bool(held)), e["declared"], e["true_wires"], delta))
      elif e["type"] == "cut":
        cuts[roles[e["cutter"]]][e["result"]] += 1
        cut_rows.append((roles[e["cutter"]], e["result"]))

  return dict(games=games, good=good, by_numbad=by_numbad, decl=decl, cuts=cuts,
              decl_rows=decl_rows, cut_rows=cut_rows)


def _pct(n, d):
  return "%5.1f%%" % (100 * n / d) if d else "  n/a"


def report(s):
  print("=" * 64)
  print("OUTCOMES   (%d games)" % s["games"])
  print("  good-guy win rate: %s" % _pct(s["good"], s["games"]))
  for nb in sorted(k for k in s["by_numbad"] if k is not None):
    g, w = s["by_numbad"][nb]
    print("    true num_bad=%d: %s  (%d games)" % (nb, _pct(w, g), g))

  print("\nBLUFFING   (declaration - true wire count)")
  print("  %-12s %7s %9s %9s   distribution" % ("group", "decls", "lie rate", "mean Δ"))
  for grp in ROLE_GROUPS:
    deltas = s["decl"][grp]
    if not deltas:
      continue
    lie = sum(d != 0 for d in deltas) / len(deltas)
    mean = sum(deltas) / len(deltas)
    hist = " ".join("%+d:%d" % (k, v) for k, v in sorted(Counter(deltas).items()))
    print("  %-12s %7d %8.0f%% %+9.2f   %s" % (grp, len(deltas), 100 * lie, mean, hist))
  print("  (good_bomb = a GOOD guy who happens to hold the bomb -- model.md §3.6)")

  print("\nCUT BEHAVIOUR   (what each role's cuts reveal)")
  print("  %-12s %7s %8s %8s %8s" % ("cutter", "cuts", "wire", "dud", "bomb"))
  for role, label in ((0, "good"), (1, "bad")):
    c = s["cuts"][role]
    tot = sum(c.values())
    if not tot:
      continue
    print("  %-12s %7d %8s %8s %8s" % (
        label, tot, _pct(c["wire"], tot),
        _pct(c["dud"], tot), _pct(c["bomb"], tot)))
  print("=" * 64)


def write_csv(s, out_dir):
  os.makedirs(out_dir, exist_ok=True)
  with open(os.path.join(out_dir, "declarations.csv"), "w") as f:
    f.write("role,held_bomb,declared,true_wires,delta\n")
    for row in s["decl_rows"]:
      f.write(",".join(map(str, row)) + "\n")
  with open(os.path.join(out_dir, "cuts.csv"), "w") as f:
    f.write("cutter_role,result\n")
    for row in s["cut_rows"]:
      f.write("%d,%s\n" % row)
  print("wrote declarations.csv, cuts.csv to %s" % out_dir)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("path", help="a run directory or a single .jsonl game log")
  ap.add_argument("--csv", default=None, help="also dump per-record CSVs to this directory")
  args = ap.parse_args()
  s = collect(args.path)
  if not s["games"]:
    raise SystemExit("no game logs found at %s" % args.path)
  report(s)
  if args.csv:
    write_csv(s, args.csv)


if __name__ == "__main__":
  main()
