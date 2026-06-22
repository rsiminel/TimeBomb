"""Turn an ``EventLog`` into a chronological, easy-to-peruse Markdown transcript: every
player's thoughts and actions in order, round by round, with the hidden ground truth shown
alongside (this is a post-game analysis artifact, not something an agent ever sees -- the
firewall is about in-game views, SPEC.md §3/§8).

The JSONL log is the machine-readable replay; this is the human read. ``run.py`` writes
both per game.
"""

import json


def _role(roles, i):
  return "BAD" if roles[i] else "good"


def _quote(text):
  """Render a reasoning string as a single-paragraph blockquote."""
  if not text:
    return None
  return "  > " + " ".join(str(text).split())


def _fmt_agent(d):
  """One-line summary of an agent descriptor (see ``Agent.describe``)."""
  if "model" in d:
    return "%s · model=%s · thinking_tokens=%s · retries=%s" % (
        d.get("type", "LLMAgent"), d["model"], d.get("thinking_tokens"), d.get("retries"))
  return d.get("type", d.get("name", "?"))


def _render_agents(out, names, agents):
  """Group players by identical agent config and list them, so a transcript records
  exactly what produced it (model, instructions, ...)."""
  if not agents:
    return
  groups = {}   # json-key -> (descriptor, [player indices])
  for i, d in enumerate(agents):
    key = json.dumps(d, sort_keys=True)
    groups.setdefault(key, (d, []))[1].append(i)
  out.append("**Agents:**")
  for d, idxs in groups.values():
    who = ", ".join("`%d` %s" % (i, names[i]) for i in idxs)
    out.append("- %s — %s" % (who, _fmt_agent(d)))
    if d.get("system"):
      out.append("  - system: _%s_" % d["system"])
    if d.get("declare_instruction"):
      out.append("  - declare/cut instructions and full config are in the .jsonl `game_start` event")


def render_markdown(log):
  events = log.events
  meta = next(e for e in events if e["type"] == "game_start")
  names, roles = meta["player_names"], meta["roles"]
  end = next((e for e in events if e["type"] == "game_end"), None)

  out = []
  out.append("# Time Bomb — Game transcript")
  out.append("")
  out.append("**Players:** " + ", ".join("`%d` %s" % (i, n) for i, n in enumerate(names)))
  out.append("**Roles (revealed):** "
             + ", ".join("%s = %s" % (names[i], _role(roles, i)) for i in range(len(names)))
             + "  ·  num_bad = %d" % meta["num_bad"])
  if end:
    who = "🟢 Good guys WIN" if end["good_guys_won"] else "🔴 Bad guys win"
    out.append("**Result:** %s — %s" % (who, end["reason"]))
  out.append("")
  _render_agents(out, names, meta.get("agents"))
  out.append("")
  out.append("---")

  cut_n = 0
  cuts_header_done = True
  for e in events:
    t = e["type"]
    if t == "round_start":
      bomb_holder = next((i for i, b in enumerate(e["bombs"]) if b), None)
      out.append("")
      out.append("## Round %d  ·  hand size %d  ·  %d active wires"
                 % (e["round"] + 1, e["hand_size"], e["active_wires"]))
      out.append("")
      out.append("_Hidden deal — wires per hand %s · bomb held by %s_"
                 % (e["wires"], names[bomb_holder] if bomb_holder is not None else "nobody"))
      out.append("")
      out.append("### Declarations")
      cut_n = 0
      cuts_header_done = False
    elif t == "declaration":
      i = e["player"]
      out.append("- **%s** (%s) declares **%s**  _(truly holds %d)_"
                 % (names[i], _role(roles, i), e["declared"], e["true_wires"]))
      q = _quote(e.get("reasoning"))
      if q:
        out.append(q)
    elif t in ("cut", "cut_skipped"):
      if not cuts_header_done:
        out.append("")
        out.append("### Cuts")
        cuts_header_done = True
      if t == "cut_skipped":
        out.append("- _(cut skipped — %s had no legal target)_" % names[e["cutter"]])
        continue
      cut_n += 1
      out.append("%d. **%s** (%s) cuts **%s** → **%s**"
                 % (cut_n, names[e["cutter"]], _role(roles, e["cutter"]),
                    names[e["target"]], e["result"]))
      if e.get("message"):                       # public table talk (vs the private reasoning below)
        out.append('   💬 _%s:_ "%s"' % (names[e["cutter"]], e["message"]))
      q = _quote(e.get("reasoning"))
      if q:
        out.append(q)

  out.append("")
  return "\n".join(out)


def write_markdown(log, path):
  with open(path, "w") as f:
    f.write(render_markdown(log))
