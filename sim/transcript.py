"""Turn an ``EventLog`` into a chronological, easy-to-peruse Markdown transcript: every
player's thoughts and actions in order, round by round, with the hidden ground truth shown
alongside (this is a post-game analysis artifact, not something an agent ever sees -- the
firewall is about in-game views, SPEC.md §3/§8).

The JSONL log is the machine-readable replay; this is the human read. ``run.py`` writes
both, plus one per-agent *session* transcript (``render_agent_session``) -- the raw
`claude -p` conversation each player saw, prompts and replies verbatim.
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
    return "%s · model=%s · thinking_tokens=%s · retries=%s%s" % (
        d.get("type", "LLMAgent"), d["model"], d.get("thinking_tokens"), d.get("retries"),
        " · session" if d.get("session_mode") else "")
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


def _run_line(run_meta):
  """The optional run-level header line: token/cost usage and any free-text note. This is
  the durable home for what the retired manifest.json used to hold (one game per run now,
  so run-level and game-level metadata are the same thing)."""
  bits = []
  u = (run_meta or {}).get("usage")
  if u:
    bits.append("%d calls · %d in / %d out tok · $%.4f"
                % (u.get("calls", 0), u.get("input_tokens", 0),
                   u.get("output_tokens", 0), u.get("cost_usd", 0.0)))
  if (run_meta or {}).get("notes"):
    bits.append(run_meta["notes"])
  return "**Run:** " + "  ·  ".join(bits) if bits else None


def render_markdown(log, run_meta=None):
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
  run_line = _run_line(run_meta)
  if run_line:
    out.append(run_line)
  out.append("")
  _render_agents(out, names, meta.get("agents"))
  out.append("")
  out.append("---")

  cut_n = 0
  cuts_header_done = True
  disc_header_done = True
  for e in events:
    t = e["type"]
    if t == "round_start":
      bomb_holder = next((i for i, b in enumerate(e["bombs"]) if b), None)
      out.append("")
      out.append("## Round %d  ·  hand size %d  ·  %d wires"
                 % (e["round"] + 1, e["hand_size"], e["active_wires"]))
      out.append("")
      out.append("_Hidden deal — wires per hand %s · bomb held by %s_"
                 % (e["wires"], names[bomb_holder] if bomb_holder is not None else "nobody"))
      out.append("")
      out.append("### Declarations")
      cut_n = 0
      cuts_header_done = False
      disc_header_done = False
    elif t == "declaration":
      i = e["player"]
      out.append("- **%s** (%s) declares **%s**  _(truly holds %d)_"
                 % (names[i], _role(roles, i), e["declared"], e["true_wires"]))
      q = _quote(e.get("reasoning"))
      if q:
        out.append(q)
    elif t == "statement":
      if not disc_header_done:
        out.append("")
        out.append("### Discussion")
        disc_header_done = True
      i = e["player"]
      out.append('- **%s** (%s): "%s"' % (names[i], _role(roles, i), e["message"]))
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


def write_markdown(log, path, run_meta=None):
  with open(path, "w") as f:
    f.write(render_markdown(log, run_meta))


def render_agent_session(name, role, descriptor, turns):
  """Render one player's raw `claude -p` session as Markdown: the persona system prompt,
  then every committed turn's exact prompt and the reply the model returned, verbatim. This
  is the game from that one seat's perspective -- what it was told and what it answered -- so
  a move can be traced to the precise context behind it. Post-game artifact; an agent never
  sees another's session (nor its own rendered like this).

  ``descriptor`` is the agent's ``describe()`` dict (for the model + persona); ``turns`` is
  the agent's ``transcript`` list of ``{decision, prompt, reply}`` (committed turns only)."""
  model = descriptor.get("model", "?")
  out = ["# Time Bomb — %s's session  ·  %s  ·  model=%s" % (name, role, model), ""]
  out.append("_The exact conversation %s's `claude -p` session saw: the persona, then each "
             "turn's prompt and the reply it returned (verbatim; committed turns only). "
             "Post-game artifact — no agent ever sees this._" % name)
  if descriptor.get("system"):
    out += ["", "## Persona (system prompt)", "", "```", descriptor["system"], "```"]
  for n, turn in enumerate(turns, 1):
    out += ["", "## Turn %d — %s" % (n, turn["decision"]), "",
            "**Prompt sent:**", "", "```", turn["prompt"], "```", "",
            "**Reply:**", "", "```json", turn["reply"], "```"]
  out.append("")
  return "\n".join(out)


def write_agent_session(name, role, descriptor, turns, path):
  with open(path, "w") as f:
    f.write(render_agent_session(name, role, descriptor, turns))
