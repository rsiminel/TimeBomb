"""The LLM agent -- the point of the project. Each decision is one headless `claude -p`
call that authenticates with the existing Claude Code login (no ANTHROPIC_API_KEY), with
the default system prompt replaced by a short player persona and run from a neutral cwd so
the repo's CLAUDE.md is not auto-injected (see sim/prototypes/cut_decision.py for the
proof-of-concept and SPEC.md §6 for the design defaults).

Session mode: each player keeps one persistent `claude -p` session and only sends what is
new each turn (``--resume <session_id>``). The first call seeds full context via
``render_agent``; later calls send a small delta (``render_session_delta``) and the
unchanged prefix is served from the prompt cache, so the growing history is paid for once,
not re-sent every turn. The firewall is unchanged -- a session only ever receives its own
player's legal views. The model returns a JSON object; we validate and the engine falls
back on malformed output.

Speed/robustness tuning (see the timing measurements in the git history):
  * ``MAX_THINKING_TOKENS=0`` is the decisive one. Claude Code runs the model with
    extended thinking by default, which emitted 1.6k-7.8k *thinking* tokens per decision
    (20-70s/call) regardless of the model, ``--effort``, or any "be terse" instruction.
    Disabling it drops a call to ~110 output tokens and ~4s while keeping a coherent,
    strategic ``reasoning`` field. Configurable via ``thinking_tokens`` if you want
    deeper (slower) play.
  * ``--strict-mcp-config`` with no MCP config skips MCP server startup (~1.5s/call).
  * ``stdin=DEVNULL`` so a call can never block waiting on stdin.
  * retry-with-backoff over transient failures (timeout / non-zero exit / API ``is_error``),
    with the last error kept on ``self.last_error`` for diagnosis.
The agent stashes its reasoning on ``self.last_reasoning`` so the engine can log it.
"""

import json
import os
import subprocess
import tempfile
import time

from base import Agent
import state as st
from state import render_agent
# All agent-facing copy lives in prompts.py; re-exported here so the ``system=SYSTEM``
# default, ``describe()``, and any caller using ``llm.SYSTEM`` keep working.
from prompts import SYSTEM, DECLARE_INSTRUCTION, DISCUSS_INSTRUCTION, CUT_INSTRUCTION

MODEL = "claude-haiku-4-5"   # fast model for play; reserve opus for deep-dives

# Neutral cwd shared by all agents, so no CLAUDE.md is auto-discovered into a player.
_NEUTRAL_CWD = tempfile.mkdtemp(prefix="tb_agent_")


def _clamp_urgency(u):
  """The reply's piggybacked speak-bid, as a clamped int in [0, 9]; None if absent/junk."""
  try:
    return max(0, min(9, int(u)))
  except (TypeError, ValueError):
    return None


def _name_to_index(view):
  """Cast for a cut target: the prompt names players (CUT_INSTRUCTION asks for a name), so
  resolve a returned name back to its seat index -- case-insensitively, tolerating a bare
  index too. Raises ``ValueError`` on anything else, so ``_decide`` retries."""
  names = [n.lower() for n in view.public.player_names]

  def cast(v):
    s = str(v).strip()
    if s.lstrip("-").isdigit():
      return int(s)
    if s.lower() in names:
      return names.index(s.lower())
    raise ValueError("unknown player name %r" % (v,))
  return cast


class LLMAgent(Agent):
  name = "llm"

  def __init__(self, model=MODEL, system=SYSTEM, timeout=60, retries=3, thinking_tokens=0):
    self.model = model
    self.system = system
    self.timeout = timeout
    self.retries = retries
    self.thinking_tokens = thinking_tokens   # 0 disables extended thinking (the big speedup)
    self.last_reasoning = None
    self.last_message = None                  # the cutter's public table-talk for its last cut
    self.last_statement = None                # this agent's last discussion-phase statement
    self.last_urgency = None                  # the last reply's speak-bid (engine rations turns)
    self.last_error = None
    self.memory = []                         # this agent's own past decisions + reasoning
    # Full raw session log for the post-game per-agent transcript (sim/transcript.py):
    # the persona (``self.system``) plus, per committed turn, the exact prompt sent and the
    # reply the model returned. Only committed turns are recorded -- a failed-parse retry is
    # re-narrated by the next turn and is not part of the session's logical history.
    self.transcript = []
    # One persistent `claude -p` session per agent (session mode). ``session_id`` is None
    # until the first successful call seeds it; ``cursor`` tracks how far the session has
    # been narrated so each later turn sends only the delta.
    self.session_id = None
    self.cursor = st.new_session_cursor()
    # Running token/cost tally, summed from the `--output-format json` envelope's `usage`
    # block over every subprocess call (retries included -- a failed call still costs). In
    # session mode the replayed prefix shows up as `cache_read_input_tokens`.
    self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
                  "cost_usd": 0.0}

  def describe(self):
    d = super().describe()
    d.update(model=self.model, thinking_tokens=self.thinking_tokens, session_mode=True,
             timeout=self.timeout, retries=self.retries, system=self.system,
             declare_instruction=DECLARE_INSTRUCTION, discuss_instruction=DISCUSS_INSTRUCTION,
             cut_instruction=CUT_INSTRUCTION)
    return d

  def declare(self, view):
    value, self.last_reasoning, _ = self._decide(view, "declare", "declaration",
                                                 DECLARE_INSTRUCTION)
    if value is not None:
      self._remember(view.public.round_index, "declared %d" % value)
    return value

  def choose_cut(self, view):
    value, self.last_reasoning, obj = self._decide(view, "cut", "target", CUT_INSTRUCTION,
                                                   cast=_name_to_index(view))
    self.last_message = obj.get("message") if value is not None else None
    if value is not None:
      who = view.public.player_names[value] if 0 <= value < view.public.num_players else value
      said = (' I told the table: "%s"' % self.last_message) if self.last_message else ""
      self._remember(view.public.round_index, "cut %s (Player %s).%s" % (who, value, said))
    return value

  def discuss(self, view):
    # Discuss replies carry no "reasoning" field (public words only), so the reasoning
    # slot holds "" on success and the failure diagnostic otherwise — keep only the latter.
    msg, reasoning, _ = self._decide(view, "discuss", "message",
                                     DISCUSS_INSTRUCTION, cast=str)
    self.last_reasoning = reasoning or None
    self.last_statement = msg or None
    if self.last_statement:
      self._remember(view.public.round_index, 'said to the table: "%s"' % self.last_statement)
    return self.last_statement

  # -- a private running record of this agent's own moves (for inspection; the live session
  #    is the model's real memory, so this is no longer fed back into the prompt) ----------

  def _memory_block(self):
    if not self.memory:
      return ""
    return ("\n\nYOUR OWN PRIVATE NOTES from earlier this game (your past moves and the "
            "reasoning behind them):\n"
            + "\n".join(self.memory))

  def _remember(self, round_index, action):
    self.memory.append("- [Round %d] You %s. Your reasoning then: %s"
                       % (round_index + 1, action, self.last_reasoning))

  # -- one decision: view -> text -> validated action, over a resumed session, with retries -

  def _decide(self, view, decision, key, instruction, cast=int):
    """Return ``(value, reasoning, obj)``. Sends a full opener on the first turn (seeding the
    session) and a small delta on every later turn; the session id and narration cursor are
    advanced only on success, so a failed turn's events are re-narrated by the next one.
    ``cast`` coerces the action field (``int`` for a declaration/target, ``str`` for a
    table-talk message). ``value`` is ``None`` after all retries fail, so the engine falls
    back to a legal move (or, for a statement, to silence)."""
    if self.session_id is None:
      body, pending = render_agent(view, decision), st.cursor_after_opener(view, decision)
    else:
      body, pending = st.render_session_delta(view, decision, self.cursor)
    prompt = body + "\n\n" + instruction
    for attempt in range(self.retries):
      if attempt:
        time.sleep(1.5 * attempt)              # brief backoff on transient failures
      text, sid = self._call(prompt)
      if text is None:
        continue                               # subprocess/API failure -> retry
      try:
        s, e = text.find("{"), text.rfind("}")
        obj = json.loads(text[s:e + 1])
        value = cast(obj[key])
        self.last_urgency = _clamp_urgency(obj.get("urgency"))
        if sid:
          self.session_id = sid                # capture/refresh the session to resume next turn
        self.cursor = pending                  # commit the narration cursor only on success
        self.transcript.append({"decision": decision, "prompt": prompt, "reply": text.strip()})
        return value, obj.get("reasoning", ""), obj
      except (ValueError, KeyError):
        self.last_error = "parse failure: %r" % text[:200]
    return None, "(model call failed after %d attempts: %s)" % (self.retries, self.last_error), {}

  def _account(self, env):
    """Fold one CLI envelope's token usage + cost into ``self.usage``. Keys mirror the
    Anthropic ``usage`` block; absent keys count as 0 so a thin envelope can't crash a run."""
    u = env.get("usage") or {}
    self.usage["calls"] += 1
    for k in ("input_tokens", "output_tokens",
              "cache_creation_input_tokens", "cache_read_input_tokens"):
      self.usage[k] += u.get(k, 0) or 0
    self.usage["cost_usd"] += env.get("total_cost_usd", 0) or 0

  def _env(self):
    """Child env. ``thinking_tokens >= 0`` caps extended thinking (0 disables it -- the big
    speedup); a negative value leaves it unset, so Claude Code's default thinking is on."""
    env = dict(os.environ)
    if self.thinking_tokens is not None and self.thinking_tokens >= 0:
      env["MAX_THINKING_TOKENS"] = str(self.thinking_tokens)
    return env

  def _call(self, prompt):
    """One headless `claude -p` call. Returns ``(reply_text, session_id)``, or
    ``(None, None)`` on a retryable failure (cause recorded on ``self.last_error``). On the
    opener (no session yet) we set the persona via ``--system-prompt``; on a resumed turn we
    pass ``--resume <id>`` instead -- the session already carries the persona and history."""
    cmd = ["claude", "-p", "--output-format", "json",
           "--model", self.model,
           "--strict-mcp-config",               # no --mcp-config => skip MCP startup
           "--tools", "",                       # no built-in tools: a player only answers
           "--setting-sources", ""]             # no user/project settings leak into a player
    # The three flags above keep a player's context PURE: nothing but our persona, our
    # rendered game state, and one SDK identity line (~166 tokens vs ~11.5k with the
    # default Claude Code harness prompt + tool definitions).
    if self.session_id is None:
      cmd += ["--system-prompt", self.system]
    else:
      cmd += ["--resume", self.session_id]      # continue this player's session; send only the delta
    # The prompt goes LAST, after a `--` sentinel: a delta can start with "--- Round N ..."
    # and the CLI would otherwise parse a positional beginning with `--` as an unknown option.
    cmd += ["--", prompt]
    try:
      proc = subprocess.run(
          cmd, cwd=_NEUTRAL_CWD, stdin=subprocess.DEVNULL, env=self._env(),
          capture_output=True, text=True, timeout=self.timeout)
    except subprocess.TimeoutExpired:
      self.last_error = "timeout after %ss" % self.timeout
      return None, None
    except OSError as exc:
      self.last_error = "OSError: %s" % exc
      return None, None
    if proc.returncode != 0:
      self.last_error = "exit %d: %s" % (proc.returncode, proc.stderr[:200])
      return None, None
    try:
      env = json.loads(proc.stdout)
    except ValueError:
      self.last_error = "unparseable CLI envelope"
      return None, None
    self._account(env)                         # count tokens before any is_error early-out
    if env.get("is_error"):
      self.last_error = "API error: %s" % str(env.get("result"))[:200]
      return None, None
    return env.get("result"), env.get("session_id")
