"""The LLM agent -- the point of the project. Each decision is one headless `claude -p`
call that authenticates with the existing Claude Code login (no ANTHROPIC_API_KEY), with
the default system prompt replaced by a short player persona and run from a neutral cwd so
the repo's CLAUDE.md is not auto-injected (see sim/prototypes/cut_decision.py for the
proof-of-concept and SPEC.md §6 for the design defaults).

Stateless by construction: every call is a fresh, isolated subprocess fed the full
observable history via ``render_agent``. The model returns a JSON object; we validate and
the engine falls back on malformed output.

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
from state import render_agent

MODEL = "claude-haiku-4-5"   # fast model for play; reserve opus for deep-dives

SYSTEM = ("You are an expert, strategic Time Bomb player. Play to win for your secret "
          "team, reading the public declarations and cut history for tells. You think "
          "privately, then commit to one move.")

DECLARE_INSTRUCTION = (
    "Respond with ONLY a JSON object and nothing else:\n"
    '{"reasoning": "<your private thinking, shown to no one>", '
    '"declaration": <the wire count you announce>}')

CUT_INSTRUCTION = (
    "Respond with ONLY a JSON object and nothing else:\n"
    '{"reasoning": "<your private thinking, shown to no one>", '
    '"target": <the player index you cut>}')

# Neutral cwd shared by all agents, so no CLAUDE.md is auto-discovered into a player.
_NEUTRAL_CWD = tempfile.mkdtemp(prefix="tb_agent_")


class LLMAgent(Agent):
  name = "llm"

  def __init__(self, model=MODEL, system=SYSTEM, timeout=60, retries=3, thinking_tokens=0):
    self.model = model
    self.system = system
    self.timeout = timeout
    self.retries = retries
    self.thinking_tokens = thinking_tokens   # 0 disables extended thinking (the big speedup)
    self.last_reasoning = None
    self.last_error = None
    self.memory = []                         # this agent's own past decisions + reasoning

  def describe(self):
    d = super().describe()
    d.update(model=self.model, thinking_tokens=self.thinking_tokens,
             timeout=self.timeout, retries=self.retries, system=self.system,
             declare_instruction=DECLARE_INSTRUCTION, cut_instruction=CUT_INSTRUCTION)
    return d

  def declare(self, view):
    prompt = render_agent(view, "declare") + self._memory_block() + "\n\n" + DECLARE_INSTRUCTION
    value, self.last_reasoning = self._decide(prompt, "declaration")
    if value is not None:
      self._remember(view.public.round_index, "declared %d" % value)
    return value

  def choose_cut(self, view):
    prompt = render_agent(view, "cut") + self._memory_block() + "\n\n" + CUT_INSTRUCTION
    value, self.last_reasoning = self._decide(prompt, "target")
    if value is not None:
      who = view.public.player_names[value] if 0 <= value < view.public.num_players else value
      self._remember(view.public.round_index, "cut %s (Player %s)" % (who, value))
    return value

  # -- this agent's private running memory (no re-deriving each turn) --------

  def _memory_block(self):
    if not self.memory:
      return ""
    return ("\n\nYOUR OWN PRIVATE NOTES from earlier this game (your past moves and the "
            "reasoning behind them — build on these instead of re-analysing from scratch):\n"
            + "\n".join(self.memory))

  def _remember(self, round_index, action):
    self.memory.append("- [Round %d] You %s. Your reasoning then: %s"
                       % (round_index + 1, action, self.last_reasoning))

  # -- one decision: state -> text -> validated action, with retries --------

  def _decide(self, prompt, key):
    """Return ``(value, reasoning)``. ``value`` is ``None`` after all retries fail, so the
    engine's validation falls back to a safe legal default."""
    for attempt in range(self.retries):
      if attempt:
        time.sleep(1.5 * attempt)              # brief backoff on transient failures
      text = self._call(prompt)
      if text is None:
        continue                               # subprocess/API failure -> retry
      try:
        s, e = text.find("{"), text.rfind("}")
        obj = json.loads(text[s:e + 1])
        return int(obj[key]), obj.get("reasoning", "")
      except (ValueError, KeyError):
        self.last_error = "parse failure: %r" % text[:200]
    return None, "(model call failed after %d attempts: %s)" % (self.retries, self.last_error)

  def _env(self):
    """Child env. ``thinking_tokens >= 0`` caps extended thinking (0 disables it -- the big
    speedup); a negative value leaves it unset, so Claude Code's default thinking is on."""
    env = dict(os.environ)
    if self.thinking_tokens is not None and self.thinking_tokens >= 0:
      env["MAX_THINKING_TOKENS"] = str(self.thinking_tokens)
    return env

  def _call(self, prompt):
    """One headless `claude -p` call. Returns the model's reply text, or ``None`` on a
    retryable failure (with the cause recorded on ``self.last_error``)."""
    try:
      proc = subprocess.run(
          ["claude", "-p", prompt,
           "--output-format", "json",
           "--system-prompt", self.system,
           "--model", self.model,
           "--strict-mcp-config"],            # no --mcp-config => skip MCP startup
          cwd=_NEUTRAL_CWD, stdin=subprocess.DEVNULL,
          env=self._env(),
          capture_output=True, text=True, timeout=self.timeout)
    except subprocess.TimeoutExpired:
      self.last_error = "timeout after %ss" % self.timeout
      return None
    except OSError as exc:
      self.last_error = "OSError: %s" % exc
      return None
    if proc.returncode != 0:
      self.last_error = "exit %d: %s" % (proc.returncode, proc.stderr[:200])
      return None
    try:
      env = json.loads(proc.stdout)
    except ValueError:
      self.last_error = "unparseable CLI envelope"
      return None
    if env.get("is_error"):
      self.last_error = "API error: %s" % str(env.get("result"))[:200]
      return None
    return env.get("result")
