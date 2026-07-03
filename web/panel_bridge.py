"""Hosted-game public events -> 001 GameRecord (specs/002-host-local-game, T026).

The opt-in stats panel reuses the v1 assistant's replay pipeline wholesale: this
module only reshapes the hosted game's *public* record into the GameRecord that
``replay.replay_record`` already consumes (research R4). Zero probability code, and
nothing hidden ever enters the record -- it is built from ``PublicState`` alone.
"""

# Engine result vocabulary -> GameRecord result vocabulary.
RESULT_MAP = {"wire": "safe", "dud": "nothing", "bomb": "bomb"}


def game_record(pub, num_bad_override):
  """Build the GameRecord for a live ``PublicState``.

  ``num_bad_override`` is the *setup* override (or None for the official deal) --
  never the sampled truth, which is hidden information.

  Within a round every declaration precedes every cut, so ordering by round index
  reproduces the true public timeline. Rounds whose declarations are still being
  collected are not yet in ``declaration_history`` and so (correctly) not in the
  record: the panel has nothing new to say until the round's declarations close.
  """
  events = []
  for r, values in enumerate(pub.declaration_history):
    events.append({"type": "declarations", "values": [int(v) for v in values]})
    events.extend({"type": "cut", "player": cut["target"],
                   "result": RESULT_MAP[cut["result"]]}
                  for cut in pub.cut_log if cut["round"] == r)
  return {
      "setup": {"players": list(pub.player_names), "bomb": True,
                "numBadOverride": num_bad_override},
      "events": events,
  }
