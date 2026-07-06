"""Single source of ALL agent-facing copy — every string that can end up in an agent's
prompt lives here, so you can tune what the model sees in one place.

The *assembly* logic (which strings, in what order, with which values) stays in
``state.py`` (the renderer) and ``llm.py`` (the call); this module is only the words.

Most strings are printf ``%`` templates, filled by the renderer; the comment on each names
its fields, in order. Edit the wording freely — just keep the ``%`` placeholders (and their
order) intact. The action protocols (``*_INSTRUCTION``) and ``RULES`` are used verbatim, not
formatted (note the literal JSON braces in the instructions).
"""

# === Persona / system prompt ===============================================
SYSTEM = ("You are an expert, strategic Time Bomb player, playing to win for your secret "
          "team. Read the declarations, the table talk, and the cut results for tells. In "
          "your private reasoning, argue from the specific evidence in front of you — who "
          "declared what, who said what, what cuts revealed; then commit to one move. Keep "
          "your reasoning brief and concrete: never restate the rules or the visible record.")

# === Action protocols (used verbatim — the literal {...} is the JSON the model returns) ===
# Every reply carries an "urgency" bid: speaking turns are rationed, so each discussion
# pass calls only the highest recent bidders. (No reserved seat for the next cutter — the
# cut reply's own "message" field is their mic.) A skipped player spends no turn at all.
_URGENCY_FIELD = (
    '"urgency": <0-9 — your bid to speak in the next table-talk pass; only the '
    'highest bidders get a turn>')

DECLARE_INSTRUCTION = (
    "Respond with ONLY a JSON object and nothing else:\n"
    '{"reasoning": "<private thinking about THIS situation, 2-3 short sentences, '
    'shown to no one>", '
    '"declaration": <the wire count you announce>, ' + _URGENCY_FIELD + "}")

# A discuss turn is public words only — no private reasoning field, half the output tokens.
DISCUSS_INSTRUCTION = (
    "Respond with ONLY a JSON object and nothing else:\n"
    '{"message": "<one or two sentences you say OUT LOUD to the whole table; '
    'everyone hears and remembers it. Or \\"\\" to stay silent — the usual choice '
    'when you have nothing new to add>", ' + _URGENCY_FIELD + "}")

CUT_INSTRUCTION = (
    "Respond with ONLY a JSON object and nothing else:\n"
    '{"reasoning": "<private thinking about THIS situation, 2-3 short sentences, '
    'shown to no one>", '
    '"target": "<the name of the player whose card you cut>", '
    '"message": "<one short sentence you say OUT LOUD to the whole table; '
    'everyone hears and remembers it>", ' + _URGENCY_FIELD + "}")

# === Rules preamble (used verbatim; seeds every session opener) =============
RULES = (
    "TIME BOMB — HOW TO WIN:\n"
    "- Two secret teams: GOOD guys and BAD guys. Your role is fixed for the whole game.\n"
    "- When a face-down card is cut it turns out to be exactly one of three things:\n"
    "    - a WIRE — one of the cards the GOOD guys must cut to win;\n"
    "    - a DUD — a harmless blank; cutting it just wastes the cut;\n"
    "    - the BOMB — the single bomb card.\n"
    "  A DUD and the BOMB are NOT wires. 'Cutting a wire' always means a winning card.\n"
    "- Every round, ALL cards are gathered up, reshuffled, and dealt out fresh. So how many\n"
    "  wires you hold, and who holds the bomb, change each round and are independent from\n"
    "  one round to the next.\n"
    "- On a turn, whoever holds the wire-cutters cuts one OTHER player's face-down card,\n"
    "  revealing a wire, a dud, or the bomb. Whoever is cut takes the cutters next.\n"
    "- GOOD guys WIN by cutting ALL the wires before time runs out — that DEFUSES the\n"
    "  bomb.\n"
    "- BAD guys WIN if the BOMB is ever cut (it explodes; the game ends INSTANTLY in their\n"
    "  favour), OR if time runs out before every wire is found.\n"
    "- So BAD guys WANT the bomb cut and want cuts wasted on duds; GOOD guys want to find\n"
    "  the wires and must NOT cut the bomb. (Bad guys do NOT 'protect' the bomb.)\n"
    "- Hands shrink by one card each round (5 down to 2). A declaration is a player's\n"
    "  claim about how many wires they hold this round, and may be a lie.")

# === The three cut outcomes, as shown in the public record =================
RESULT_WORDS = {"wire": "a WIRE", "dud": "a dud", "bomb": "THE BOMB"}

# === Role & hand block =====================================================
ROLE_GOOD = ("You are %s — a GOOD GUY. You win when every WIRE is cut (the bomb is then "
             "defused), and you must never cut the bomb.")   # (name)
ROLE_BAD = ("You are %s — a BAD GUY. You win if the BOMB is cut, or if time runs out "
            "with wires still hidden.")                      # (name)
WIRE_DESC_ONE = "1 of them is a WIRE"
WIRE_DESC_MANY = "%d of them are WIRES"                       # (count)
BOMB_HELD = "You ARE holding the bomb this round."
BOMB_NOT_HELD = "You are NOT holding the bomb this round."
HAND_LINE = "Your hand this round: %d cards, %s. %s"          # (hand_size, wire_desc, bomb_line)
TABLE_LINE = "Table: %d players — %s; %s."                   # (num_players, roster, bad_count_phrase)
BAD_COUNT_SINGULAR = "there is exactly %d bad guy"           # (k)  -- k == 1
BAD_COUNT_PLURAL = "there are exactly %d bad guys"           # (k)
BAD_COUNT_UNCERTAIN = "the number of bad guys is uncertain: %s"   # (joined parts)
BAD_COUNT_PART = "%d (%.0f%%)"                               # (k, percent)

# === Section tags (the four labelled blocks of a full render) ==============
# XML-style tags rather than prose headers: Claude models are trained to respect
# tag-delimited structure, which helps smaller models keep the sections straight.
RULES_OPEN, RULES_CLOSE = "<rules>", "</rules>"
ROLE_OPEN, ROLE_CLOSE = "<your_role_and_hand>", "</your_role_and_hand>"
HISTORY_OPEN = "<public_record> (what players CLAIMED, SAID, and what cuts REVEALED)"
HISTORY_CLOSE = "</public_record>"
NOW_OPEN, NOW_CLOSE = "<decision>", "</decision>"
ASSISTANT_OPEN = ("<assistant_readout> (Bayesian probabilities computed from the PUBLIC "
                  "record only — declarations and cut results; it cannot see anyone's "
                  "role or hand, not even yours)")
ASSISTANT_CLOSE = "</assistant_readout>"

# === Per-round record (full render / session opener) =======================
YOU_MARKER = " (you)"                                        # appended to your own name
ROUND_HEADER = "Round %d (hand size %d):"                    # (round_no, hand_size)
ROUND_HEADER_CURRENT = "THIS ROUND — Round %d (hand size %d):"
DECLS_PENDING = "  Declarations: being made now, simultaneously — you don't see others' yet."
DECLS_NONE = "  Declarations: (none recorded)"
DECLS_LINE = "  Declared wire counts — %s"                   # (joined DECL_PART)
DECL_PART = "%s%s=%s"                                        # (name, you_marker_or_blank, value)
SAID_LINE = "  Said — %s"                                    # (joined STMT_PART)
STMT_PART = '%s%s: "%s"'                                     # (name, you_marker_or_blank, message)
CUTS_LINE = "  Cuts — %s"                                    # (joined CUT_PART [+ CUT_PART_SAID])
CUT_PART = "%d. %s cut %s → %s"                              # (n, cutter, target, result_word)
CUT_PART_SAID = ' — said: "%s"'                             # (message)  appended to a CUT_PART
CUTS_NONE_YET = "  Cuts so far this round: none yet."
SNAPSHOT_HIDDEN = "  Wires still hidden across all hands: %d."        # (n)
# Per-hand public bookkeeping (wires already found in that hand vs cards still face-down)
# -- pure arithmetic over the record above, tallied so no player has to re-derive it.
SNAPSHOT_HANDS = "  Hands now — %s"                          # (joined HAND_PART)
HAND_PART = "%s%s: %d found, %d face-down"                   # (name, you_marker_or_blank, found, left)

# === Decision asks (the NOW block) =========================================
ASK_DECLARE = ("YOUR TURN TO DECLARE. Announce a wire count from 0 to %d — the truth, "
               "or a bluff that serves your team.")          # (hand_size)
ASK_DISCUSS = ("YOUR TURN TO SPEAK to the whole table — it gets a word before every cut. "
               "%s. If you have nothing to add, stay silent.")   # (DISCUSS_NEXT[_YOU] phrase)
DISCUSS_NEXT = "%s cuts next, once this pass ends"               # (next cutter's name)
DISCUSS_NEXT_YOU = "YOU cut next, right after this pass"
ASK_CUT = ("YOUR TURN TO CUT — you hold the wire-cutters. Cut one OTHER player's "
           "face-down card. You may cut: %s")                # (legal target names)

# === Session delta (incremental, resumed turns) ============================
ROUND_BANNER = ("--- Round %d begins (hand size %d). All cards were collected, reshuffled, "
                "and dealt out fresh. ---")                  # (round_no, hand_size)
HAND_LINE_DELTA = "Your hand now: %d cards, %s. %s"          # (hand_size, wire_desc, bomb_line)
DELTA_DECLS = "Declared wire counts this round — %s"         # (joined DECL_PART)
DELTA_TALK = "Table talk since your last turn — %s"          # (joined STMT_PART)
DELTA_CUTS = "Cuts since your last turn — %s"                # (joined DELTA_CUT_PART [+ CUT_PART_SAID])
DELTA_CUT_PART = "%s cut %s → %s"                            # (cutter, target, result_word)
DELTA_HIDDEN = "Wires still hidden across all hands: %d."    # (n)
DELTA_HANDS = "Hands now — %s"                               # (joined HAND_PART)
DELTA_ASSISTANT = "Assistant readout (public info only): %s"  # (panel json)
