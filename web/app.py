"""Flask app for the Time Bomb web assistant (specs/001-web-cut-panel,
specs/002-host-local-game).

Transport only: routes serve static pages and the endpoints of contracts/api.md.
All rules live in ``tbgame.engine.TableGame``; all probability in ``General.py``
(via ``replay.py`` / ``panel_bridge.py``); this module adds neither.
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))   # repo root, for `import tbgame` (a package)

from flask import Flask, jsonify, request

import replay
import game_service
from game_service import ActiveGameError

app = Flask(__name__, static_folder="static", static_url_path="")


@app.get("/")
def index():
  return app.send_static_file("index.html")


@app.get("/assistant/")
def assistant():
  # The v1 page, moved verbatim (T022); Flask redirects bare /assistant here, so
  # its relative styles.css/main.js resolve under /assistant/.
  return app.send_static_file("assistant/index.html")


@app.get("/play")
def play():
  return app.send_static_file("play/index.html")


@app.post("/api/panel")
def panel():
  record = request.get_json(silent=True)
  if record is None:
    return jsonify({"error": "body must be JSON"}), 400
  try:
    return jsonify(replay.replay_record(record)), 200
  except replay.RecordError as err:
    return jsonify({"error": str(err), "eventIndex": err.event_index}), 422


# ---------------------------------------------------------------------------
# Hosted game (specs/002-host-local-game)
# ---------------------------------------------------------------------------

@app.errorhandler(ActiveGameError)
def _handle_active_game_error(err):
  return jsonify({"error": err.reason}), err.status


@app.post("/api/game")
def create_game():
  setup = request.get_json(silent=True)
  if setup is None:
    return jsonify({"error": "body must be JSON"}), 400
  version = game_service.create_game(setup)
  return jsonify({"version": version}), 201


@app.get("/api/game")
def get_game():
  return jsonify(game_service.table_view()), 200


@app.delete("/api/game")
def delete_game():
  game_service.abandon_game()
  return "", 204


@app.post("/api/game/unlock")
def unlock_seat():
  body = request.get_json(silent=True)
  if not isinstance(body, dict):
    return jsonify({"error": "body must be JSON"}), 400
  view = game_service.unlock(body.get("seat"), body.get("version"))
  return jsonify(view), 200


@app.post("/api/game/lock")
def lock_seat():
  game_service.lock()
  return "", 204


@app.post("/api/game/intent")
def submit_intent():
  body = request.get_json(silent=True)
  if not isinstance(body, dict):
    return jsonify({"error": "body must be JSON"}), 400
  seat, kind, value = body.get("seat"), body.get("kind"), body.get("value")
  claim, version = body.get("claim"), body.get("version")
  version, events = game_service.submit_intent(seat, kind, value, claim, version)
  return jsonify({"version": version, "events": events}), 200


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="Time Bomb web assistant")
  parser.add_argument("--host", default="127.0.0.1",
                      help="bind address (0.0.0.0 to reach it from a phone on the LAN)")
  parser.add_argument("--port", type=int, default=5000)
  args = parser.parse_args()
  app.run(host=args.host, port=args.port)
