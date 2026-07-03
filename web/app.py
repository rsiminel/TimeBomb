"""Flask app for the Time Bomb web assistant (specs/001-web-cut-panel).

Transport only: serves the static page and exposes the one stateless endpoint of
contracts/api.md. All game logic lives in ``replay.py``; all math in the solver.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, request

import replay

app = Flask(__name__, static_folder="static", static_url_path="")


@app.get("/")
def index():
  return app.send_static_file("index.html")


@app.post("/api/panel")
def panel():
  record = request.get_json(silent=True)
  if record is None:
    return jsonify({"error": "body must be JSON"}), 400
  try:
    return jsonify(replay.replay_record(record)), 200
  except replay.RecordError as err:
    return jsonify({"error": str(err), "eventIndex": err.event_index}), 422


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="Time Bomb web assistant")
  parser.add_argument("--host", default="127.0.0.1",
                      help="bind address (0.0.0.0 to reach it from a phone on the LAN)")
  parser.add_argument("--port", type=int, default=5000)
  args = parser.parse_args()
  app.run(host=args.host, port=args.port)
