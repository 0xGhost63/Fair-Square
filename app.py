"""Flask app: serves the web page and a small JSON API."""
import time

from flask import Flask, jsonify, render_template, request

import config
from src import chesscom, predict

app = Flask(__name__)
_cache = {}          # (username, n) -> (timestamp, result)
CACHE_SECONDS = 600


@app.route("/")
def index():
    return render_template("index.html", min_games=config.MIN_GAMES, max_games=config.MAX_GAMES, max_elo=config.MAX_ELO)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": predict.load_bundle() is not None})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", "")).strip()
    try:
        n = int(data.get("games", 30))
    except (TypeError, ValueError):
        return jsonify({"error": "Number of games must be a number."}), 400
    if not username or not username.replace("_", "").replace("-", "").isalnum():
        return jsonify({"error": "Enter a valid chess.com username."}), 400
    if not config.MIN_GAMES <= n <= config.MAX_GAMES:
        return jsonify({"error": f"Games must be between {config.MIN_GAMES} and {config.MAX_GAMES}."}), 400

    key = (username.lower(), n)
    if key in _cache and time.time() - _cache[key][0] < CACHE_SECONDS:
        return jsonify(_cache[key][1])
    try:
        result = predict.analyze_player(username, n)
    except chesscom.PlayerNotFound:
        return jsonify({"error": "That chess.com player was not found."}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except Exception:
        return jsonify({"error": "Could not reach chess.com. Please try again."}), 502
    _cache[key] = (time.time(), result)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
