import json
import time

from flask import Flask, jsonify, render_template, request, Response, stream_with_context

import config
from src import chesscom, predict

app = Flask(__name__)
_cache = {}          # (username, n) -> (timestamp, result)
CACHE_SECONDS = 600


@app.route("/")
def index():
    return render_template(
        "index.html",
        min_games=config.MIN_GAMES,
        max_games=config.MAX_GAMES,
        max_elo=config.MAX_ELO,
        min_depth=config.MIN_DEPTH,
        max_depth=config.MAX_DEPTH,
        default_depth=config.ENGINE_DEPTH,
    )


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
    try:
        depth = int(data.get("depth", config.ENGINE_DEPTH))
    except (TypeError, ValueError):
        depth = config.ENGINE_DEPTH
    if not username or not username.replace("_", "").replace("-", "").isalnum():
        return jsonify({"error": "Enter a valid chess.com username."}), 400
    if not config.MIN_GAMES <= n <= config.MAX_GAMES:
        return jsonify({"error": f"Games must be between {config.MIN_GAMES} and {config.MAX_GAMES}."}), 400
    if not config.MIN_DEPTH <= depth <= config.MAX_DEPTH:
        depth = config.ENGINE_DEPTH

    key = (username.lower(), n, depth)
    if key in _cache and time.time() - _cache[key][0] < CACHE_SECONDS:
        return jsonify(_cache[key][1])
    try:
        result = predict.analyze_player(username, n, depth=depth)
    except chesscom.PlayerNotFound:
        return jsonify({"error": "That chess.com player was not found."}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except FileNotFoundError as e:
        return jsonify({"error": "Stockfish engine not found on system. Install it with: sudo pacman -S stockfish"}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error during analysis: {str(e)}"}), 502
    _cache[key] = (time.time(), result)
    return jsonify(result)


@app.route("/api/analyze/stream")
def analyze_stream():
    username = str(request.args.get("username", "")).strip()
    try:
        n = int(request.args.get("games", 30))
    except (TypeError, ValueError):
        return jsonify({"error": "Number of games must be a number."}), 400
    try:
        depth = int(request.args.get("depth", config.ENGINE_DEPTH))
    except (TypeError, ValueError):
        depth = config.ENGINE_DEPTH
    if not username or not username.replace("_", "").replace("-", "").isalnum():
        return jsonify({"error": "Enter a valid chess.com username."}), 400
    if not config.MIN_GAMES <= n <= config.MAX_GAMES:
        return jsonify({"error": f"Games must be between {config.MIN_GAMES} and {config.MAX_GAMES}."}), 400
    if not config.MIN_DEPTH <= depth <= config.MAX_DEPTH:
        depth = config.ENGINE_DEPTH

    def generate():
        key = (username.lower(), n, depth)
        if key in _cache and time.time() - _cache[key][0] < CACHE_SECONDS:
            cached = _cache[key][1]
            yield f"data: {json.dumps({'type': 'step', 'step': 4, 'text': 'Loaded from cache'})}\n\n"
            yield f"data: {json.dumps({'type': 'progress', 'current': cached['games_scored'], 'total': n, 'step': 4})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'result': cached})}\n\n"
            return

        try:
            for event in predict.analyze_player_stream(username, n, depth=depth):
                if event.get("type") == "done":
                    _cache[key] = (time.time(), event["result"])
                yield f"data: {json.dumps(event)}\n\n"
        except chesscom.PlayerNotFound:
            yield f"data: {json.dumps({'type': 'error', 'error': 'That chess.com player was not found.'})}\n\n"
        except ValueError as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        except RuntimeError as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        except FileNotFoundError:
            yield f"data: {json.dumps({'type': 'error', 'error': 'Stockfish engine not found on system. Install it with: sudo pacman -S stockfish'})}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': f'Error during analysis: {str(e)}'})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True)
