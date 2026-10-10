"""Downloads a player's recent games from the public chess.com API."""
import io

import chess
import chess.pgn
import requests

BASE = "https://api.chess.com/pub"
HEADERS = {"User-Agent": "ChessCheatDetector/1.0 (university project)"}


class PlayerNotFound(Exception):
    pass


def usable(g):
    """Keep rated standard games that have clock data."""
    return bool(
        g.get("rules") == "chess" and g.get("rated") and g.get("time_class") != "daily"
        and "%clk" in g.get("pgn", "")
    )


def stream_usable_games(username, max_archives=None):
    """Yield usable games (standard rated with clock data) one by one, newest first."""
    r = requests.get(f"{BASE}/player/{username}/games/archives", headers=HEADERS, timeout=15)
    if r.status_code == 404:
        raise PlayerNotFound(username)
    r.raise_for_status()
    archives = r.json().get("archives", [])
    if max_archives is not None:
        archives = archives[::-1][:max_archives]
    else:
        archives = archives[::-1]

    for url in archives:
        try:
            month = requests.get(url, headers=HEADERS, timeout=20)
            month.raise_for_status()
        except Exception:
            continue
        for g in reversed(month.json().get("games", [])):
            if usable(g):
                yield g


def fetch_recent_games(username, n, max_archives=None):
    games = []
    for g in stream_usable_games(username, max_archives=max_archives):
        games.append(g)
        if len(games) >= n:
            break
    return games


def side_of(g, username):
    """Return (color, rating, opponent) of the player in this game."""
    if g["white"]["username"].lower() == username.lower():
        return chess.WHITE, g["white"]["rating"], g["black"]["username"]
    return chess.BLACK, g["black"]["rating"], g["white"]["username"]


def parse_pgn(pgn_text):
    return chess.pgn.read_game(io.StringIO(pgn_text))
