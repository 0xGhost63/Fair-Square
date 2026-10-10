"""Central settings. Change values here, nowhere else."""
import os

MAX_ELO = 2000          # the app only supports players up to this rating
MIN_MOVES = 15          # minimum timed moves a game needs after the opening
SKIP_OPENING = 8        # first moves are often book or premoved, so ignore them
MIN_GAMES = 10          # smallest selectable number of games
MAX_GAMES = 100         # largest selectable number of games

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "cheat_model.joblib")
DATA_PATH = os.path.join(BASE_DIR, "data", "features.csv")

import shutil

STOCKFISH_PATH = (
    os.environ.get("STOCKFISH_PATH")
    or shutil.which("stockfish")
    or (os.path.join(BASE_DIR, "stockfish.exe") if os.path.exists(os.path.join(BASE_DIR, "stockfish.exe")) else None)
    or (os.path.join(BASE_DIR, "stockfish") if os.path.exists(os.path.join(BASE_DIR, "stockfish")) else None)
    or "/usr/bin/stockfish"
)
ENGINE_DEPTH = 10
MIN_DEPTH = 6
MAX_DEPTH = 14
ENGINE_MAX_MOVES = 30   # moves per game sent to the engine (keeps it fast)
