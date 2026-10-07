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

# Stockfish is optional. Only needed if the model was trained with engine features.
STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "/usr/bin/stockfish")
ENGINE_DEPTH = 10
ENGINE_MAX_MOVES = 30   # moves per game sent to the engine (keeps it fast)
