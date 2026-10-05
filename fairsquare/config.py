import os

# System Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ENGINE_PATH = os.path.join(BASE_DIR, "bin", "stockfish-engine")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
CACHE_DB_PATH = os.path.join(CACHE_DIR, "stockfish_cache.sqlite")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "fair_square_model.joblib")
BASELINES_DIR = os.path.join(BASE_DIR, "baselines")
BASELINES_PATH = os.path.join(BASELINES_DIR, "rating_baselines.joblib")
DEFAULT_DATASET_PATH = os.path.join(BASE_DIR, "lichess_db_standard_rated_2018-01.pgn.zst")

# Stockfish Analysis Settings
DEFAULT_DEPTH = 10
DEFAULT_MULTIPV = 2
DEFAULT_THREADS = 1
DEFAULT_HASH_MB = 64
MAX_PARALLEL_WORKERS = 2  # Protect host device CPU/RAM

# Game Processing Bounds
MIN_GAME_MOVES = 14       # Ignore short games (<14 full moves)
BOOK_MOVE_CUTOFF = 10     # Skip first 10 moves (20 plies) as opening book
CHESSCOM_RATING_OFFSET = 100

# Rating Bands for Baseline Normalization
RATING_BANDS = [
    (0, 1200, "<1200"),
    (1200, 1400, "1200-1400"),
    (1400, 1600, "1400-1600"),
    (1600, 1800, "1600-1800"),
    (1800, 2000, "1800-2000"),
    (2000, 3000, "2000+"),
]

def get_rating_band_label(rating: float) -> str:
    r = float(rating)
    for low, high, label in RATING_BANDS:
        if low <= r < high:
            return label
    return "2000+"
