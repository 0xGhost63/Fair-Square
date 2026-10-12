"""Turns games into move-level engine features using Stockfish 16, with legacy clock support.

Features are computed strictly from move 11 onwards, excluding opening book.
Decided positions (> 600 cp) are skipped, loss is clipped at 300 cp,
recaptures are marked trivial, and only-moves are isolated.
"""
import chess
import numpy as np

import config

ENGINE_FEATURE_NAMES = [
    "non_trivial_top3_match",
    "longest_top3_streak",
    "only_move_match_rate",
    "top1_match",
    "acpl",
    "blunder_rate",
    "elo",
]

CLOCK_FEATURES = [
    "n_moves", "mean_t", "median_t", "cv_t", "log_std", "near_median",
    "frac_sub_sec", "max_ratio", "corr_legal", "capture_ratio", "late_early",
]


# ==============================================================================
# Legacy Clock Feature Helpers (for backward compatibility and tests)
# ==============================================================================

def parse_time_control(tc):
    """'300+3' -> (300, 3). Daily or unknown formats give (0, 0)."""
    try:
        base, _, inc = str(tc).partition("+")
        return int(base), int(inc or 0)
    except ValueError:
        return 0, 0


def collect_moves(game, color, analyzer=None):
    base, inc = parse_time_control(game.headers.get("TimeControl", ""))
    if base == 0:
        return None, 1.0
    prev_clock = {chess.WHITE: float(base), chess.BLACK: float(base)}
    board = game.board()
    rows, own_idx = [], 0
    for node in game.mainline():
        mover = board.turn
        clk = node.clock()
        if clk is None:
            return None, 1.0
        if mover == color:
            if own_idx >= config.SKIP_OPENING_MOVES:
                row = {
                    "t": max(prev_clock[mover] - clk + inc, 0.0),
                    "legal": board.legal_moves.count(),
                    "cap": board.is_capture(node.move),
                }
                rows.append(row)
            own_idx += 1
        prev_clock[mover] = clk
        board.push(node.move)
    budget = max((base + 40 * inc) / 40.0, 0.5)
    return rows, budget


def safe_corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def extract_features(game, color, analyzer=None):
    """Extract legacy clock features from a game."""
    rows, budget = collect_moves(game, color, analyzer)
    if not rows or len(rows) < config.MIN_MOVES:
        return None
    t = np.array([r["t"] for r in rows])
    tn = t / budget
    legal = np.array([r["legal"] for r in rows])
    cap = np.array([r["cap"] for r in rows])
    med = float(np.median(tn))
    half = len(tn) // 2
    mean = tn.mean() + 1e-9
    feats = {
        "n_moves": len(t),
        "mean_t": tn.mean(),
        "median_t": med,
        "cv_t": tn.std() / mean,
        "log_std": float(np.log(t + 0.1).std()),
        "near_median": float(np.mean(np.abs(tn - med) <= 0.25 * med)) if med > 0 else 1.0,
        "frac_sub_sec": float(np.mean(t < 1.0)),
        "max_ratio": tn.max() / mean,
        "corr_legal": safe_corr(tn, legal),
        "capture_ratio": (tn[cap].mean() / (tn[~cap].mean() + 1e-9)) if cap.any() and (~cap).any() else 1.0,
        "late_early": (tn[half:].mean() + 1e-9) / (tn[:half].mean() + 1e-9),
    }
    return {k: float(v) for k, v in feats.items()}


# ==============================================================================
# Pure Move-Level Stockfish 16 MultiPV 3 Features
# ==============================================================================

def _compute_side_features(records, elo):
    """Aggregate per-move engine stats for one side into the final feature row."""
    if not records or len(records) < 5:
        return None

    non_decided = [r for r in records if not r["is_decided"]]
    if not non_decided:
        return None

    # Streak: longest run of consecutive top-3 engine moves in non-decided positions
    max_streak = 0
    curr_streak = 0
    for r in non_decided:
        if r["matches_top3"]:
            curr_streak += 1
            if curr_streak > max_streak:
                max_streak = curr_streak
        else:
            curr_streak = 0

    non_trivial = [r for r in non_decided if r["is_non_trivial"]]
    only_moves = [r for r in non_decided if r["is_only_move"]]

    non_trivial_top3 = (
        float(np.mean([r["matches_top3"] for r in non_trivial]))
        if non_trivial else None
    )
    only_move_match = (
        float(np.mean([r["matches_top1"] for r in only_moves]))
        if only_moves else None
    )

    top1 = float(np.mean([r["matches_top1"] for r in non_decided]))
    acpl = float(np.mean([r["loss_cp"] for r in non_decided]))
    blunder = float(np.mean([r["loss_cp"] >= 200 for r in non_decided]))

    # Fallbacks if a short game has 0 non-trivial or 0 only-moves
    if non_trivial_top3 is None:
        non_trivial_top3 = top1
    if only_move_match is None:
        only_move_match = top1

    return {
        "non_trivial_top3_match": float(round(non_trivial_top3, 4)),
        "longest_top3_streak": float(max_streak),
        "only_move_match_rate": float(round(only_move_match, 4)),
        "top1_match": float(round(top1, 4)),
        "acpl": float(round(acpl, 2)),
        "blunder_rate": float(round(blunder, 4)),
        "elo": float(elo),
    }


def extract_game_engine_features(game, elo_w, elo_b, analyzer):
    """Single-pass engine evaluation for both White and Black.

    Walks the game once from move 11 onwards and queries the engine once per ply.
    Returns (white_features_dict, black_features_dict).
    """
    board = game.board()
    records = {chess.WHITE: [], chess.BLACK: []}
    last_move = None
    ply_idx = 0
    skip_plies = 2 * config.SKIP_OPENING_MOVES

    for node in game.mainline():
        move = node.move
        color = board.turn
        if ply_idx >= skip_plies:
            stats = analyzer.analyse_position(board, move, last_move=last_move)
            if stats is not None:
                records[color].append(stats)
        last_move = move
        board.push(move)
        ply_idx += 1

    w_feats = _compute_side_features(records[chess.WHITE], elo_w)
    b_feats = _compute_side_features(records[chess.BLACK], elo_b)
    return w_feats, b_feats


def extract_single_side_engine_features(game, target_color, elo, analyzer):
    """Engine feature extractor for a single target player side (used in live web app)."""
    board = game.board()
    records = []
    last_move = None
    ply_idx = 0
    skip_plies = 2 * config.SKIP_OPENING_MOVES

    for node in game.mainline():
        move = node.move
        color = board.turn
        if ply_idx >= skip_plies and color == target_color:
            stats = analyzer.analyse_position(board, move, last_move=last_move)
            if stats is not None:
                records.append(stats)
        last_move = move
        board.push(move)
        ply_idx += 1

    return _compute_side_features(records, elo)
