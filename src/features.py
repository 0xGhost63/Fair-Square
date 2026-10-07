"""Turns one player's side of one game into a row of numbers (features).

Clock features capture HOW the player spends time (cheaters often use
very even time per move). Engine features (optional) capture HOW GOOD the moves are.
"""
import chess
import numpy as np

import config

CLOCK_FEATURES = [
    "n_moves", "mean_t", "median_t", "cv_t", "log_std", "near_median",
    "frac_sub_sec", "max_ratio", "corr_legal", "capture_ratio", "late_early",
]
ENGINE_FEATURES = ["acpl", "top1_match", "blunder_rate"]


def parse_time_control(tc):
    """'300+3' -> (300, 3). Daily or unknown formats give (0, 0)."""
    try:
        base, _, inc = str(tc).partition("+")
        return int(base), int(inc or 0)
    except ValueError:
        return 0, 0


def collect_moves(game, color, analyzer=None):
    """Walk the game and record time spent, legal move count and capture flag for one side."""
    base, inc = parse_time_control(game.headers.get("TimeControl", ""))
    if base == 0:
        return None, 1.0
    prev_clock = {chess.WHITE: float(base), chess.BLACK: float(base)}
    board = game.board()
    rows, own_idx, engine_count = [], 0, 0
    for node in game.mainline():
        mover = board.turn
        clk = node.clock()
        if clk is None:
            return None, 1.0  # game has no clock data
        if mover == color:
            if own_idx >= config.SKIP_OPENING:
                row = {
                    "t": max(prev_clock[mover] - clk + inc, 0.0),
                    "legal": board.legal_moves.count(),
                    "cap": board.is_capture(node.move),
                }
                if analyzer and engine_count < config.ENGINE_MAX_MOVES:
                    row["loss"], row["top1"] = analyzer.move_stats(board, node.move)
                    engine_count += 1
                rows.append(row)
            own_idx += 1
        prev_clock[mover] = clk
        board.push(node.move)
    budget = max((base + 40 * inc) / 40.0, 0.5)  # average seconds available per move
    return rows, budget


def safe_corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def extract_features(game, color, analyzer=None):
    """Return a dict of features, or None if the game is unusable."""
    rows, budget = collect_moves(game, color, analyzer)
    if not rows or len(rows) < config.MIN_MOVES:
        return None
    t = np.array([r["t"] for r in rows])
    tn = t / budget                      # time normalised by the time control
    legal = np.array([r["legal"] for r in rows])
    cap = np.array([r["cap"] for r in rows])
    med = float(np.median(tn))
    half = len(tn) // 2
    mean = tn.mean() + 1e-9
    feats = {
        "n_moves": len(t),
        "mean_t": tn.mean(),
        "median_t": med,
        "cv_t": tn.std() / mean,                                   # spread of thinking time
        "log_std": float(np.log(t + 0.1).std()),                   # spread on a log scale
        "near_median": float(np.mean(np.abs(tn - med) <= 0.25 * med)) if med > 0 else 1.0,
        "frac_sub_sec": float(np.mean(t < 1.0)),
        "max_ratio": tn.max() / mean,
        "corr_legal": safe_corr(tn, legal),                        # do hard positions take longer?
        "capture_ratio": (tn[cap].mean() / (tn[~cap].mean() + 1e-9)) if cap.any() and (~cap).any() else 1.0,
        "late_early": (tn[half:].mean() + 1e-9) / (tn[:half].mean() + 1e-9),
    }
    if analyzer:
        scored = [r for r in rows if "loss" in r]
        if not scored:
            return None
        feats["acpl"] = float(np.mean([r["loss"] for r in scored]))
        feats["top1_match"] = float(np.mean([r["top1"] for r in scored]))
        feats["blunder_rate"] = float(np.mean([r["loss"] >= 200 for r in scored]))
    return {k: float(v) for k, v in feats.items()}
