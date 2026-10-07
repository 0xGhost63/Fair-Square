"""Unit tests for feature extraction. Run with:  pytest tests -q"""
import random

import chess
import chess.pgn

from src.features import extract_features, parse_time_control, CLOCK_FEATURES


def make_game(times, tc="300+0", seed=3):
    """Random legal game where each side spends the given seconds per move."""
    random.seed(seed)
    game = chess.pgn.Game()
    game.headers["TimeControl"] = tc
    base, inc = parse_time_control(tc)
    clocks = {chess.WHITE: base, chess.BLACK: base}
    node, board = game, game.board()
    for i in range(len(times) * 2):
        moves = list(board.legal_moves)
        if not moves:
            break
        move = random.choice(moves)
        node = node.add_variation(move)
        clocks[board.turn] = clocks[board.turn] - times[i // 2] + inc
        node.set_clock(clocks[board.turn])
        board.push(move)
    return game


def test_time_control_parsing():
    assert parse_time_control("300+3") == (300, 3)
    assert parse_time_control("600") == (600, 0)
    assert parse_time_control("1/86400") == (0, 0)


def test_all_features_present_and_finite():
    feats = extract_features(make_game([3.0] * 40), chess.WHITE)
    assert feats is not None
    for name in CLOCK_FEATURES:
        assert name in feats and feats[name] == feats[name]  # not NaN


def test_constant_time_has_low_spread():
    even = extract_features(make_game([4.0] * 40), chess.WHITE)
    uneven = extract_features(make_game([1, 12, 2, 20, 1, 6, 3, 15] * 5), chess.WHITE)
    assert even["cv_t"] < uneven["cv_t"]
    assert even["near_median"] > uneven["near_median"]


def test_short_game_rejected():
    assert extract_features(make_game([3.0] * 6), chess.WHITE) is None
