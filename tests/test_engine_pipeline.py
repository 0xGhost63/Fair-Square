"""Unit tests for the move-level Stockfish MultiPV 3 engine pipeline."""
import io

import chess
import chess.pgn
import numpy as np

import config
from src.engine import Analyzer
from src.features import (
    ENGINE_FEATURE_NAMES,
    extract_game_engine_features,
    extract_single_side_engine_features,
)


def test_engine_determinism():
    """Verify that Analyzer with Threads=1 yields 100% deterministic output across calls."""
    analyzer = Analyzer(depth=10)
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    try:
        res1 = analyzer.analyse_position(board, move)
        res2 = analyzer.analyse_position(board, move)
        assert res1["score_cp"] == res2["score_cp"]
        assert res1["loss_cp"] == res2["loss_cp"]
        assert res1["matches_top1"] == res2["matches_top1"]
        assert res1["matches_top3"] == res2["matches_top3"]
    finally:
        analyzer.close()


def test_decided_and_only_move_classification():
    """Verify that decided positions (> 600 cp) and forced moves are correctly flagged."""
    analyzer = Analyzer(depth=10)
    try:
        # Extreme decided position (Queen + Rook vs King)
        board_decided = chess.Board("8/2R5/8/8/8/2K5/2Q5/k7 w - - 0 1")
        m = chess.Move.from_uci("c2b2")
        res_decided = analyzer.analyse_position(board_decided, m)
        assert res_decided["is_decided"] is True

        # Forced move: legal_count == 1
        board_forced = chess.Board("r6k/8/8/8/8/2b5/8/K7 w - - 0 1")
        assert board_forced.legal_moves.count() == 1
        forced_move = list(board_forced.legal_moves)[0]
        res_forced = analyzer.analyse_position(board_forced, forced_move)
        assert res_forced["is_only_move"] is True
    finally:
        analyzer.close()


def test_single_pass_feature_extraction():
    """Verify single-pass feature extraction produces complete, finite feature dictionaries."""
    pgn_sample = (
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 "
        "8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 11. c4 c6 12. cxb5 axb5 13. Nc3 Bb7 "
        "14. Bg5 b4 15. Nb1 h6 16. Bh4 c5 17. dxe5 Nxe4 18. Bxe7 Qxe7 19. exd6 Qf6 "
        "20. Nbd2 Nxd6 21. Nc4 Nxc4 22. Bxc4 Nb6 23. Ne5 Rae8 24. Bxf7+ Rxf7 25. Nxf7 Rxe1+ 26. Qxe1 Kxf7 1-0"
    )
    game = chess.pgn.read_game(io.StringIO(pgn_sample))
    analyzer = Analyzer(depth=10)
    try:
        wf, bf = extract_game_engine_features(game, 1600, 1600, analyzer)
        assert wf is not None
        assert bf is not None
        for name in ENGINE_FEATURE_NAMES:
            assert name in wf and not np.isnan(wf[name])
            assert name in bf and not np.isnan(bf[name])
        assert wf["elo"] == 1600.0
        assert 0.0 <= wf["top1_match"] <= 1.0
        assert 0.0 <= wf["non_trivial_top3_match"] <= 1.0
    finally:
        analyzer.close()
