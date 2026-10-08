"""Optional Stockfish helper. Measures how close each played move is to the engine's best move."""
import chess.engine

import config


class Analyzer:
    def __init__(self, path=None, depth=None):
        self.depth = depth or config.ENGINE_DEPTH
        self.engine = chess.engine.SimpleEngine.popen_uci(path or config.STOCKFISH_PATH)

    def close(self):
        self.engine.quit()

    def best(self, board):
        # score is from the point of view of the side to move
        info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
        pv = info.get("pv", [])
        score = info.get("score")
        cp = score.relative.score(mate_score=1000) if score is not None else 0
        best_move = pv[0] if (pv and len(pv) > 0) else None
        return (cp if cp is not None else 0), best_move

    def move_stats(self, board, move):
        """Return (centipawn loss, matched engine best move)."""
        best_cp, best_move = self.best(board)
        if best_move is None:
            return 0.0, False
        if move == best_move:
            return 0.0, True
        board.push(move)
        cp_after, _ = self.best(board)
        board.pop()
        cp_after = cp_after if cp_after is not None else 0
        return float(min(max(0, best_cp + cp_after), 500)), False
