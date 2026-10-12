"""Stockfish 16 engine helper with deterministic MultiPV 3 analysis.

Evaluates positions at fixed depth with Threads=1 and Hash=32MB.
Classifies moves into non-trivial vs only-moves vs trivial recaptures,
and skips decided/extreme evaluations (> 600 cp).
"""
import chess
import chess.engine

import config


class Analyzer:
    def __init__(self, path=None, depth=None):
        self.depth = depth or config.ENGINE_DEPTH
        self.engine = chess.engine.SimpleEngine.popen_uci(path or config.STOCKFISH_PATH)
        # Enforce single-thread determinism and lightweight memory
        self.engine.configure({"Threads": 1, "Hash": 32})

    def close(self):
        try:
            self.engine.quit()
        except Exception:
            pass

    def analyse_position(self, board, played_move, last_move=None):
        """Analyze board at current ply before played_move is pushed.

        Returns a dict:
            score_cp: int, evaluation of best move in cp (clipped at [-1000, 1000])
            is_decided: bool, True if |score_cp| > config.MAX_EVAL_CP or mate
            is_only_move: bool, True if 1 legal move or top-1 >= ONLY_MOVE_GAP_CP ahead of top-2
            is_recapture: bool, True if played move is a direct recapture
            is_non_trivial: bool, True if position has branching choices and not decided/only-move/recapture
            matches_top1: bool
            matches_top3: bool
            loss_cp: float, centipawn loss of played move clipped at config.MAX_MOVE_LOSS_CP
        """
        legal_count = board.legal_moves.count()
        if legal_count == 0:
            return None

        # Query MultiPV 3
        multipv_count = min(3, legal_count)
        try:
            info = self.engine.analyse(
                board,
                chess.engine.Limit(depth=self.depth),
                multipv=multipv_count,
            )
        except Exception:
            return None

        if not isinstance(info, list):
            info = [info]

        top_moves = []
        top_scores = []
        for item in info:
            pv = item.get("pv", [])
            if pv:
                top_moves.append(pv[0])
                score = item.get("score")
                cp = score.relative.score(mate_score=1000) if score is not None else 0
                top_scores.append(int(cp if cp is not None else 0))

        if not top_moves:
            return None

        best_score = top_scores[0]
        is_decided = abs(best_score) > config.MAX_EVAL_CP

        # Only-move check: either 1 legal move or top-1 is significantly better than top-2
        is_only_move = (legal_count == 1) or (
            len(top_scores) >= 2 and (top_scores[0] - top_scores[1]) >= config.ONLY_MOVE_GAP_CP
        )

        # Direct recapture check
        is_recapture = bool(
            last_move is not None
            and board.is_capture(played_move)
            and played_move.to_square == last_move.to_square
        )

        is_non_trivial = (
            (not is_decided)
            and (not is_only_move)
            and (not is_recapture)
            and (legal_count >= 3)
        )

        matches_top1 = bool(played_move == top_moves[0])
        matches_top3 = bool(played_move in top_moves[:3])

        # Centipawn loss calculation
        if matches_top1:
            loss_cp = 0.0
        elif played_move in top_moves:
            idx = top_moves.index(played_move)
            loss_cp = float(max(0, best_score - top_scores[idx]))
        else:
            # Played move was outside top-3: evaluate board after the move
            board.push(played_move)
            try:
                after_info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
                score_after = after_info.get("score")
                cp_after = score_after.relative.score(mate_score=1000) if score_after is not None else 0
                # after_info is from opponent's view, so relative score is negative of mover's score
                loss_cp = float(max(0, best_score + (cp_after if cp_after is not None else 0)))
            except Exception:
                loss_cp = float(config.MAX_MOVE_LOSS_CP)
            finally:
                board.pop()

        loss_cp = float(min(config.MAX_MOVE_LOSS_CP, loss_cp))

        return {
            "score_cp": best_score,
            "is_decided": is_decided,
            "is_only_move": is_only_move,
            "is_recapture": is_recapture,
            "is_non_trivial": is_non_trivial,
            "matches_top1": matches_top1,
            "matches_top3": matches_top3,
            "loss_cp": loss_cp,
        }
