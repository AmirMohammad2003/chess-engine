import datetime
import sys
import time
from functools import lru_cache

import chess
import chess.pgn
import chess.polyglot
import numpy as np
from chess.polyglot import zobrist_hash
from numba import float32, int32, njit

from tables import piece_tables, piece_values

TRANSPOSITION_TABLE_SIZE = 2**20
transposition_table = {}


@njit
def piece_value(
    piece_type: int, color: bool, square: int, p_vals: np.ndarray, p_tables: np.ndarray
) -> float32:
    material = p_vals[piece_type - 1]
    table = p_tables[piece_type - 1]
    idx = 63 - square if color == chess.WHITE else square
    position_value = table[idx]
    value = material + position_value
    return value if color == chess.WHITE else -value


piece_tables_array = np.array([piece_tables[i] for i in range(1, 7)], dtype=np.int32)
piece_values_array = np.array([piece_values[p] for p in "PNBRQK"], dtype=np.int32)


class IncrementalEvaluator:
    def __init__(self, board):
        self.value = 0.0
        self.piece_map = board.piece_map()
        for sq, piece in self.piece_map.items():
            self.value += piece_value(
                piece.piece_type,
                piece.color,
                sq,
                piece_values_array,
                piece_tables_array,
            )

    def unmove(self, move, board):
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            try:
                self.value += piece_value(
                    captured.piece_type,
                    captured.color,
                    move.to_square,
                    piece_values_array,
                    piece_tables_array,
                )
            except Exception:
                pass
        moved_piece = board.piece_at(move.from_square)
        self.value += piece_value(
            moved_piece.piece_type,
            moved_piece.color,
            move.from_square,
            piece_values_array,
            piece_tables_array,
        )
        self.value -= piece_value(
            moved_piece.piece_type,
            moved_piece.color,
            move.to_square,
            piece_values_array,
            piece_tables_array,
        )

    def move(self, move, board):
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            try:
                self.value -= piece_value(
                    captured.piece_type,
                    captured.color,
                    move.to_square,
                    piece_values_array,
                    piece_tables_array,
                )
            except Exception:
                pass
        moved_piece = board.piece_at(move.from_square)
        self.value -= piece_value(
            moved_piece.piece_type,
            moved_piece.color,
            move.from_square,
            piece_values_array,
            piece_tables_array,
        )
        self.value += piece_value(
            moved_piece.piece_type,
            moved_piece.color,
            move.to_square,
            piece_values_array,
            piece_tables_array,
        )


@njit
def evaluate_board_static(
    value: float32, is_checkmate: bool, turn: bool, is_game_over: bool
) -> float32:
    if is_checkmate:
        return 1000.0 if turn else -1000.0
    elif is_game_over:
        return 0.0
    return value


def evaluate_board(board: chess.Board, evaluator: IncrementalEvaluator):
    return evaluate_board_static(
        evaluator.value, board.is_checkmate(), board.turn, board.is_game_over()
    )


def quiesce(board, alpha, beta, evaluator):
    stand_pat = evaluate_board(board, evaluator)
    if board.turn:
        if stand_pat >= beta:
            return stand_pat
        alpha = max(alpha, stand_pat)
    else:
        if stand_pat <= alpha:
            return stand_pat
        beta = min(beta, stand_pat)

    delta = 900
    if stand_pat + delta < alpha:
        return stand_pat

    for move in board.legal_moves:
        if board.is_capture(move):
            evaluator.move(move, board)
            board.push(move)
            score = quiesce(board, alpha, beta, evaluator)
            board.pop()
            evaluator.unmove(move, board)
            if board.turn:

                if score >= beta:
                    return score
                alpha = max(alpha, score)
            else:
                if score <= alpha:
                    return score
                beta = min(beta, score)
    return alpha if board.turn else beta


def minimax(board: chess.Board, alpha: float, beta: float, depth: int, evaluator):
    if depth == 0:
        return (
            quiesce(
                board,
                alpha,
                beta,
                evaluator,
            ),
            None,
        )

    board_key = zobrist_hash(board)
    if board_key in transposition_table:
        tt_depth, tt_value, tt_move = transposition_table[board_key]
        if tt_depth >= depth:
            return tt_value, tt_move

    moves = np.array([m for m in board.legal_moves])
    if len(moves) == 0:
        return evaluate_board(board, evaluator), None

    is_capture = np.array([board.is_capture(m) for m in moves], dtype=np.bool_)
    sort_indices = np.argsort(is_capture)[::-1]
    moves = moves[sort_indices]

    best_move = moves[0]
    value = -99999.0 if board.turn else 99999.0

    for move in moves:
        evaluator.move(move, board)
        board.push(move)
        new_value, _ = minimax(board, alpha, beta, depth - 1, evaluator)
        board.pop()
        evaluator.unmove(move, board)

        if board.turn:
            if new_value > value:
                value = new_value
                best_move = move
            if value >= beta:
                break
            alpha = max(alpha, value)
        else:
            if new_value < value:
                value = new_value
                best_move = move
            if value <= alpha:
                break
            beta = min(beta, value)

    if len(transposition_table) < TRANSPOSITION_TABLE_SIZE:
        transposition_table[board_key] = (depth, value, best_move)
    elif depth > transposition_table.get(board_key, (0, 0, None))[0]:
        transposition_table[board_key] = (depth, value, best_move)

    return value, best_move


def make_a_move(board: chess.Board, depth_: int = 3, time_limit: float = 20.0):
    try:
        return (
            chess.polyglot.MemoryMappedReader("src/bin/Perfect2023.bin")
            .weighted_choice(board)
            .move
        )
    except Exception:
        start_time = time.time()
        evaluator = IncrementalEvaluator(board)
        depth = min(depth_, 6)
        best_move = None
        while time.time() - start_time < time_limit and depth <= depth_:
            value, move = minimax(board, -99999.0, 99999.0, depth, evaluator)
            best_move = move
            depth += 1
            if time.time() - start_time > time_limit * 0.9:
                break
        return best_move


def uci():
    board = chess.Board()
    while True:
        command = input()
        if command == "quit":
            break
        elif command == "uci":
            print("id name badchessbot")
            print("id author badchessbot")
            print("uciok")
        elif command == "isready":
            print("readyok")
        elif command == "ucinewgame":
            board = chess.Board()
        elif command.startswith("position"):
            moves = command.split(" ")
            if moves[1] == "startpos":
                board = chess.Board()
                if len(moves) > 2 and moves[2] == "moves":
                    for move in moves[3:]:
                        board.push(chess.Move.from_uci(move))
            elif moves[1] == "fen":
                fen = " ".join(moves[2:8])
                board.set_fen(fen)
                if len(moves) > 8 and moves[8] == "moves":
                    for move in moves[9:]:
                        board.push(chess.Move.from_uci(move))
        elif command.startswith("go"):
            depth = 20
            if "depth" in command:
                depth = int(command.split(" ")[2])
            move = make_a_move(board, depth)
            print(f"bestmove {move.uci()}")
        elif command == "print":
            print(board)


def start():
    game = chess.pgn.Game()
    game.headers["Event"] = "Testing"
    game.headers["Site"] = "Don't have one"
    game.headers["Date"] = str(datetime.datetime.now().date())
    game.headers["Round"] = 1
    game.headers["White"] = "badchessbot"
    game.headers["Black"] = "badchessbot"
    board = chess.Board()
    history = []
    while not board.is_game_over():
        move = make_a_move(board, 4)
        history.append(move)
        board.push(move)
        print(board)
        print()

    game.add_line(history)
    game.headers["Result"] = str(board.result())
    print(game, file=sys.stderr)


if __name__ == "__main__":
    uci()
