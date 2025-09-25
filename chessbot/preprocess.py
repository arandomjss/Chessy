import argparse
import chess.pgn
import numpy as np
import json
from train import extract_features

def encode_board(board):
    # Basic encoding: piece type * color
    arr = np.zeros((8,8), dtype=np.int8)
    for square, piece in board.piece_map().items():
        arr[square//8][square%8] = piece.piece_type * (1 if piece.color else -1)
    return arr.flatten()

def preprocess(pgn_file, vocab_file, out_file, max_positions=50000):
    with open(vocab_file) as f:
        vocab = json.load(f)
    X, y = [], []
    with open(pgn_file) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                X.append(encode_board(board))
                y.append(vocab.get(move.uci(), -1))
                board.push(move)
                if len(X) >= max_positions:
                    break
            if len(X) >= max_positions:
                break
    X, y = np.array(X), np.array(y)
    print("Extracting additional features...")
    X_features = np.array([extract_features(board) for board in X])
    np.savez(out_file, X=X_features, y=y)
    print(f"Saved {len(X_features)} positions with enhanced features to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_positions", type=int, default=50000)
    args = parser.parse_args()
    preprocess(args.pgn, args.vocab, args.out, args.max_positions)
