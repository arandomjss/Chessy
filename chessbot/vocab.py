import argparse
import chess.pgn
import json

def build_vocab(pgn_file, out_file):
    moves = set()
    with open(pgn_file) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                moves.add(move.uci())
                board.push(move)
    moves = sorted(list(moves))
    with open(out_file, "w") as f:
        json.dump({m: i for i, m in enumerate(moves)}, f)
    print(f"Wrote {len(moves)} moves to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    build_vocab(args.pgn, args.out)
