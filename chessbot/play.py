import argparse
import chess
import torch
import json
import numpy as np
from train import ChessNet

def encode_board(board):
    arr = np.zeros((8,8), dtype=np.int8)
    for square, piece in board.piece_map().items():
        arr[square//8][square%8] = piece.piece_type * (1 if piece.color else -1)
    return arr.flatten()

def pick_move(board, model, vocab):
    inv_vocab = {v: k for k, v in vocab.items()}
    x = torch.tensor(encode_board(board), dtype=torch.float32).unsqueeze(0)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).detach().numpy()[0]

    # Mask illegal moves
    legal_moves = [m.uci() for m in board.legal_moves]
    legal_idx = [vocab[m] for m in legal_moves if m in vocab]
    mask = np.zeros_like(probs)
    mask[legal_idx] = 1
    probs = probs * mask
    if probs.sum() == 0:
        return None
    move_idx = np.argmax(probs)
    return inv_vocab[move_idx]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fen", default="startpos")
    args = parser.parse_args()

    with open(args.vocab) as f:
        vocab = json.load(f)
    model = ChessNet(len(vocab))
    model.load_state_dict(torch.load(args.model))
    model.eval()

    board = chess.Board() if args.fen == "startpos" else chess.Board(args.fen)
    move = pick_move(board, model, vocab)
    print(f"Picked move: {move}")
