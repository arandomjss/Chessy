"""
Generate visuals for the Chessy PPT slides.

Saves PNG files into ./slides_images/:
 - top_moves_hist.png         (histogram of most frequent moves)
 - piece_heatmap.png         (heatmap of piece frequencies per square)
 - feature_importance.png    (top feature importances from model)
 - move_history_example.png  (rendered SAN move list from first PGN game)
 - flow_diagram.png          (engine flow diagram)
 - architecture_diagram.png  (deployment/architecture sketch)

Usage:
    python scripts\generate_visuals.py

Notes:
 - Requires: python-chess, numpy, matplotlib, seaborn (optional), networkx, pillow
 - If a file (model/pgn/vocab) is missing the script will skip that visual and print a message.
"""

import os
import json
import pickle
import math
from collections import Counter

import chess.pgn
import chess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from PIL import Image, ImageDraw, ImageFont

# Paths (adjust if needed)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'data')
PGN_PATH = os.path.join(DATA_DIR, 'sample.pgn')
VOCAB_PATH = os.path.join(DATA_DIR, 'vocab.json')
MODEL_PATH = os.path.join(ROOT, 'chess_forest_model.pkl')
OUT_DIR = os.path.join(ROOT, 'slides_images')
os.makedirs(OUT_DIR, exist_ok=True)

# Helpers

def safe_load_pgn(pgn_path, max_games=None):
    games = []
    if not os.path.exists(pgn_path):
        print(f"PGN not found: {pgn_path}, skipping PGN-based visuals.")
        return games
    with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games.append(game)
            if max_games and len(games) >= max_games:
                break
    print(f"Loaded {len(games)} games from {pgn_path}")
    return games


def plot_top_moves_hist(games, out_path, top_n=30):
    # Count UCI moves across all games
    counter = Counter()
    for game in games:
        board = game.board()
        for mv in game.mainline_moves():
            counter[mv.uci()] += 1
            board.push(mv)

    most = counter.most_common(top_n)
    if not most:
        print('No moves found to plot top moves histogram.')
        return

    moves, counts = zip(*most)
    plt.figure(figsize=(12,6))
    sns.barplot(x=list(counts), y=list(moves), palette='viridis')
    plt.xlabel('Count')
    plt.title(f'Top {top_n} most frequent UCI moves (from PGN)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print('Saved', out_path)


def plot_piece_heatmap(games, out_path):
    # squares A1..H8 map to 0..63; we'll produce 8x8 counts
    counts = np.zeros((8,8), dtype=int)
    if not games:
        print('No games for piece heatmap')
        return
    for game in games:
        board = game.board()
        for mv in game.mainline_moves():
            # count pieces on board before the move
            for sq, piece in board.piece_map().items():
                r = 7 - chess.square_rank(sq)
                c = chess.square_file(sq)
                counts[r, c] += 1
            board.push(mv)
        # final position count
        for sq, piece in board.piece_map().items():
            r = 7 - chess.square_rank(sq)
            c = chess.square_file(sq)
            counts[r, c] += 1

    plt.figure(figsize=(6,6))
    sns.heatmap(counts, annot=False, cmap='magma', xticklabels=list('abcdefgh'), yticklabels=list(reversed([str(i) for i in range(1,9)])))
    plt.title('Piece presence frequency per square (aggregate)')
    plt.xlabel('File')
    plt.ylabel('Rank')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print('Saved', out_path)


def plot_feature_importance(model_path, out_path, top_n=20):
    if not os.path.exists(model_path):
        print(f'Model not found: {model_path}, skipping feature importance plot.')
        return
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    if not hasattr(model, 'feature_importances_'):
        print('Model has no feature_importances_, skipping.')
        return
    fi = model.feature_importances_
    # pick top indices
    idx = np.argsort(fi)[::-1][:top_n]
    labels = [f'f{i}' for i in idx]
    vals = fi[idx]

    plt.figure(figsize=(10,6))
    sns.barplot(x=vals, y=labels, palette='rocket')
    plt.xlabel('Importance')
    plt.title('Top feature importances (model)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print('Saved', out_path)


def render_move_history_image(games, out_path, max_moves=20):
    # Take first game and render SAN moves into an image
    if not games:
        print('No games to render move history example')
        return
    game = games[0]
    board = game.board()
    lines = []
    move_no = 1
    for mv in game.mainline_moves():
        san = board.san(mv)
        board.push(mv)
        if board.turn == chess.BLACK:
            # after white move
            lines.append(f"{move_no}. {san}")
        else:
            # black move appended to last line
            if lines:
                lines[-1] = lines[-1] + f" {san}"
            else:
                lines.append(f"{move_no}. ... {san}")
            move_no += 1
        if len(lines) >= max_moves:
            break

    text = '\n'.join(lines)
    # render to image using PIL
    width, height = 800, 200 + 20 * len(lines)
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('arial.ttf', 16)
    except Exception:
        font = ImageFont.load_default()
    margin = 20
    draw.text((margin, margin), 'Move history (SAN) — example', fill='black', font=font)
    draw.text((margin, margin+30), text, fill='black', font=font)
    img.save(out_path)
    print('Saved', out_path)


def draw_flow_diagram(out_path):
    G = nx.DiGraph()
    nodes = [
    ('Board', {'pos':(0,0)}),
    ('Feature\nExtractor', {'pos':(1,0)}),
    ('Random\nForest', {'pos':(2,0)}),
    ('Move\nCandidates', {'pos':(3,0)}),
    ('Minimax\nSearch', {'pos':(4,0)}),
    ('Selected\nMove', {'pos':(5,0)}),
    ]
    for n,p in nodes:
        G.add_node(n, **p)
    edges = [
        ('Board','Feature\nExtractor'),
        ('Feature\nExtractor','Random\nForest'),
        ('Random\nForest','Move\nCandidates'),
        ('Move\nCandidates','Minimax\nSearch'),
        ('Minimax\nSearch','Selected\nMove')
    ]
    G.add_edges_from(edges)

    pos = {n: data['pos'] for n,data in G.nodes(data=True)}
    plt.figure(figsize=(10,2.5))
    nx.draw(G, pos, with_labels=True, node_size=2500, node_color='#f0f0f0', arrowsize=20)
    plt.title('Engine flow diagram')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print('Saved', out_path)


def draw_architecture_diagram(out_path):
    # Simple schematic: User <-> GUI <-> Engine + Model files
    width, height = 800, 400
    img = Image.new('RGB', (width, height), color='white')
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('arial.ttf', 14)
    except Exception:
        font = ImageFont.load_default()

    # Boxes
    boxes = [
        ((50, 150), (250, 230), 'User\\n(Tkinter GUI)'),
        ((300, 60), (500, 140), 'Chess Engine\\n(pick_move/minimax)'),
        ((300, 200), (500, 280), 'Model files\\nchess_forest_model.pkl\\n data/vocab.json'),
        ((550, 150), (750, 230), 'Output\\n(Board state, Move)')
    ]
    for a,b,label in boxes:
        d.rectangle([a,b], outline='black', width=2, fill='#eef6ff')
        # center text
        tx = (a[0]+5, a[1]+10)
        d.multiline_text(tx, label, fill='black', font=font)

    # Arrows
    d.line([(250,190),(300,100)], fill='black', width=2)
    d.line([(500,100),(550,190)], fill='black', width=2)
    d.line([(500,240),(550,190)], fill='black', width=2)

    img.save(out_path)
    print('Saved', out_path)


if __name__ == '__main__':
    games = safe_load_pgn(PGN_PATH, max_games=200)

    # top moves histogram
    plot_top_moves_hist(games, os.path.join(OUT_DIR, 'top_moves_hist.png'), top_n=30)

    # piece heatmap
    plot_piece_heatmap(games, os.path.join(OUT_DIR, 'piece_heatmap.png'))

    # feature importance
    plot_feature_importance(MODEL_PATH, os.path.join(OUT_DIR, 'feature_importance.png'))

    # move history example
    render_move_history_image(games, os.path.join(OUT_DIR, 'move_history_example.png'))

    # engine flow diagram
    draw_flow_diagram(os.path.join(OUT_DIR, 'flow_diagram.png'))

    # architecture diagram
    draw_architecture_diagram(os.path.join(OUT_DIR, 'architecture_diagram.png'))

    print('\nAll done. Generated images are in:', OUT_DIR)
