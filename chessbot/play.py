import argparse
import chess
import json
import numpy as np
import pickle
from train import extract_features
import time
import warnings
import logging
import sys
import io
from contextlib import redirect_stdout
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        
# Only suppress joblib parallel messages
logging.getLogger('joblib.parallel').setLevel(logging.ERROR)

# Only suppress specific UserWarnings related to parallel processing
warnings.filterwarnings("ignore", message=".*Parallel.*", category=UserWarning)

def encode_board(board):
    arr = np.zeros((8,8), dtype=np.int8)
    for square, piece in board.piece_map().items():
        arr[square//8][square%8] = piece.piece_type * (1 if piece.color else -1)
    return arr.flatten()

# Add this function to evaluate positions
# Only replace the evaluate_position function

def evaluate_position(board, model, vocab):
    """Enhanced position evaluation function with chess-specific knowledge"""
    # Terminal positions
    if board.is_checkmate():
        return -1000 if board.turn else 1000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
        
    # Get board representation and extract features
    board_vector = encode_board(board)
    features = extract_features(board_vector)
    
    # Get move probabilities
    probs = model.predict_proba([features])[0]
    
    # Material evaluation with positional bonuses
    piece_values = {
        chess.PAWN: 100, 
        chess.KNIGHT: 320, 
        chess.BISHOP: 330, 
        chess.ROOK: 500, 
        chess.QUEEN: 900, 
        chess.KING: 0
    }
    
    # Center squares for positional bonuses
    center = {chess.E4, chess.D4, chess.E5, chess.D5}
    extended_center = {chess.C3, chess.D3, chess.E3, chess.F3, 
                      chess.C4, chess.F4, chess.C5, chess.F5,
                      chess.C6, chess.D6, chess.E6, chess.F6}
                      
    # Initialize scores
    material_score = 0
    position_score = 0
    mobility_score = 0
    pawn_structure_score = 0
    king_safety_score = 0
    
    # 1. Material and Position evaluation
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece:
            continue
            
        # Base material value
        value = piece_values[piece.piece_type]
        material_score += value if piece.color else -value
        
        # Positional bonuses
        if square in center:
            position_score += 30 if piece.color else -30
        elif square in extended_center:
            position_score += 15 if piece.color else -15
            
        # Piece-specific bonuses
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        
        if piece.piece_type == chess.PAWN:
            # Passed pawns
            passed = True
            enemy_color = not piece.color
            if piece.color:  # White pawn
                for r in range(rank-1, -1, -1):
                    if board.piece_at(chess.square(file, r)) and board.piece_at(chess.square(file, r)).piece_type == chess.PAWN:
                        passed = False
                        break
                if passed:
                    # Bonus increases as pawn advances
                    position_score += (rank) * 10
            else:  # Black pawn
                for r in range(rank+1, 8):
                    if board.piece_at(chess.square(file, r)) and board.piece_at(chess.square(file, r)).piece_type == chess.PAWN:
                        passed = False
                        break
                if passed:
                    position_score -= (7-rank) * 10
                    
            # Doubled pawns penalty
            doubled = False
            if piece.color:  # White pawn
                for r in range(rank+1, 8):
                    if board.piece_at(chess.square(file, r)) and board.piece_at(chess.square(file, r)).piece_type == chess.PAWN and board.piece_at(chess.square(file, r)).color:
                        doubled = True
                        break
            else:  # Black pawn
                for r in range(rank-1, -1, -1):
                    if board.piece_at(chess.square(file, r)) and board.piece_at(chess.square(file, r)).piece_type == chess.PAWN and not board.piece_at(chess.square(file, r)).color:
                        doubled = True
                        break
            if doubled:
                pawn_structure_score -= 20 if piece.color else -20
                
        # Knight outpost bonuses
        if piece.piece_type == chess.KNIGHT:
            if (piece.color and rank <= 3) or (not piece.color and rank >= 4):
                is_outpost = True
                # Check if the square is defended by a friendly pawn
                if piece.color:  # White knight
                    if file > 0 and board.piece_at(chess.square(file-1, rank+1)) and board.piece_at(chess.square(file-1, rank+1)).piece_type == chess.PAWN and board.piece_at(chess.square(file-1, rank+1)).color:
                        position_score += 20
                    if file < 7 and board.piece_at(chess.square(file+1, rank+1)) and board.piece_at(chess.square(file+1, rank+1)).piece_type == chess.PAWN and board.piece_at(chess.square(file+1, rank+1)).color:
                        position_score += 20
                else:  # Black knight
                    if file > 0 and board.piece_at(chess.square(file-1, rank-1)) and board.piece_at(chess.square(file-1, rank-1)).piece_type == chess.PAWN and not board.piece_at(chess.square(file-1, rank-1)).color:
                        position_score -= 20
                    if file < 7 and board.piece_at(chess.square(file+1, rank-1)) and board.piece_at(chess.square(file+1, rank-1)).piece_type == chess.PAWN and not board.piece_at(chess.square(file+1, rank-1)).color:
                        position_score -= 20
    
    # 2. Mobility evaluation
    white_mobility = sum(1 for _ in board.legal_moves)
    # Save the turn and switch sides to calculate opponent mobility
    turn = board.turn
    board.turn = not turn
    black_mobility = sum(1 for _ in board.legal_moves)
    board.turn = turn  # Restore original turn
    
    mobility_score = (white_mobility - black_mobility) * 2
    
    # 3. King safety evaluation
    # Find kings
    white_king_square = board.king(True)
    black_king_square = board.king(False)
    
    # King tropism - distance of enemy pieces to king
    if white_king_square:
        white_king_file = chess.square_file(white_king_square)
        white_king_rank = chess.square_rank(white_king_square)
        
        # Penalize having enemy pieces close to king
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and not piece.color:  # Enemy black piece
                if piece.piece_type != chess.PAWN:  # Ignore pawns
                    file = chess.square_file(square)
                    rank = chess.square_rank(square)
                    distance = max(abs(white_king_file - file), abs(white_king_rank - rank))
                    if distance <= 2:
                        # Closer pieces are more dangerous
                        king_safety_score -= (3 - distance) * 15
    
    if black_king_square:
        black_king_file = chess.square_file(black_king_square)
        black_king_rank = chess.square_rank(black_king_square)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color:  # Enemy white piece
                if piece.piece_type != chess.PAWN:  # Ignore pawns
                    file = chess.square_file(square)
                    rank = chess.square_rank(square)
                    distance = max(abs(black_king_file - file), abs(black_king_rank - rank))
                    if distance <= 2:
                        king_safety_score += (3 - distance) * 15
    
    # Combine all evaluation components
    total_score = (
        material_score * 1.0 + 
        position_score * 0.5 + 
        mobility_score * 0.2 + 
        pawn_structure_score * 0.3 + 
        king_safety_score * 0.7
    )
    
    # Add model confidence (smaller weight)
    best_move_confidence = np.max(probs) if len(probs) > 0 else 0
    total_score += best_move_confidence * 50  # Add model's confidence with smaller weight
    
    # Normalize and return from white's perspective
    return total_score / 1000.0

# Add minimax look-ahead function
def pick_move_with_lookahead(board, model, vocab, depth=2, max_time=5):
    """Pick a move with minimax lookahead"""
    legal_moves = list(board.legal_moves)
    best_score = float('-inf')
    best_move = None
    start_time = time.time()
    
    for i, move in enumerate(legal_moves):
        # Time check
        if time.time() - start_time > max_time:
            print(f"Time limit reached after analyzing {i}/{len(legal_moves)} moves")
            break
            
        # Make the move
        board.push(move)
        
        # Calculate score using minimax - SUPPRESS OUTPUT HERE
        with SuppressOutput():
            score = -minimax(board, depth-1, float('-inf'), float('inf'), False, model, vocab, start_time, max_time)
        
        # Undo the move
        board.pop()
        
        # Update best move
        if score > best_score:
            best_score = score
            best_move = move
    
    return best_move.uci() if best_move else None

def minimax(board, depth, alpha, beta, maximizing_player, model, vocab, start_time, max_time):
    """Minimax algorithm with alpha-beta pruning"""
    
    if time.time() - start_time > max_time:
        return evaluate_position(board, model, vocab)
    
    # Terminal state check
    if depth == 0 or board.is_game_over():
        return evaluate_position(board, model, vocab)
    
    if maximizing_player:
        max_eval = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            # Fixed call - added start_time and max_time parameters
            eval = minimax(board, depth-1, alpha, beta, False, model, vocab, start_time, max_time)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cutoff
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            # Fixed call - added start_time and max_time parameters
            eval = minimax(board, depth-1, alpha, beta, True, model, vocab, start_time, max_time)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cutoff
        return min_eval

# Add simple opening book
OPENING_BOOK = {
    "": ["e2e4", "d2d4", "c2c4", "g1f3"],
    "e2e4": ["e7e5", "c7c5", "e7e6", "c7c6"],
    "e2e4 e7e5": ["g1f3", "f1c4", "d2d4"],
    "e2e4 c7c5": ["g1f3", "c2c3", "b1c3"],
    "e2e4 e7e6": ["d2d4", "g1f3"],
    "e2e4 c7c6": ["d2d4", "b1c3"],
    "d2d4": ["d7d5", "g8f6", "e7e6", "c7c5"],
    "d2d4 d7d5": ["c2c4", "g1f3", "e2e3"],
    "d2d4 g8f6": ["c2c4", "g1f3", "c1g5"],
    "d2d4 e7e6": ["c2c4", "g1f3", "e2e3"],
    "c2c4": ["e7e5", "c7c5", "g8f6", "e7e6"],
    "g1f3": ["g8f6", "d7d5", "c7c5", "e7e6"],
    # Additional lines
    "e2e4 e7e5 g1f3": ["b8c6", "g8f6"],
    "d2d4 d7d5 c2c4": ["e7e6", "c7c6"],
    "d2d4 g8f6 c2c4": ["e7e6", "g7g6"],
}

def get_opening_move(board, move_history):
    """Get move from opening book if available with improved key generation"""
    # Generate key from the move history
    position_key = " ".join(move_history[-4:]) if move_history else ""
    
    # Try different length keys, starting with the longest
    while position_key and position_key not in OPENING_BOOK:
        position_key = " ".join(position_key.split()[1:])
    
    if position_key in OPENING_BOOK:
        legal_moves = [m.uci() for m in board.legal_moves]
        book_moves = [m for m in OPENING_BOOK[position_key] if m in legal_moves]
        if book_moves:
            return np.random.choice(book_moves)
    return None

def pick_move(board, model, vocab, move_history=[]):
    """Pick a chess move using multiple strategies"""
    # 1. Try opening book first (with extended move history)
    opening_move = get_opening_move(board, move_history)
    if opening_move:
        print("Using opening book")
        return opening_move
    
    # 2. Use variable depth minimax search based on game phase
    piece_count = len(board.piece_map())
    
    if piece_count > 25:  # Early game
        print("Using lookahead (depth=3)")
        return pick_move_with_lookahead(board, model, vocab, depth=3, max_time=10)
    elif piece_count > 15:  # Mid game
        print("Using lookahead (depth=4)")
        return pick_move_with_lookahead(board, model, vocab, depth=4, max_time=20)
    else:  # Late game
        print("Using lookahead (depth=5)")
        return pick_move_with_lookahead(board, model, vocab, depth=5, max_time=25)
    
def interactive_play(model, vocab):
    """Play an interactive game against the bot"""
    board = chess.Board()
    move_history = []
    
    print("Welcome to Chess Bot!")
    print("Enter moves in UCI format (e.g., e2e4)")
    print("Type 'quit' to exit")
    
    while not board.is_game_over():
        print("\n" + str(board))
        print(f"Turn: {'White' if board.turn else 'Black'}")
        
        if board.turn:  # Human's turn (White)
            while True:
                move_uci = input("Your move: ")
                if move_uci.lower() == 'quit':
                    return
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in board.legal_moves:
                        board.push(move)
                        move_history.append(move_uci)
                        break
                    else:
                        print("Illegal move, try again.")
                except ValueError:
                    print("Invalid move format, try again.")
        else:  # Bot's turn (Black)
            move_uci = pick_move(board, model, vocab, move_history)
            if move_uci:
                print(f"Bot plays: {move_uci}")
                board.push_uci(move_uci)
                move_history.append(move_uci)
            else:
                print("Bot couldn't find a legal move!")
                break
    
    print("\nGame over!")
    print(str(board))
    print(f"Result: {board.result()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fen", default="startpos")
    parser.add_argument("--interactive", action="store_true", 
                        help="Play an interactive game against the bot")
    args = parser.parse_args()

    with open(args.vocab) as f:
        vocab = json.load(f)
    
    with open(args.model, 'rb') as f:
        model = pickle.load(f)
    
    if args.interactive:
        interactive_play(model, vocab)
    else:
        board = chess.Board() if args.fen == "startpos" else chess.Board(args.fen)
        move = pick_move(board, model, vocab)
        print(f"Picked move: {move}")