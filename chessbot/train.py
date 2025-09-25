import argparse
import json
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time

def extract_features(board_state):
    """Extract chess-specific features from the board state"""
    # Original board representation
    board_flat = board_state.copy()  # Keep the original features
    
    # Reshape to 8x8 to extract features
    board = board_flat.reshape(8, 8)
    
    # Initialize additional features
    features = []
    
    # 1. Material count and value (12 features)
    piece_values = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}  # Pawn to King values
    white_material = 0
    black_material = 0
    piece_counts = [0] * 12  # White and black pieces (6 types each)
    
    for i in range(8):
        for j in range(8):
            piece = board[i][j]
            if piece > 0:  # White pieces
                piece_type = piece
                white_material += piece_values[piece_type]
                piece_counts[piece_type - 1] += 1
            elif piece < 0:  # Black pieces
                piece_type = -piece
                black_material += piece_values[piece_type]
                piece_counts[6 + piece_type - 1] += 1
    
    # Add piece counts
    features.extend(piece_counts)
    
    # Add material advantage
    features.append(white_material - black_material)
    
    # 2. Center control (16 features - influence on central squares)
    central_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]
    for row, col in central_squares:
        # Count attackers and defenders of this square
        white_attackers = 0
        black_attackers = 0
        
        # Simplified approach: adjacent pieces contribute to control
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if 0 <= row + dr < 8 and 0 <= col + dc < 8:
                    piece = board[row + dr, col + dc]
                    if piece > 0:  # White piece
                        white_attackers += 1
                    elif piece < 0:  # Black piece
                        black_attackers += 1
        
        features.append(white_attackers)
        features.append(black_attackers)
    
    # 3. Development features
    # Count developed pieces (moved from starting position)
    knight_bishop_developed = [
        1 if board[7][1] == 0 else 0,  # White knight from b1 developed
        1 if board[7][6] == 0 else 0,  # White knight from g1 developed
        1 if board[7][2] == 0 else 0,  # White bishop from c1 developed
        1 if board[7][5] == 0 else 0,  # White bishop from f1 developed
        1 if board[0][1] == 0 else 0,  # Black knight from b8 developed
        1 if board[0][6] == 0 else 0,  # Black knight from g8 developed
        1 if board[0][2] == 0 else 0,  # Black bishop from c8 developed
        1 if board[0][5] == 0 else 0,  # Black bishop from f8 developed
    ]
    features.extend(knight_bishop_developed)
    
    # 4. King safety features
    # Count pieces around kings
    white_king_pos = None
    black_king_pos = None
    
    for i in range(8):
        for j in range(8):
            if board[i][j] == 6:  # White king
                white_king_pos = (i, j)
            elif board[i][j] == -6:  # Black king
                black_king_pos = (i, j)
    
    if white_king_pos:
        white_king_defenders = 0
        row, col = white_king_pos
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if 0 <= row + dr < 8 and 0 <= col + dc < 8:
                    piece = board[row + dr, col + dc]
                    if piece > 0 and piece != 6:  # White piece (not king itself)
                        white_king_defenders += 1
        features.append(white_king_defenders)
    else:
        features.append(0)
        
    if black_king_pos:
        black_king_defenders = 0
        row, col = black_king_pos
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if 0 <= row + dr < 8 and 0 <= col + dc < 8:
                    piece = board[row + dr, col + dc]
                    if piece < 0 and piece != -6:  # Black piece (not king itself)
                        black_king_defenders += 1
        features.append(black_king_defenders)
    else:
        features.append(0)
    
    # Combine all features with the original board representation
    return np.concatenate([board_flat, np.array(features)])

def train(data_file, vocab_file, max_depth=10, min_samples_leaf=10, n_estimators=50):
    start_time = time.time()
    
    # Load training data
    print(f"Loading data from {data_file}...")
    data = np.load(data_file)
    X, y = data["X"], data["y"]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Free up memory
    del X, y
    
    # Load vocabulary to get number of moves
    with open(vocab_file) as f:
        vocab = json.load(f)
    num_moves = len(vocab)
    
    print(f"Training RandomForest on {len(X_train)} samples with {num_moves} possible moves")
    
    # Train RandomForest classifier with memory-efficient parameters
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        criterion='entropy',
        class_weight='balanced',
        random_state=42,
        n_jobs=1,  # Use single CPU to reduce memory usage
        bootstrap=True,
        max_samples=0.2,  # Use only 20% of samples per tree
        verbose=1,
        warm_start=False
    )
    
    # Train the model in batches
    print("Training model...")
    batch_size = 20000  # Smaller batch size
    n_batches = (len(X_train) + batch_size - 1) // batch_size
    
    for batch in range(n_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, len(X_train))
        
        print(f"Training on batch {batch+1}/{n_batches} (samples {start_idx} to {end_idx})")
        model.fit(X_train[start_idx:end_idx], y_train[start_idx:end_idx])
    
    # Evaluate on a small subset of test data to avoid memory issues
    test_subset_size = min(5000, len(X_test))
    print(f"Evaluating on {test_subset_size} test samples...")
    
    X_test_subset = X_test[:test_subset_size]
    y_test_subset = y_test[:test_subset_size]
    
    # Free more memory
    del X_train, y_train
    
    y_pred = model.predict(X_test_subset)
    accuracy = accuracy_score(y_test_subset, y_pred)
    
    print("\n--- Model Evaluation ---")
    print(f"Test accuracy: {accuracy:.4f}")
    print("Note: Training accuracy calculation skipped to save memory")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        top_n = min(10, len(model.feature_importances_))
        indices = np.argsort(model.feature_importances_)[::-1][:top_n]
        print(f"\nTop {top_n} important features:")
        for i, idx in enumerate(indices):
            print(f"{i+1}. Feature {idx}: {model.feature_importances_[idx]:.6f}")
    
    # Save model
    print("\nSaving model...")
    with open("chess_forest_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print(f"\nTraining completed in {time.time() - start_time:.2f} seconds")
    print("Saved chess_forest_model.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--max_depth", type=int, default=15)
    parser.add_argument("--min_samples_leaf", type=int, default=10)
    parser.add_argument("--n_estimators", type=int, default=50, 
                        help="Number of trees in the forest")
    args = parser.parse_args()
    
    train(args.data, args.vocab, args.max_depth, args.min_samples_leaf, args.n_estimators)