import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import json

# Add this at the beginning of visualize_model function:
def visualize_model(model_path, vocab_path, max_depth=3):
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Check if model is a RandomForest
    is_forest = hasattr(model, 'estimators_')
    if is_forest:
        print(f"RandomForest detected with {len(model.estimators_)} trees.")
        print("Visualizing the first tree only.")
        tree_to_plot = model.estimators_[0]
    else:
        tree_to_plot = model
    
    # Load vocabulary
    with open(vocab_path) as f:
        vocab = json.load(f)
    
    # Invert vocabulary for label names
    inv_vocab = {v: k for k, v in vocab.items()}
    
    # Get feature names
    n_features = model.n_features_in_
    feature_names = [f"f{i}" for i in range(n_features)]
    
    # Get class names for some common moves
    class_names = []
    for i in range(model.n_classes_):
        if i in inv_vocab:
            class_names.append(inv_vocab[i])
        else:
            class_names.append(f"move_{i}")
    
    # Plot the tree
    plt.figure(figsize=(20, 10))
    plot_tree(tree_to_plot, 
              max_depth=max_depth,
              feature_names=feature_names,
              class_names=class_names,
              filled=True,
              fontsize=10)
    plt.title(f"Decision Tree (max_depth={max_depth})")
    plt.tight_layout()
    plt.savefig("chess_tree_visualization.png", dpi=300)
    plt.close()
    
    print(f"Visualization saved as chess_tree_visualization.png")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        # Get top 20 important features
        top_n = min(20, len(model.feature_importances_))
        indices = np.argsort(model.feature_importances_)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importances')
        plt.bar(range(top_n), 
                [model.feature_importances_[i] for i in indices],
                align='center')
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=300)
        plt.close()
        
        print(f"Feature importance visualization saved as feature_importance.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--max_depth", type=int, default=3)
    args = parser.parse_args()
    
    visualize_model(args.model, args.vocab, args.max_depth)