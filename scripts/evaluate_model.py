"""
Evaluate the saved RandomForest model on the saved dataset and print summary metrics.

Usage:
    python scripts\evaluate_model.py

This script is safe for a moderately large dataset: it samples up to `MAX_SAMPLES` positions for evaluation to keep memory/time reasonable.
"""
import os
import time
import json
import pickle
import numpy as np
from collections import Counter

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
except Exception as e:
    print('Please install scikit-learn in the interpreter you run this with.')
    raise

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'data')
MODEL_PATH = os.path.join(ROOT, 'chess_forest_model.pkl')
NPZ_PATH = os.path.join(DATA_DIR, 'train_data.npz')
VOCAB_PATH = os.path.join(DATA_DIR, 'vocab.json')

MAX_SAMPLES = 10000  # cap evaluation size
RANDOM_SEED = 42

def topk_accuracy(probs, y_true, ks=(1,3,5)):
    # probs: n_samples x n_classes
    idx_sorted = np.argsort(probs, axis=1)[:, ::-1]  # descending indices
    results = {}
    for k in ks:
        topk = idx_sorted[:, :k]
        # check if true label is among topk
        hits = (topk == y_true.reshape(-1,1)).any(axis=1)
        results[f'top_{k}'] = float(np.mean(hits))
    return results


def main():
    summary = {}

    if not os.path.exists(MODEL_PATH):
        print(f'Model file not found at {MODEL_PATH}. Cannot evaluate.')
        return
    if not os.path.exists(NPZ_PATH):
        print(f'Data file not found at {NPZ_PATH}. Cannot evaluate.')
        return

    # Load data
    print('Loading dataset...')
    data = np.load(NPZ_PATH, allow_pickle=True)
    if 'X' in data and 'y' in data:
        X = data['X']
        y = data['y']
    else:
        # fallback to first two arrays
        keys = list(data.keys())
        if len(keys) >= 2:
            X = data[keys[0]]
            y = data[keys[1]]
        else:
            raise RuntimeError('Unexpected .npz format; expected X and y arrays')

    n_total = len(y)
    summary['n_total_positions'] = int(n_total)

    # sample to limit eval cost
    rng = np.random.RandomState(RANDOM_SEED)
    if n_total > MAX_SAMPLES:
        idx = rng.choice(n_total, size=MAX_SAMPLES, replace=False)
        Xs = X[idx]
        ys = y[idx]
    else:
        Xs = X
        ys = y

    # cast to float32 to save memory (if numeric)
    try:
        Xs = Xs.astype(np.float32)
    except Exception:
        pass

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=0.2, random_state=RANDOM_SEED)

    summary['n_eval_samples'] = int(len(y_test))

    # load vocab
    vocab = None
    if os.path.exists(VOCAB_PATH):
        try:
            with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
                import json
                vocab = json.load(f)
        except Exception:
            vocab = None

    # load model
    print('Loading model...')
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    # Predict & time
    print('Running inference...')
    t0 = time.time()
    try:
        probs = model.predict_proba(X_test)
        preds = np.argmax(probs, axis=1)
    except Exception as e:
        # fallback to predict only
        t0 = time.time()
        preds = model.predict(X_test)
        probs = None
    t1 = time.time()

    elapsed = t1 - t0
    summary['inference_seconds_total'] = float(elapsed)
    summary['inference_seconds_per_sample'] = float(elapsed / len(y_test))

    # Accuracy
    acc = float(accuracy_score(y_test, preds))
    summary['top_1_accuracy'] = acc

    if probs is not None:
        tk = topk_accuracy(probs, y_test, ks=(1,3,5))
        summary.update(tk)

    # Top-move baselines
    cnt = Counter(ys)
    most_common = cnt.most_common(100)
    top100 = set([m for m,_ in most_common])
    mask_top100 = np.isin(y_test, list(top100))
    if mask_top100.sum() > 0:
        acc_top100 = float(np.mean(preds[mask_top100] == y_test[mask_top100]))
    else:
        acc_top100 = None

    summary['top100_fraction_in_eval'] = float(mask_top100.mean())
    summary['accuracy_on_top100_moves'] = acc_top100 if acc_top100 is None else float(acc_top100)

    # Per-class coverage: how many classes seen in eval
    n_classes_seen = int(len(set(y_test.tolist())))
    summary['n_classes_seen_in_eval'] = n_classes_seen

    # Basic class balance for top labels
    top10 = cnt.most_common(10)
    summary['top10_train_distribution'] = [{ 'move_index': int(k), 'count': int(v) } for k,v in top10]

    # Print human-friendly summary
    print('\n=== Evaluation summary ===')
    print(json.dumps(summary, indent=2))

    # If vocab present, show example mapping for top-5 predicted classes
    if vocab is not None and probs is not None:
        inv_vocab = {int(v):k for k,v in vocab.items()}
        # sample first 5 test rows
        print('\nExample predictions (first 5 eval samples):')
        for i in range(min(5, len(y_test))):
            true_idx = int(y_test[i])
            true_move = inv_vocab.get(true_idx, str(true_idx))
            top5 = np.argsort(probs[i])[::-1][:5]
            top5_moves = [inv_vocab.get(int(j), str(int(j))) for j in top5]
            print(f'  sample {i}: true={true_move} top5={top5_moves}')

if __name__ == '__main__':
    main()
