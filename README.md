# ChessBot Starter

Steps to run:

1. Create a virtual environment
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .\.venv\Scripts\activate    # Windows
   pip install -r requirements.txt

2. create vocab.json by vocab.py 
example python chessbot/vocab.py --pgn data/sample.pgn --out data/vocab.json

3. for preprocess python chessbot/preprocess.py --pgn data/sample.pgn --vocab data/vocab.json --out data/train_data.npz

4. for training python chessbot/train.py --data data/train_data.npz --vocab data/vocab.json