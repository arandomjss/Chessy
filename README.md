# ChessBot - RandomForest Chess Engine

This project trains a RandomForest ensemble to predict chess moves based on historical games and uses minimax search with enhanced evaluation.

## Steps to run:

1. Create a virtual environment
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate    # Windows
   pip install -r requirements.txt

2. python chessbot/vocab.py --pgn data/sample.pgn --out data/vocab.json

3. python chessbot/preprocess.py --pgn data/sample.pgn --vocab data/vocab.json --out data/train_data.npz --max_positions 100000 

4. python chessbot/train.py --data data/train_data.npz --vocab data/vocab.json --max_depth 12 --min_samples_leaf 10 --n_estimators 30

5. python chessbot/visualize.py --model chess_forest_model.pkl --vocab data/vocab.json (optional)

6. python chessbot/play.py --vocab data/vocab.json --model chess_forest_model.pkl --interactive

or just run the gui which is better since 6 will give out annoying parallel processing outputs warnings etc
7. python chess_gui.py