import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json

class ChessNet(nn.Module):
    def __init__(self, num_moves):
        super().__init__()
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, num_moves)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

def train(data_file, vocab_file, epochs=5, batch_size=128):
    data = np.load(data_file)
    X, y = torch.tensor(data["X"], dtype=torch.float32), torch.tensor(data["y"], dtype=torch.long)

    with open(vocab_file) as f:
        vocab = json.load(f)
    num_moves = len(vocab)

    model = ChessNet(num_moves)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        perm = torch.randperm(len(X))
        X, y = X[perm], y[perm]
        for i in range(0, len(X), batch_size):
            xb, yb = X[i:i+batch_size], y[i:i+batch_size]
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, loss {loss.item():.4f}")

    torch.save(model.state_dict(), "checkpoint.pth")
    print("Saved checkpoint.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    train(args.data, args.vocab, args.epochs, args.batch_size)
