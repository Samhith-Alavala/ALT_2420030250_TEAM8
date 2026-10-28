
import json
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# FedFit: CNN-LSTM Activity Recognition
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = [
    "Downstairs", "Jogging", "Sitting",
    "Standing", "Upstairs", "Walking"
]

DEVICE = torch.device("cpu")
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001


class CNNLSTM(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Input: batch x time x sensor features
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)

        _, (hidden, _) = self.lstm(x)
        return self.classifier(hidden[-1])


def main():
    print("Loading prepared WISDM windows...")

    X = np.load(DATA_DIR / "X.npy")
    y = np.load(DATA_DIR / "y.npy")
    users = np.load(DATA_DIR / "users.npy")

    print("Data shape:", X.shape)
    print("Number of users:", len(np.unique(users)))

    # Split by user to reduce leakage between train and test sets.
    unique_users = np.unique(users)
    train_users, test_users = train_test_split(
        unique_users,
        test_size=0.25,
        random_state=SEED
    )

    train_mask = np.isin(users, train_users)
    test_mask = np.isin(users, test_users)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # Use training data only to estimate normalization values.
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.maximum(std, 1e-6)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    print("Training windows:", len(X_train))
    print("Testing windows:", len(X_test))
    print("Training users:", train_users.tolist())
    print("Testing users:", test_users.tolist())

    train_dataset = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.int64))
    )

    test_tensor = torch.from_numpy(X_test.astype(np.float32))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = CNNLSTM().to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )
    criterion = nn.CrossEntropyLoss()

    print("\nStarting CNN-LSTM training...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            correct += (logits.argmax(dim=1) == batch_y).sum().item()
            total += len(batch_y)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Loss: {total_loss / total:.4f} | "
            f"Training accuracy: {correct / total:.4f}"
        )

    print("\nEvaluating on held-out users...")
    model.eval()

    predictions = []
    with torch.no_grad():
        for start in range(0, len(test_tensor), BATCH_SIZE):
            batch_x = test_tensor[start:start + BATCH_SIZE]
            logits = model(batch_x)
            predictions.extend(logits.argmax(dim=1).cpu().numpy())

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nTest accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    report = classification_report(
        y_test,
        predictions,
        labels=list(range(len(LABEL_NAMES))),
        target_names=LABEL_NAMES,
        zero_division=0,
        output_dict=True
    )
    print(classification_report(
        y_test,
        predictions,
        labels=list(range(len(LABEL_NAMES))),
        target_names=LABEL_NAMES,
        zero_division=0
    ))

    print("Confusion matrix:")
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=list(range(len(LABEL_NAMES)))
    )
    print(matrix)

    # Save weights and preprocessing parameters for future predictions.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_features": 3,
            "num_classes": len(LABEL_NAMES),
            "label_names": LABEL_NAMES,
            "window_size": 100,
            "normalization_mean": mean.reshape(-1).tolist(),
            "normalization_std": std.reshape(-1).tolist(),
            "test_accuracy": float(accuracy),
            "test_users": test_users.tolist()
        },
        MODEL_DIR / "fedfit_cnn_lstm.pt"
    )

    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "test_accuracy": float(accuracy),
                "classification_report": report,
                "confusion_matrix": matrix.tolist(),
                "train_users": train_users.tolist(),
                "test_users": test_users.tolist(),
                "epochs": EPOCHS
            },
            file,
            indent=2
        )

    print("\nSaved trained model: models/fedfit_cnn_lstm.pt")
    print("Saved evaluation metrics: models/metrics.json")


if __name__ == "__main__":
    main()
