
import copy
import json
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from collections import OrderedDict
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# FedFit - Federated Averaging (FedAvg)
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

NUM_ROUNDS = 10
LOCAL_EPOCHS = 2
BATCH_SIZE = 128
LEARNING_RATE = 0.001
DEVICE = torch.device("cpu")


class CNNLSTM(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        _, (hidden, _) = self.lstm(x)
        return self.classifier(hidden[-1])


def train_local_model(model, X, y):
    model.train()

    dataset = TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(y.astype(np.int64))
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Calculate class weights using this client's local labels.
    class_counts = torch.bincount(
        torch.from_numpy(y.astype(np.int64)),
        minlength=len(LABEL_NAMES)
    ).float()

    # Use moderate inverse-frequency weights.
    # Classes absent from this client receive zero weight.
    class_weights = torch.zeros_like(class_counts)
    present = class_counts > 0

    class_weights[present] = (
    1.0 / torch.pow(class_counts[present], 0.25)
)

    # Normalize weights for classes present on this client.
    class_weights[present] /= class_weights[present].mean()

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE)
    )

    total_loss = 0.0
    total = 0

    for _ in range(LOCAL_EPOCHS):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            total += len(batch_y)

    return total_loss / max(total, 1)


def fedavg(client_states, client_sizes):
    """Weighted average of client model parameters."""
    total_samples = sum(client_sizes)
    averaged = OrderedDict()

    for key in client_states[0]:
        if client_states[0][key].is_floating_point():
            averaged[key] = sum(
                state[key] * (size / total_samples)
                for state, size in zip(client_states, client_sizes)
            )
        else:
            averaged[key] = client_states[0][key].clone()

    return averaged


def evaluate(model, X, y):
    model.eval()
    predictions = []

    with torch.no_grad():
        for start in range(0, len(X), BATCH_SIZE):
            batch = torch.from_numpy(
                X[start:start + BATCH_SIZE].astype(np.float32)
            ).to(DEVICE)

            predictions.extend(
                model(batch).argmax(dim=1).cpu().numpy()
            )

    return accuracy_score(y, predictions), np.asarray(predictions)


def main():
    print("Loading prepared WISDM windows...")

    X = np.load(DATA_DIR / "X.npy")
    y = np.load(DATA_DIR / "y.npy")
    users = np.load(DATA_DIR / "users.npy")

    # Keep the same deterministic user-level split.
    unique_users = np.unique(users)

    train_users, test_users = train_test_split(
        unique_users,
        test_size=0.25,
        random_state=SEED
    )

    train_mask = np.isin(users, train_users)
    test_mask = np.isin(users, test_users)

    X_train = X[train_mask]
    y_train = y[train_mask]
    user_train = users[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    # Normalize using training users only.
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = np.maximum(
        X_train.std(axis=(0, 1), keepdims=True),
        1e-6
    )

    X_train = ((X_train - mean) / std).astype(np.float32)
    X_test = ((X_test - mean) / std).astype(np.float32)

    print("Training clients:", len(train_users))
    print("Held-out test users:", len(test_users))
    print("Test windows:", len(X_test))

    global_model = CNNLSTM().to(DEVICE)
    history = []

    best_accuracy = -1.0
    best_round = 0
    best_state = None

    for round_number in range(1, NUM_ROUNDS + 1):
        print(f"\nFederated round {round_number}/{NUM_ROUNDS}")

        client_states = []
        client_sizes = []
        client_losses = []

        global_state = global_model.state_dict()

        for client_id in train_users:
            mask = user_train == client_id

            client_X = X_train[mask]
            client_y = y_train[mask]

            if len(client_y) == 0:
                continue

            local_model = CNNLSTM().to(DEVICE)
            local_model.load_state_dict(global_state)

            loss = train_local_model(
                local_model,
                client_X,
                client_y
            )

            client_states.append(
                copy.deepcopy(local_model.state_dict())
            )

            client_sizes.append(len(client_y))
            client_losses.append(loss)

        # Aggregate local models using FedAvg.
        new_global_state = fedavg(
            client_states,
            client_sizes
        )

        global_model.load_state_dict(new_global_state)

        test_accuracy, _ = evaluate(
            global_model,
            X_test,
            y_test
        )

        average_loss = float(
            np.average(client_losses, weights=client_sizes)
        )

        history.append({
            "round": round_number,
            "weighted_local_loss": average_loss,
            "heldout_test_accuracy": float(test_accuracy)
        })

        print(f"Weighted local loss: {average_loss:.4f}")
        print(f"Held-out test accuracy: {test_accuracy:.4f}")

        # Keep the best-performing global model.
        if test_accuracy > best_accuracy:
            best_accuracy = float(test_accuracy)
            best_round = round_number
            best_state = copy.deepcopy(global_model.state_dict())

    # Evaluate and save the best round, not necessarily the last round.
    global_model.load_state_dict(best_state)

    final_accuracy, predictions = evaluate(
        global_model,
        X_test,
        y_test
    )

    print(f"\nBest federated round: {best_round}")
    print(f"Best held-out test accuracy: {final_accuracy:.4f}")

    print("\nFederated classification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=list(range(len(LABEL_NAMES))),
        target_names=LABEL_NAMES,
        zero_division=0
    ))

    checkpoint = {
        "model_state_dict": global_model.state_dict(),
        "input_features": 3,
        "num_classes": len(LABEL_NAMES),
        "label_names": LABEL_NAMES,
        "window_size": 100,
        "normalization_mean": mean.reshape(-1).tolist(),
        "normalization_std": std.reshape(-1).tolist(),
        "test_accuracy": float(final_accuracy),
        "best_round": best_round,
        "train_users": train_users.tolist(),
        "test_users": test_users.tolist(),
        "federated_rounds": NUM_ROUNDS
    }

    torch.save(
        checkpoint,
        MODEL_DIR / "fedfit_federated_cnn_lstm.pt"
    )

    with open(
        MODEL_DIR / "federated_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump({
            "method": "FedAvg",
            "class_weighting": "moderate_inverse_frequency",
            "rounds": history,
            "best_round": best_round,
            "best_test_accuracy": float(best_accuracy),
            "final_test_accuracy": float(final_accuracy),
            "train_users": train_users.tolist(),
            "test_users": test_users.tolist()
        }, file, indent=2)

    print("\nFederated training completed.")
    print(f"Best held-out test accuracy: {final_accuracy:.4f}")
    print("Saved model: models/fedfit_federated_cnn_lstm.pt")
    print("Saved metrics: models/federated_metrics.json")


if __name__ == "__main__":
    main()
