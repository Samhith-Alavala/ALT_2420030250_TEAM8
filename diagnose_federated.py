import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from pathlib import Path

DEVICE = torch.device("cpu")
MODEL_PATH = Path("models/fedfit_federated_gentle_69_50.pt")
DATA_DIR = Path("data/processed")

LABEL_NAMES = [
    "Downstairs", "Jogging", "Sitting",
    "Standing", "Upstairs", "Walking"
]

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
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, num_classes))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        _, (hidden, _) = self.lstm(x)
        return self.classifier(hidden[-1])

def main():
    checkpoint = torch.load(
        MODEL_PATH, map_location=DEVICE, weights_only=False
    )

    X = np.load(DATA_DIR / "X.npy")
    y = np.load(DATA_DIR / "y.npy")
    users = np.load(DATA_DIR / "users.npy")

    test_users = np.array(checkpoint["test_users"])
    mask = np.isin(users, test_users)

    X_test = X[mask].astype(np.float32)
    y_test = y[mask].astype(np.int64)

    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
    std[std == 0] = 1.0
    X_test = (X_test - mean) / std

    model = CNNLSTM(num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    predictions = []
    with torch.no_grad():
        for start in range(0, len(X_test), 256):
            batch = torch.from_numpy(X_test[start:start + 256])
            logits = model(batch)
            predictions.extend(logits.argmax(dim=1).cpu().numpy())

    predictions = np.asarray(predictions)
    print("\nCheckpoint best round:", checkpoint["best_round"])
    print("Saved best accuracy:", f'{checkpoint["test_accuracy"]:.2%}')
    print("Diagnostic accuracy:", f'{accuracy_score(y_test, predictions):.2%}')

    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print("Labels:", LABEL_NAMES)
    print(confusion_matrix(y_test, predictions, labels=range(6)))

    print("\nClassification report:")
    print(classification_report(
        y_test, predictions,
        labels=range(6),
        target_names=LABEL_NAMES,
        zero_division=0,
        digits=3
    ))

    upstairs_index = LABEL_NAMES.index("Upstairs")
    upstairs_rows = y_test == upstairs_index
    print("\nActual Upstairs windows:", int(upstairs_rows.sum()))
    print("Correctly predicted Upstairs:",
          int(((predictions == upstairs_index) & upstairs_rows).sum()))

    mistaken = predictions[upstairs_rows]
    print("Predictions for actual Upstairs windows:")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name:12s}: {int((mistaken == i).sum())}")

if __name__ == "__main__":
    main()

