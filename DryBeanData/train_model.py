"""Train DryBeanClassifier and save state_dict + StandardScaler stats."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn

from model import CLASS_NAMES, FEATURE_NAMES, DryBeanClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Data" / "dry_bean.csv"
ARTIFACTS = BASE_DIR / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
WEIGHTS_PATH = ARTIFACTS / "drybean_classification_model.pth"
SCALER_PATH = ARTIFACTS / "scaler.json"


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    with DATA_PATH.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    X = np.array([[float(row[name]) for name in FEATURE_NAMES] for row in rows], dtype=np.float32)
    y_raw = np.array([row["Class"] for row in rows])
    encoder = LabelEncoder()
    encoder.fit(CLASS_NAMES)
    y = encoder.transform(y_raw)
    return X, y


def main() -> None:
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    torch.manual_seed(42)
    model = DryBeanClassifier()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 200
    for epoch in range(epochs):
        model.train()
        logits = model(X_train_t)
        loss = loss_fn(logits, y_train_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.inference_mode():
                train_pred = torch.argmax(logits, dim=1)
                train_acc = (train_pred == y_train_t).float().mean().item() * 100
                test_logits = model(X_test_t)
                test_loss = loss_fn(test_logits, y_test_t)
                test_pred = torch.argmax(test_logits, dim=1)
                test_acc = (test_pred == y_test_t).float().mean().item() * 100
            print(
                f"Epoch {epoch:3d} | Loss: {loss:.4f} | Acc: {train_acc:.2f}% | "
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
            )

    torch.save(model.state_dict(), WEIGHTS_PATH)
    SCALER_PATH.write_text(
        json.dumps(
            {
                "feature_names": FEATURE_NAMES,
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
                "class_names": CLASS_NAMES,
            },
            indent=2,
        )
    )
    print(f"Saved state_dict → {WEIGHTS_PATH}")
    print(f"Saved scaler     → {SCALER_PATH}")


if __name__ == "__main__":
    main()
