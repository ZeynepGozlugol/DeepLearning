"""Train IrisClassifier and save state_dict to artifacts/iris_classifier.pth."""

from pathlib import Path

import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch import nn

from model import IrisClassifier

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
WEIGHTS_PATH = ARTIFACTS / "iris_classifier.pth"


def main() -> None:
    iris = load_iris()
    X = torch.tensor(iris.data, dtype=torch.float32)
    y = torch.tensor(iris.target, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    torch.manual_seed(42)
    model = IrisClassifier()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 200
    for epoch in range(epochs):
        model.train()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.inference_mode():
                train_pred = torch.argmax(logits, dim=1)
                train_acc = (train_pred == y_train).float().mean().item() * 100
                test_logits = model(X_test)
                test_loss = loss_fn(test_logits, y_test)
                test_pred = torch.argmax(test_logits, dim=1)
                test_acc = (test_pred == y_test).float().mean().item() * 100
            print(
                f"Epoch {epoch:3d} | Loss: {loss:.4f} | Acc: {train_acc:.2f}% | "
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
            )

    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Saved state_dict → {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
