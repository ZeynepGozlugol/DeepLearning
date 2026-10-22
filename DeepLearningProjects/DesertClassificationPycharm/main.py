import torch
from torch import nn
from torchvision import transforms
import torchvision
from pathlib import Path

import model_creation
import setup_data
import training_testing_engine
import utils

HIDDEN_UNITS = 32
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 10
MODEL_SAVE_PATH = 'models/desert_classifier.pth'

train_dir = "data/desert101/train"
test_dir = "data/desert101/test"

data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


def run():
    # 1. DataLoader'ları hazırla (num_workers=0 ile güvenli mod)
    train_dataloader, test_dataloader, class_names = setup_data.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE,
        num_workers=0
    )

    # 2. Modeli kur
    model = model_creation.DesertClassifier(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    )

    # 3. Kayıp fonksiyonu ve Optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    # 4. Modeli eğit
    print("Model eğitiliyor...")
    training_testing_engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=NUM_EPOCHS
    )

    # 5. Modeli kaydet
    utils.save_model(
        model=model,
        target_dir="models",
        model_name="desert_classifier.pth"
    )

    # 6. Kaydedilen modeli yükle ve doğrula
    loaded_model = model_creation.DesertClassifier(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    )
    loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    # 7. Örnek görsel tahmini
    data_path = Path("data/")
    online_image_path = data_path / "baklava-online.jpg"

    single_image = torchvision.io.read_image(str(online_image_path)).type(torch.float32)
    single_image = single_image / 255.0

    single_image_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
    ])
    single_image = single_image_transform(single_image)
    single_image = single_image.unsqueeze(dim=0)

    loaded_model.eval()
    with torch.inference_mode():
        logits = loaded_model(single_image)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()

    print("Predicted class:", class_names[pred_idx])


if __name__ == '__main__':
    run()