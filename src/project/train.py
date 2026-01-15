from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split

from src.project.data import MyDataset   # adjust import
from src.project.model import Model      # adjust import

from torchvision import transforms


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train():
    transform = transforms.Compose([
    transforms.Resize((224, 224)),   # pick size your model expects
    transforms.ToTensor(),
    ])      
    processed_root = Path("data/preprocessed")
    raw_root = Path("data/raw/house_plant_species")

    dataset = MyDataset(processed_root=processed_root, raw_root=raw_root,transform=transform)  # + transform if you use it




    # split
    n = len(dataset)
    n_val = int(0.1 * n)
    n_train = n - n_val
    idx_train, idx_val = random_split(
        range(n),
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        Subset(dataset, list(idx_train)),
        batch_size=32,
        shuffle=True,
        num_workers=0,   # mac: start with 0, later try 2-4
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(dataset, list(idx_val)),
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    device = get_device()

    model = Model(num_classes=dataset.num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print("training starting")
    epochs = 5
    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_loss /= total
        train_acc = correct / total

        # ---- val ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                loss = loss_fn(logits, y)

                val_loss += loss.item() * x.size(0)
                pred = logits.argmax(dim=1)
                val_correct += (pred == y).sum().item()
                val_total += y.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )


if __name__ == "__main__":
    train()
