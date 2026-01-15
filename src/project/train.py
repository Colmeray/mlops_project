from project.model import Model
from project.data import MyDataset

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Subset

def train():
    dataset = MyDataset("data/processed")

    # Split dataset into training and validation sets (e.g. 90% / 10%)
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
    )
    val_loader = DataLoader(
        Subset(dataset, list(idx_val)),
        batch_size=32,
        shuffle=False,
    )

    # Select device (CUDA, Apple MPS, or CPU)
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    # Model: number of output classes is inferred from the dataset
    model = Model(num_classes=len(dataset.class_to_idx)).to(device)

    # Loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    epochs = 5
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        # Validation can be added later
        print(f"Epoch {epoch}/{epochs} finished")

    # Save model checkpoint
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_to_idx": dataset.class_to_idx,
        },
        "models/model.pt",
    )

if __name__ == "__main__":
    train()