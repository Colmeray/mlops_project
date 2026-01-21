from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split

from project.data import MyDataset   # adjust import
from project.model import Model      # adjust import

from torchvision import transforms

# ---- NEW IMPORTS TO PROFILER ---- #
from torch.profiler import profile, ProfilerActivity, record_function


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("cuda virker!!")
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Sindssyg mac M-chip aktiveret!")
        return torch.device("mps")
    return torch.device("cpu")


def train():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    processed_root = Path("data/preprocessed")
    raw_root = Path("data/raw/house_plant_species")

    dataset = MyDataset(
        processed_root=processed_root,
        raw_root=raw_root,
        transform=transform,
    )

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
        num_workers=0,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("training starting")
    epochs = 1

    # ================== TORCH PROFILER ==================
    with profile(
        activities=[ProfilerActivity.CPU],  # <- kun CPU på M2
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:

    # with profile(
    #     activities=[
    #         ProfilerActivity.CPU,
    #         ProfilerActivity.CUDA,   # vigtigt hvis du bruger GPU
    #     ],
    #     schedule=torch.profiler.schedule(
    #         wait=1,        # ignorer første batch
    #         warmup=1,      # warmup
    #         active=3,      # profiler 3 batches
    #         repeat=2       # gentag 2 gange
    #     ),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs"),
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True,
    # ) as prof:
        # ================================================

        for epoch in range(1, epochs + 1):
            print(f"epoch {epoch} is running:\n")
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (x, y) in enumerate(train_loader):
                x = x.to(device)
                y = y.to(device)

                with record_function("forward_pass"):
                    logits = model(x)
                    loss = loss_fn(logits, y)

                optimizer.zero_grad()

                with record_function("backward"):
                    loss.backward()

                with record_function("optimizer_step"):
                    optimizer.step()

                train_loss += loss.item() * x.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

                print(f"{epoch} : {loss.item()} , ")

                # ---- MEGET VIGTIGT: STEP PROFILEREN HVER BATCH ----
                prof.step()

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
