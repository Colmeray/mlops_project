from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from project.data import MyDataset   # adjust import
from project.model import SimpleModel      # adjust import
from torchvision import transforms
import hydra
from project.model import VGG16Transfer
import wandb
from omegaconf import OmegaConf


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_impl(cfg, max_batches: int | None = None):
    if cfg.wandb.enable:
        wandb.init(
            project="mlops_project",
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    transform = transforms.Compose([
    transforms.Resize((224, 224)),   # pick size your model expects
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                     std=(0.229, 0.224, 0.225))
    ])      
    processed_root = Path("data/preprocessed")
    raw_root = Path("data/raw/house_plant_species")

    dataset = MyDataset(processed_root=processed_root, raw_root=raw_root,transform=transform)  
    num_classes = dataset.num_classes 

    if cfg.get("smoke_test", False):
        dataset = torch.utils.data.Subset(dataset, range(min(16, len(dataset))))

    # split
    n = len(dataset)
    n_val = int(0.1 * n)
    n_train = n - n_val
    idx_train, idx_val = random_split(
        range(n),
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    train_loader = DataLoader(
        Subset(dataset, list(idx_train)),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,   # mac: start with 0, later try 2-4
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(dataset, list(idx_val)),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    device = get_device()

    if cfg.model.name == "simple":
        model = SimpleModel(num_classes=num_classes).to(device)
    elif cfg.model.name == "vgg16":
        model = VGG16Transfer(num_classes=num_classes,freeze_features=cfg.model.freeze_features).to(device)

    loss_fn = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=cfg.lr)

    print("training starting")
    epochs = cfg.epochs
    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for i, (x, y) in enumerate(train_loader):
            if max_batches is not None and i > max_batches:
                break
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
            if i % 50 == 0:
                print(f"Epoch {epoch}/{epochs} | batch {i}/{len(train_loader)}")
            

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
        if cfg.wandb.enable:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": cfg.lr,
                })

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )
    if cfg.wandb.enable:
        wandb.finish()
    return {"train_loss": train_loss, "train_acc": train_acc}


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg):
    return train_impl(cfg)


if __name__ == "__main__":
    train()
