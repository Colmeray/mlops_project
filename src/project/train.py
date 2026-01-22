from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from project.data import MyDataset  # adjust import
from project.model import SimpleModel  # adjust import
from torchvision import transforms
import hydra
from project.model import VGG16Transfer
import wandb
from omegaconf import OmegaConf
from loguru import logger
import time

# ---- NEW IMPORTS TO PROFILER ---- #
from torch.profiler import profile, ProfilerActivity, record_function


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("cuda virker!!" , flush = True)
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Sindssyg mac M-chip aktiveret!" , flush= True)
        return torch.device("mps")
    logger.critical("Warning CPU is being used 🤡🤡🤡🤡🤡🤡")
    return torch.device("cpu")


def train_impl(cfg, max_batches: int | None = None):
    # disable wandb early in smoke tests / CI tests
    if cfg.get("smoke_test", False):
        cfg.wandb.enable = False

    if cfg.wandb.enable:
        wandb.init(
            project="mlops_project",
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # pick size your model expects
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    # Allow overriding paths (useful for tests/CI); keep defaults for normal runs.
    processed_root = Path(cfg.get("processed_root", "data/preprocessed"))

    raw_root_cfg = Path(cfg.get("raw_root", "data/raw"))

    # Keep your KaggleHub folder fallback, but only relative to the chosen raw_root
    candidate = raw_root_cfg / "house_plant_species"
    raw_root = candidate if candidate.exists() else raw_root_cfg

    print("processed_root =", processed_root, flush=True)
    print("raw_root       =", raw_root, flush=True)

    dataset = MyDataset(processed_root=processed_root, raw_root=raw_root, transform=transform)
    num_classes = dataset.num_classes

    if cfg.get("smoke_test", False):
        dataset = torch.utils.data.Subset(dataset, range(min(16, len(dataset))))

    # split
    n = len(dataset)
    val_frac = float(cfg.get("val_frac", 0.1))
    n_val = max(1, int(val_frac * n)) if n >= 2 else 0
    n_train = n - n_val
    if n_val == 0:
        n_train = n
    idx_train, idx_val = random_split(
        range(n),
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    train_loader = DataLoader(
        Subset(dataset, list(idx_train)),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
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
        model = VGG16Transfer(num_classes=num_classes, freeze_features=cfg.model.freeze_features).to(device)

    loss_fn = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=cfg.lr)

    print("training starting", flush=True)
    epochs = cfg.epochs

    # ================== TORCH PROFILER ==================
    with profile(
        activities=[ProfilerActivity.CPU],  # <- kun CPU på M2
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs"),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
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
            tid_start = time.perf_counter()
            print(f"epoch {epoch} is running:\n", flush=True)
            model.train()

            train_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (x, y) in enumerate(train_loader):
                if max_batches is not None and batch_idx > max_batches:
                    break
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
                if batch_idx % 10 == 0:
                    logger.info(f"epoch: {epoch}  batch id completed:{batch_idx}/{len(train_loader)} Loss: {loss.item()}")
                    logger.info(f"total time in epoch: {time.perf_counter() - tid_start}")
                # print(f"{epoch} : {loss.item()}", flush=True)

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

            if cfg.get("wandb", {}).get("enable", False) and not cfg.get("smoke_test", False):
            if cfg.wandb.enable:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": cfg.lr,
                })
            
            logger.info(f"Epoch {epoch:02d} | ")
            logger.info(f"train loss {train_loss:.4f} acc {train_acc:.3f} | ")
            logger.info(f"val loss {val_loss:.4f} acc {val_acc:.3f}")

            # i know i should have defined a varible: but it's too late now :D
            if (time.perf_counter() - tid_start)*(epochs - epoch +1) > 60*60*24:
                logger.critical(f"estimated time to finish: {(time.perf_counter() - tid_start)*(epochs - epoch +1)}")
            elif (time.perf_counter() - tid_start)*(epochs - epoch +1) > 60*60*4:
                logger.warning(f"estimated time to finish: {(time.perf_counter() - tid_start)*(epochs - epoch +1)}")
            else:
                logger.info(f"estimated time to finish: {(time.perf_counter() - tid_start)*(epochs - epoch +1)}")
            
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "lr": cfg.lr,
                    }
                )

            
            # print(
            #     f"Epoch {epoch:02d} | "
            #     f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            #     f"val loss {val_loss:.4f} acc {val_acc:.3f}"
            # )

    if cfg.wandb.enable:
        wandb.finish()
    logger.complete()
    return {"train_loss": train_loss, "train_acc": train_acc}


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg):
    return train_impl(cfg)


if __name__ == "__main__":
    train()
