from pathlib import Path
import typer
from torch.utils.data import Dataset
import kagglehub  
import json
from PIL import Image
import torch
from typing import Optional, Callable, Any
from pathlib import Path
from typing import Any, Callable, Optional
import csv



app = typer.Typer()


@app.command("ensure-dataset")
def ensure_dataset(data_dir: Path = Path("data/raw")) -> None:
    """
    Download dataset via kagglehub and expose it at data/raw via symlink
    """
    data_dir = data_dir.resolve()
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    cache_path = Path(kagglehub.dataset_download("kacpergregorowicz/house-plant-species")).resolve()

    # check if data is already downloaded
    if data_dir.exists() or data_dir.is_symlink():
        typer.echo(f"{data_dir} already exists -> {data_dir}")
        typer.echo("If you want to recreate it: rm -rf data/raw and rerun")
        return

    #create symlink from cache to data folder
    data_dir.symlink_to(cache_path, target_is_directory=True)
    typer.echo(f"Linked {data_dir} -> {cache_path}")




@app.command("preprocess")
def preprocess(raw_root: Path, out_root: Path) -> None:
    """
    Creates:
      out_root/meta/classes.json     meta data (class name to label mapping)
      out_root/index/all.csv         one row per image: relative_path,label,class_name
    """

    # Define allowed file types 
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

    # getting the paths and make directories
    raw_root = raw_root.resolve()
    out_root = out_root.resolve()
    meta_dir = out_root / "meta"
    index_dir = out_root / "index"
    meta_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)


    # 1) classes = folder names
    classes = sorted([d.name for d in raw_root.iterdir() if d.is_dir()])
    if not classes:
        raise ValueError(f"No class folders found in {raw_root}")

    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    # 2) write mapping
    (meta_dir / "classes.json").write_text(
        json.dumps(
            {
                "classes": classes,
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class,
                "raw_root": str(raw_root),
                "img_exts": sorted(IMG_EXTS),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # 3) write one big index CSV
    lines = ["relpath,label,class_name"]
    n = 0
    for class_name in classes:
        class_dir = raw_root / class_name
        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                rel = p.relative_to(raw_root).as_posix()
                label = class_to_idx[class_name]
                lines.append(f"{rel},{label},{class_name}")
                n += 1

    if n == 0:
        raise ValueError(f"No images found under {raw_root} with extensions {sorted(IMG_EXTS)}")

    (index_dir / "all.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved mapping: {meta_dir / 'classes.json'}")
    print(f"Saved index:   {index_dir / 'all.csv'}  ({n} images)")



IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


class MyDataset(Dataset):
    def __init__(
        self,
        processed_root: Path,
        raw_root: Path,
        transform: Optional[Callable[[Any], Any]] = None,
    ):
        self.processed_root = Path(processed_root)
        self.raw_root = Path(raw_root)
        self.transform = transform

        index_path = self.processed_root / "index" / "all.csv"
        meta_path = self.processed_root / "meta" / "classes.json"

        if not index_path.exists():
            raise FileNotFoundError(f"Missing index file: {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing meta file: {meta_path}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.classes = meta["classes"]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # samples = [(absolute_image_path, label_int), ...]
        self.samples: list[tuple[Path, int]] = []
        with index_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel = row["relpath"]
                label = int(row["label"])

                p = (self.raw_root / rel).resolve()
                if p.suffix.lower() not in IMG_EXTS:
                    continue
                if not p.exists():
                    # If you want strictness, raise instead:
                    # raise FileNotFoundError(f"Image listed in CSV not found: {p}")
                    continue

                self.samples.append((p, label))

        if not self.samples:
            raise ValueError(
                f"No valid images found. Checked CSV: {index_path} with raw_root={self.raw_root}"
            )

        # handy for train.py
        self.num_classes = len(self.classes)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        p, y = self.samples[i]
        x = Image.open(p).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, torch.tensor(y, dtype=torch.long)


if __name__ == "__main__":
    app()


