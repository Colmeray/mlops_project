import json
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms

from src.project.data import preprocess, MyDataset


def test_preprocess(tmp_path: Path):
    raw_root = tmp_path / "raw"
    out_root = tmp_path / "preprocessed"

    # tiny raw dataset
    (raw_root / "ClassA").mkdir(parents=True)
    (raw_root / "ClassB").mkdir(parents=True)
    img = Image.new("RGB", (32, 32), color=(255, 0, 0))
    img.save(raw_root / "ClassA" / "1.jpg")
    img.save(raw_root / "ClassB" / "1.jpg")

    # run preprocess command
    preprocess(raw_root=raw_root, out_root=out_root)

    # Asserts wether files and folders are created
    assert (out_root / "meta" / "classes.json").exists()
    assert (out_root / "index" / "all.csv").exists()


    # assert wether the content is created as expected in json file
    meta_path = out_root / "meta" / "classes.json"
    meta = json.loads(meta_path.read_text())
    assert "classes" in meta
    assert set(meta["classes"]) == {"ClassA", "ClassB"}

    # check CSV file
    csv_path = out_root / "index" / "all.csv"
    lines = csv_path.read_text().strip().splitlines()
    assert lines[0].strip() == "relpath,label,class_name"
    assert len(lines) >= 3

    # Test wether
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    ds = MyDataset(processed_root=out_root, raw_root=raw_root, transform=transform)

    assert len(ds) > 0
    x, y = ds[0]

    assert isinstance(x, torch.Tensor)
    assert x.shape == (3, 64, 64)
    assert y.dtype == torch.long
    assert 0 <= int(y.item()) < ds.num_classes
