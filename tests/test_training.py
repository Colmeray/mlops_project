import os
from omegaconf import OmegaConf
from project.train import train_impl


def test_training_runs_one_batch():
    # make sure wandb never activates in tests
    os.environ["WANDB_MODE"] = "disabled"

    # minimal config overrides for a super tiny run
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "val_frac": 0.1,
            "epochs": 1,
            "batch_size": 2,
            "lr": 1e-4,
            "model": {"name": "simple", "freeze_features": True},
            "wandb": {"enable": False},
            "smoke_test": True,
        }
    )

    metrics = train_impl(cfg, max_batches=1)

    assert metrics["train_loss"] == metrics["train_loss"]  # not NaN
    assert metrics["train_loss"] > 0
    assert 0.0 <= metrics["train_acc"] <= 1.0
