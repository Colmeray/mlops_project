import json
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from project.model import VGG16Transfer, SimpleModel

def test_vgg16_forward_shape():
    model = VGG16Transfer(num_classes=3, weights=None)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 3)

def test_simplemodel_forward_shape():
    model = SimpleModel(num_classes=3)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 3)


def test_vgg16_freeze_features_sets_requires_grad_false():
    model = VGG16Transfer(num_classes=3, freeze_features=True, weights=None)

    assert all(not p.requires_grad for p in model.backbone.features.parameters())
    assert model.backbone.classifier[-1].weight.requires_grad is True
    assert model.backbone.classifier[-1].bias.requires_grad is True

def test_vgg16_backward_produces_gradients_on_head():
    model = VGG16Transfer(num_classes=3, freeze_features=True, weights=None)
    model.train()

    x = torch.randn(2, 3, 224, 224)
    target = torch.tensor([0, 2]) 
    criterion = torch.nn.CrossEntropyLoss()

    y = model(x)
    loss = criterion(y, target)
    loss.backward()

    assert model.backbone.classifier[-1].weight.grad is not None
    assert model.backbone.classifier[-1].bias.grad is not None
    assert all(p.grad is None for p in model.backbone.features.parameters())

