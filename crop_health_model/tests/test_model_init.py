import pytest

from crop_health_model.models.model import ResNet


def test_model_init():
    model = ResNet(num_classes=2, num_layers=18)
    assert model.get_hyperparameters() == {"num_classes": 2, "num_layers": 18}

    model = ResNet(num_classes=5, num_layers=50)
    assert model.get_hyperparameters() == {"num_classes": 5, "num_layers": 50}

    model = ResNet(num_classes=10, num_layers=101)
    assert model.get_hyperparameters() == {"num_classes": 10, "num_layers": 101}

    model = ResNet(num_classes=20, num_layers=152)
    assert model.get_hyperparameters() == {"num_classes": 20, "num_layers": 152}


def test_model_pretrained():
    model = ResNet(num_classes=2, num_layers=18, weights="DEFAULT")
    assert model.get_hyperparameters() == {"num_classes": 2, "num_layers": 18}


def test_model_invalid():
    with pytest.raises(ValueError):
        model = ResNet(num_classes=2, num_layers=20)
    with pytest.raises(ValueError):
        model = ResNet(num_classes=5, num_layers=30)
    with pytest.raises(ValueError):
        model = ResNet(num_classes=10, num_layers=40)
