import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNet(nn.Module):

    def __init__(
        self, num_classes: int = 2, num_layers: int = 18, weights: str | None = None
    ):
        super(ResNet, self).__init__()

        if num_layers == 18:
            self.resnet = models.resnet18(weights=weights)
        elif num_layers == 34:
            self.resnet = models.resnet34(weights=weights)
        elif num_layers == 50:
            self.resnet = models.resnet50(weights=weights)
        elif num_layers == 101:
            self.resnet = models.resnet101(weights=weights)
        elif num_layers == 152:
            self.resnet = models.resnet152(weights=weights)
        else:
            raise ValueError("Invalid number of layers: {}".format(num_layers))

        # Replace the last fully connected layer of the model
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        return F.log_softmax(self.resnet(x), dim=-1)
