import torch
import torch.nn as nn
from torchvision import models

class PneumoniaModel(nn.Module):
    def __init__(self, model_name='vgg16', num_classes=2, pretrained=True):
        super(PneumoniaModel, self).__init__()
        if model_name == 'vgg16':
            self.model = models.vgg16(pretrained=pretrained)
            for param in self.model.parameters():
                param.requires_grad = False
            n_inputs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Sequential(
                nn.Linear(n_inputs, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes),
                nn.LogSoftmax(dim=1)
            )
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            for param in self.model.parameters():
                param.requires_grad = False
            n_inputs = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(n_inputs, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes),
                nn.LogSoftmax(dim=1)
            )
        else:
            raise ValueError("Model not supported")

    def forward(self, x):
        return self.model(x)

    def to_device(self, device):        
        self.to(device)
        if device.type == 'cuda' and torch.cuda.device_count() > 1:
            self = nn.DataParallel(self)
        return self