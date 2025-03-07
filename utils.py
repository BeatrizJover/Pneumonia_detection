import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models

def load_checkpoint(path, model_name, num_classes):
    """Loads a model from a checkpoint."""
    checkpoint = torch.load(path)
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=False)
        model.classifier = checkpoint['classifier']
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=False)
        model.fc = checkpoint['fc']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = checkpoint['idx_to_class']
    return model

def imshow_tensor(image, ax=None, title=None):
    """Displays a tensor image."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    if title:
        ax.set_title(title)
    ax.axis('off')
    return ax
