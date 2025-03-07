import os
import torch
from data_preparation import get_data_loaders
from model import PneumoniaModel
from train import train_model

# Configuration
data_dir = 'data/chest_xray'
model_name = 'vgg16'
num_classes = 2
batch_size = 128
n_epochs = 10
save_file_name = 'models/vgg16-chest-4.pth'

# Load data
dataloaders, class_to_idx = get_data_loaders(data_dir, batch_size)

model = PneumoniaModel(model_name, num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to_device(device)
model.device = device

criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters())

model, history = train_model(
    model, criterion, optimizer, dataloaders['train'], dataloaders['val'],
    save_file_name, n_epochs=n_epochs
)

torch.save({
    'class_to_idx': class_to_idx,
    'idx_to_class': {v: k for k, v in class_to_idx.items()},
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, save_file_name)

print(history)
