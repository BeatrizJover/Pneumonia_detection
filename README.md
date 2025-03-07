# Pneumonia Detection Using Chest X-ray Images

This repository implements a deep learning pipeline for detecting pneumonia from chest x-ray images. The project includes data preparation, model training with early stopping, checkpoint saving, and utility functions for visualization and model loading.

## Overview

The project is designed to:

- Load and preprocess chest x-ray images.
- Train a convolutional neural network (CNN) using VGG16 and ResNet50.
- Save the best model based on validation loss.
- Visualize image tensors and track training history.

## Requirements

```bash
pip install -r requirements.txt
```

## Data Preparation

File: ```data_preparation.py```

Function: ```get_data_loaders(data_dir, batch_size)```

_Creates DataLoaders for training, validation, and testing sets, applying appropriate data augmentation and normalization transforms._

## Model Definition & Checkpoint Loading

File: ```model.py```

Class: ```PneumoniaModel```

_Defines the model architecture used for pneumonia detection._

File: ```utils.py (or equivalent)```

Function: ```load_checkpoint(path, model_name, num_classes)```

_Loads a model checkpoint for VGG16 or ResNet50 architectures, restoring the model state and class mappings._

## Training

File: ```train.py```

Function: ```train_model(model, criterion, optimizer, train_loader, val_loader, save_file_name, max_epochs_stop, n_epochs, print_every)```

_Trains the model while monitoring validation loss. It implements early stopping and saves the best performing model weights._

## Visualization

File: ```utils.py (or equivalent)```

Function: ```imshow_tensor(image, ax, title)```

_Visualizes an image tensor after applying inverse normalization._

