import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

def train_model(model, criterion, optimizer, train_loader, val_loader, save_file_name, max_epochs_stop=3, n_epochs=20, print_every=2):
    """Trains the model and saves the best weights based on validation loss."""
    epochs_no_improve = 0
    valid_loss_min = float('inf')
    history = []

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0

        for data, target in train_loader:
            data, target = data.to(model.device), target.to(model.device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            _, pred = torch.max(output, dim=1)
            correct = pred.eq(target).sum().item()
            train_acc += correct

        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)

        model.eval()
        valid_loss = 0.0
        valid_acc = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(model.device), target.to(model.device)
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item() * data.size(0)
                _, pred = torch.max(output, dim=1)
                correct = pred.eq(target).sum().item()
                valid_acc += correct

        valid_loss /= len(val_loader.dataset)
        valid_acc /= len(val_loader.dataset)

        history.append([train_loss, valid_loss, train_acc, valid_acc])

        if (epoch + 1) % print_every == 0:
            print(f'Epoch: {epoch+1}/{n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {valid_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {valid_acc:.4f}')

        if valid_loss < valid_loss_min:
            torch.save(model.state_dict(), save_file_name)
            valid_loss_min = valid_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= max_epochs_stop:
                print(f'Early stopping after {epoch+1} epochs.')
                break

    return model, pd.DataFrame(history, columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
