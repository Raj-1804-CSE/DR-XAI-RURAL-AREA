import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model.architecture import get_model

class DRTrainer:
    """
    Training harness for the DR Classifier.
    Handles the training loop, validation, and model saving.
    """
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return running_loss / total, 100 * correct / total

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return running_loss / total, 100 * correct / total

    def fit(self, epochs, save_path='models/best_model.pth'):
        best_acc = 0.0
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            print(f"Epoch {epoch+1}/{epochs} -> Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved new best model to {save_path}")

if __name__ == "__main__":
    print("DRTrainer initialized. Ready for training.")
