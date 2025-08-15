# /src/trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import config

def save_checkpoint(epoch, model, optimizer, filename):
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filename)

def plot_confusion_matrix(all_labels, all_preds, class_names, output_path):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.close()

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, checkpoint_dir, plot_path, model_name):
    best_val_acc = 0.0
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        train_loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            train_loop.set_postfix(loss=loss.item())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        corrects = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, leave=True, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for inputs, labels in val_loop:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                corrects += torch.sum(preds == labels.data)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = corrects.double() / len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
        
        # Plotting and Checkpointing
        plot_confusion_matrix(all_labels, all_preds, ['real', 'fake'], f"{plot_path}_epoch_{epoch+1}.png")
        
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{epoch+1}.pth.tar")
        save_checkpoint(epoch, model, optimizer, checkpoint_path)
        
        if epoch_val_acc > best_val_acc:
            print(f"Validation accuracy improved from {best_val_acc:.4f} to {epoch_val_acc:.4f}. Saving best model.")
            best_val_acc = epoch_val_acc
            best_model_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth.tar")
            save_checkpoint(epoch, model, optimizer, best_model_path)