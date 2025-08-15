# /src/data_loader.py

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import config

class PDTCN_Dataset(Dataset):
    """Dataset for training the Pulse-DeepTCN model."""
    def __init__(self, data_dir, label_map):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        self.label_map = label_map

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = np.load(file_path)
        
        # We get all signals and use them as individual samples
        signals = data['signals'] # Shape: (num_clips, signal_length)
        label_str = data['label'].item()
        label = self.label_map[label_str]
        
        # Prepare for batching: Convert to tensors
        signals_tensor = torch.tensor(signals, dtype=torch.float32).unsqueeze(1) # Add channel dim
        labels_tensor = torch.tensor([label] * len(signals), dtype=torch.long)
        
        return signals_tensor, labels_tensor

class ConsolidatorDataset(Dataset):
    """Dataset for training the final Consolidator model."""
    def __init__(self, data_dir, pdtcn_model, device, label_map):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        self.pdtcn_model = pdtcn_model
        self.pdtcn_model.eval() # Ensure PD-TCN is in evaluation mode
        self.device = device
        self.label_map = label_map

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = np.load(file_path)
        
        fda_scores = data['fda_scores']
        snr_scores = data['snr_scores']
        signals = data['signals']
        label_str = data['label'].item()
        label = self.label_map[label_str]

        # Get PD-TCN predictions for each clip's signal
        with torch.no_grad():
            signals_tensor = torch.tensor(signals, dtype=torch.float32).unsqueeze(1).to(self.device)
            pdtcn_outputs = self.pdtcn_model(signals_tensor)
            pdtcn_probs = torch.softmax(pdtcn_outputs, dim=1)
            # We need the probability of the 'fake' class (index 1)
            pdtcn_fake_probs = pdtcn_probs[:, 1].cpu().numpy()

        # Consolidate features: [fda_1, snr_1, pdtcn_1, fda_2, snr_2, pdtcn_2, ...]
        features = np.stack([fda_scores, snr_scores, pdtcn_fake_probs], axis=1).flatten()
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def collate_pdtcn(batch):
    """Custom collate function to handle variable number of clips per video."""
    all_signals = []
    all_labels = []
    for signals, labels in batch:
        all_signals.append(signals)
        all_labels.append(labels)
    return torch.cat(all_signals, dim=0), torch.cat(all_labels, dim=0)