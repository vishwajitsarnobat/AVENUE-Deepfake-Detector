# /src/models.py (Corrected)

import torch
import torch.nn as nn
import config

# --- MODIFIED: Simplified the TCN block to a standard Conv -> ReLU ---
class NonCausalTCN(nn.Module):
    """
    A standard Non-Causal Temporal Convolutional Network block.
    It consists of a 1D convolution with 'same' padding and a ReLU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(NonCausalTCN, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding='same', # 'same' padding is simpler and more robust
            dilation=dilation
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1d(x)
        return self.relu(out)

# --- MODIFIED: Corrected the channel flow and FC layer ---
class PDTCN(nn.Module):
    """
    Pulse-DeepTCN (PD-TCN) model for classifying rPPG signals from stable clips.
    """
    def __init__(self):
        super(PDTCN, self).__init__()
        # Define the number of output channels for the TCN blocks
        tcn_out_channels = 3

        # Block 1: Takes 1 input channel (the signal)
        self.tcn_block1 = NonCausalTCN(1, tcn_out_channels, kernel_size=3, dilation=1)
        
        # Block 2 & 3: Take the output of the previous block as input
        self.tcn_block2 = NonCausalTCN(tcn_out_channels, tcn_out_channels, kernel_size=3, dilation=2)
        self.tcn_block3 = NonCausalTCN(tcn_out_channels, tcn_out_channels, kernel_size=3, dilation=4)
        
        # The input to the FC layer is the flattened output of the last TCN block.
        # Shape will be (channels * signal_length)
        fc_input_features = tcn_out_channels * config.PULSE_SIGNAL_LENGTH
        self.fc = nn.Linear(fc_input_features, 2) # 2 outputs: Real, Fake

    def forward(self, x):
        # Input x shape: [batch_size, 1, signal_length]
        out1 = self.tcn_block1(x)        # -> [batch_size, 3, signal_length]
        out2 = self.tcn_block2(out1)     # -> [batch_size, 3, signal_length]
        out3 = self.tcn_block3(out2)     # -> [batch_size, 3, signal_length]
        
        # Flatten the output for the fully connected layer
        out_flat = out3.view(out3.size(0), -1) # -> [batch_size, 3 * signal_length]
        
        out = self.fc(out_flat)
        return out


class Consolidator(nn.Module):
    """
    Final consolidation network that combines features from all stable clips.
    (This class requires no changes)
    """
    def __init__(self, num_clips=config.NUM_STABLE_CLIPS):
        super(Consolidator, self).__init__()
        # Each clip provides 3 features: FDA, SNR, PD-TCN Fake Probability
        input_features = num_clips * 3
        self.fc1 = nn.Linear(input_features, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2) # Final verdict: Real, Fake

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out