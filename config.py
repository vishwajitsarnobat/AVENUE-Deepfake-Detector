# /config.py (Corrected)

import torch
import os

# --- DATASET CONFIGURATION ---
PROCESSED_DATA_PATH = "data_processed"

# --- CORRECTED: Use a list of paths for each key ---
RAW_DATA_PATHS = {
    "real": [
        "data/Celeb-DF-V2/Celeb-real",
        "data/Celeb-DF-V2/YouTube-real"
    ],
    "fake": [
        "data/Celeb-DF-V2/Celeb-synthesis"
    ]
}

# --- VALIDATION SPLIT RATIO ---
VALIDATION_SPLIT_RATIO = 0.2

# --- PARALLEL PROCESSING ---
# Set to a safe number of workers. os.cpu_count() - 4 is a reasonable choice.
NUM_WORKERS = max(1, os.cpu_count() - 4)

# --- TRAINING CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
LEARNING_RATE_PDTCN = 0.001
LEARNING_RATE_CONSOLIDATOR = 0.001
EPOCHS_PDTCN = 50
EPOCHS_CONSOLIDATOR = 100
CHECKPOINT_PATH = "checkpoints"
OUTPUT_PATH = "outputs"

# --- AVENUE METHOD HYPERPARAMETERS ---
NUM_STABLE_CLIPS = 7
STABLE_CLIP_DURATION_SECS = 1
VIDEO_FPS = 30
PULSE_SIGNAL_LENGTH = STABLE_CLIP_DURATION_SECS * VIDEO_FPS

# --- rPPG CONFIGURATION ---
RPPG_FREQ_MIN = 0.7
RPPG_FREQ_MAX = 4.0

# --- PREDICTION CONFIGURATION ---
PREDICTION_OUTPUT_PATH = "outputs/predictions"