# /main.py (Corrected Version)

import argparse
import os

# This must be set before any mediapipe or tensorflow imports.
# 2 = Filters out INFO and WARNING logs.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import shutil
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import cv2
import random
import concurrent.futures

import config
from src.analysis import get_facial_landmarks, calculate_facial_deformation, find_stable_clips, extract_rppg_signal, calculate_snr, calculate_fda
from src.data_loader import PDTCN_Dataset, ConsolidatorDataset, collate_pdtcn
from src.models import PDTCN, Consolidator
from src.trainer import train_model
from src.predictor import predict_video

def process_single_video(job_args):
    """
    A self-contained function to process one video file.
    This will be executed by each worker in the process pool.
    """
    video_path, label, output_file_path = job_args
    
    if os.path.exists(output_file_path):
        return f"Skipped: {os.path.basename(video_path)}"
        
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return f"Error opening {os.path.basename(video_path)}"

        all_frames = []
        landmarks_sequence = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            all_frames.append(frame)
            landmarks_sequence.append(get_facial_landmarks(frame))
        cap.release()

        if len(landmarks_sequence) < config.PULSE_SIGNAL_LENGTH:
             return f"Not enough frames in {os.path.basename(video_path)}"

        deformations = calculate_facial_deformation(landmarks_sequence)
        clip_len_frames = config.STABLE_CLIP_DURATION_SECS * config.VIDEO_FPS
        stable_clips_indices = find_stable_clips(deformations, config.NUM_STABLE_CLIPS, clip_len_frames)

        if len(stable_clips_indices) < config.NUM_STABLE_CLIPS:
            return f"Not enough stable clips in {os.path.basename(video_path)}"

        signals, fda_scores, snr_scores = [], [], []
        for start, end in stable_clips_indices:
            clip_frames = np.array(all_frames[start:end])
            signal = extract_rppg_signal(clip_frames)
            signals.append(signal)
            fda_scores.append(calculate_fda(deformations[start:end-1]))
            snr_scores.append(calculate_snr(signal, config.VIDEO_FPS))

        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        np.savez(output_file_path, signals=signals, fda_scores=fda_scores, snr_scores=snr_scores, label=label)
        return f"Processed: {os.path.basename(video_path)}"

    except Exception as e:
        return f"Failed {os.path.basename(video_path)}: {e}"


def preprocess_data(processed_data_path):
    print("Starting automated preprocessing. Will skip already processed files.")
    jobs = []
    
    # Iterate through labels ('real', 'fake') and their corresponding lists of directories
    for label, source_dir_list in config.RAW_DATA_PATHS.items():
        all_videos = []
        
        # Collect all video paths from all directories for the current label
        for source_dir in source_dir_list:
            if not os.path.isdir(source_dir):
                print(f"WARNING: Raw data directory not found, skipping: {source_dir}")
                continue
            
            print(f"Searching for '{label}' videos in: {source_dir}")
            all_videos.extend([os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.endswith(('.mp4', '.avi', '.mov'))])

        if not all_videos:
            print(f"WARNING: No videos found for label '{label}'.")
            continue

        random.shuffle(all_videos)
        split_index = int(len(all_videos) * config.VALIDATION_SPLIT_RATIO)
        val_set = all_videos[:split_index]
        train_set = all_videos[split_index:]
        
        print(f"Found {len(all_videos)} total '{label}' videos. Splitting into {len(train_set)} train and {len(val_set)} validation samples.")
        
        for video_path in train_set:
            output_dir = os.path.join(processed_data_path, 'train', label)
            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}.npz")
            jobs.append((video_path, label, output_file))
            
        for video_path in val_set:
            output_dir = os.path.join(processed_data_path, 'val', label)
            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}.npz")
            jobs.append((video_path, label, output_file))

    if not jobs:
        print("ERROR: No jobs were created. Please check your RAW_DATA_PATHS in config.py.")
        return

    print(f"\nDistributing {len(jobs)} total jobs across {config.NUM_WORKERS} worker(s)...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=config.NUM_WORKERS) as executor:
        results = list(tqdm(executor.map(process_single_video, jobs), total=len(jobs)))

    print("\nPreprocessing finished.")
    success_count = sum(1 for r in results if r.startswith("Processed") or r.startswith("Skipped"))
    fail_count = len(results) - success_count
    print(f"Summary: {success_count} videos processed/skipped successfully, {fail_count} failed.")
    
def train():
    device = torch.device(config.DEVICE)
    label_map = {'real': 0, 'fake': 1}

    print("\n--- Training PD-TCN Model ---")
    pdtcn_model = PDTCN().to(device)
    pdtcn_optimizer = optim.Adam(pdtcn_model.parameters(), lr=config.LEARNING_RATE_PDTCN)
    pdtcn_criterion = nn.CrossEntropyLoss()

    train_pdtcn_dataset = PDTCN_Dataset(os.path.join(config.PROCESSED_DATA_PATH, 'train', 'real'), label_map) + \
                        PDTCN_Dataset(os.path.join(config.PROCESSED_DATA_PATH, 'train', 'fake'), label_map)
    val_pdtcn_dataset = PDTCN_Dataset(os.path.join(config.PROCESSED_DATA_PATH, 'val', 'real'), label_map) + \
                      PDTCN_Dataset(os.path.join(config.PROCESSED_DATA_PATH, 'val', 'fake'), label_map)

    train_pdtcn_loader = DataLoader(train_pdtcn_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_pdtcn)
    val_pdtcn_loader = DataLoader(val_pdtcn_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_pdtcn)

    train_model(pdtcn_model, train_pdtcn_loader, val_pdtcn_loader, pdtcn_criterion, pdtcn_optimizer, 
                config.EPOCHS_PDTCN, device, 
                os.path.join(config.CHECKPOINT_PATH, 'pdtcn'), 
                os.path.join(config.OUTPUT_PATH, 'training_plots', 'pdtcn_cm'), 'pdtcn')
    
    print("\n--- Training Consolidator Model ---")
    best_pdtcn_path = os.path.join(config.CHECKPOINT_PATH, 'pdtcn', 'pdtcn_best.pth.tar')
    checkpoint = torch.load(best_pdtcn_path, map_location=device)
    pdtcn_model.load_state_dict(checkpoint['state_dict'])

    consolidator_model = Consolidator().to(device)
    consolidator_optimizer = optim.Adam(consolidator_model.parameters(), lr=config.LEARNING_RATE_CONSOLIDATOR)
    consolidator_criterion = nn.CrossEntropyLoss() 

    train_consolidator_dataset = ConsolidatorDataset(os.path.join(config.PROCESSED_DATA_PATH, 'train', 'real'), pdtcn_model, device, label_map) + \
                                 ConsolidatorDataset(os.path.join(config.PROCESSED_DATA_PATH, 'train', 'fake'), pdtcn_model, device, label_map)
    val_consolidator_dataset = ConsolidatorDataset(os.path.join(config.PROCESSED_DATA_PATH, 'val', 'real'), pdtcn_model, device, label_map) + \
                               ConsolidatorDataset(os.path.join(config.PROCESSED_DATA_PATH, 'val', 'fake'), pdtcn_model, device, label_map)
    
    train_consolidator_loader = DataLoader(train_consolidator_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_consolidator_loader = DataLoader(val_consolidator_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    train_model(consolidator_model, train_consolidator_loader, val_consolidator_loader, consolidator_criterion, consolidator_optimizer,
                config.EPOCHS_CONSOLIDATOR, device,
                os.path.join(config.CHECKPOINT_PATH, 'consolidator'),
                os.path.join(config.OUTPUT_PATH, 'training_plots', 'consolidator_cm'), 'consolidator')

    print("Training complete.")


def predict(video_path):
    print(f"Running prediction on: {video_path}")
    device = torch.device(config.DEVICE)
    
    pdtcn_model = PDTCN().to(device)
    pdtcn_checkpoint = torch.load(os.path.join(config.CHECKPOINT_PATH, 'pdtcn', 'pdtcn_best.pth.tar'), map_location=device)
    pdtcn_model.load_state_dict(pdtcn_checkpoint['state_dict'])

    consolidator_model = Consolidator().to(device)
    consolidator_checkpoint = torch.load(os.path.join(config.CHECKPOINT_PATH, 'consolidator', 'consolidator_best.pth.tar'), map_location=device)
    consolidator_model.load_state_dict(consolidator_checkpoint['state_dict'])
    
    predict_video(video_path, pdtcn_model, consolidator_model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AVENUE: A Novel Deepfake Detection Method")
    subparsers = parser.add_subparsers(dest='command', required=True)

    parser_preprocess = subparsers.add_parser('preprocess', help='Preprocess raw videos and split into train/val sets.')
    parser_train = subparsers.add_parser('train', help='Train the PD-TCN and Consolidator models.')
    parser_predict = subparsers.add_parser('predict', help='Predict if a video is a deepfake.')
    parser_predict.add_argument('video_path', type=str, help='Path to the video file to analyze.')

    args = parser.parse_args()

    if args.command == 'preprocess':
        preprocess_data(config.PROCESSED_DATA_PATH)
    elif args.command == 'train':
        train()
    elif args.command == 'predict':
        if not os.path.exists(args.video_path):
            print(f"Error: Video file not found at {args.video_path}")
        else:
            predict(args.video_path)