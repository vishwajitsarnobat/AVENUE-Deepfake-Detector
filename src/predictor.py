# /src/predictor.py

import cv2
import torch
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from scipy.fft import fft

from src.analysis import get_facial_landmarks, calculate_facial_deformation, find_stable_clips, extract_rppg_signal, calculate_snr, calculate_fda
import config

def create_analysis_visualization(video_path, deformations, stable_clips_indices, signals, spectra, freqs, verdict, confidence):
    """Generates and saves the comprehensive analysis image."""
    fig, axes = plt.subplots(config.NUM_STABLE_CLIPS + 1, 2, figsize=(15, 4 * (config.NUM_STABLE_CLIPS + 1)),
                             gridspec_kw={'width_ratios': [3, 1]})
    fig.suptitle(f"AVENUE Deepfake Analysis: {os.path.basename(video_path)}", fontsize=20)

    # --- 1. Deformation Timeline ---
    ax_timeline = plt.subplot2grid((config.NUM_STABLE_CLIPS + 1, 2), (0, 0), colspan=2)
    ax_timeline.plot(deformations, color='dodgerblue', label='Facial Deformation')
    ax_timeline.set_title("Facial Deformation Timeline & Selected Stable Clips", fontsize=14)
    ax_timeline.set_xlabel("Frame Number")
    ax_timeline.set_ylabel("Deformation Magnitude")
    for start, end in stable_clips_indices:
        ax_timeline.axvspan(start, end, color='limegreen', alpha=0.4, label='_nolegend_')
    ax_timeline.legend(['Facial Deformation', 'Selected Stable Clips'])
    ax_timeline.grid(True, linestyle=':')

    # --- 2. Per-Clip Analysis ---
    for i in range(config.NUM_STABLE_CLIPS):
        signal = signals[i]
        spectrum = spectra[i]
        freq = freqs[i]

        # rPPG Signal Plot
        axes[i+1, 0].plot(signal, color='crimson')
        axes[i+1, 0].set_title(f"Clip {i+1}: rPPG Signal")
        axes[i+1, 0].set_xlabel("Frame")
        axes[i+1, 0].set_ylabel("Amplitude")
        axes[i+1, 0].grid(True, linestyle=':')

        # Pulse Spectrum Plot
        axes[i+1, 1].plot(freq, spectrum, color='purple')
        axes[i+1, 1].set_title(f"Clip {i+1}: Pulse Spectrum")
        axes[i+1, 1].set_xlabel("Frequency (Hz)")
        axes[i+1, 1].set_ylabel("Magnitude")
        axes[i+1, 1].grid(True, linestyle=':')

    # --- 3. Final Verdict ---
    verdict_color = 'red' if verdict == 'FAKE' else 'green'
    fig.text(0.5, 0.03, f"Final Verdict: {verdict} (Confidence: {confidence:.2%})",
             ha='center', va='center', fontsize=24, color='white',
             bbox=dict(boxstyle='round,pad=0.5', fc=verdict_color, ec='black', lw=2))

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    # Save the figure
    output_filename = os.path.join(config.PREDICTION_OUTPUT_PATH, f"analysis_{os.path.basename(video_path)}.png")
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    plt.savefig(output_filename)
    print(f"Analysis visualization saved to: {output_filename}")
    plt.close()

def predict_video(video_path, pdtcn_model, consolidator_model, device):
    """Runs the full AVENUE pipeline on a single video."""
    pdtcn_model.eval()
    consolidator_model.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # --- Stage 1: Face Analysis & Stable Clip Selection ---
    all_frames = []
    landmarks_sequence = []
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Analyzing video frames")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
        landmarks = get_facial_landmarks(frame)
        landmarks_sequence.append(landmarks)
        pbar.update(1)
    pbar.close()
    cap.release()
    
    if not landmarks_sequence:
        print("Error: No faces detected in the video.")
        return

    deformations = calculate_facial_deformation(landmarks_sequence)
    clip_len_frames = config.STABLE_CLIP_DURATION_SECS * config.VIDEO_FPS
    stable_clips_indices = find_stable_clips(deformations, config.NUM_STABLE_CLIPS, clip_len_frames)
    
    if len(stable_clips_indices) < config.NUM_STABLE_CLIPS:
        print(f"Warning: Only found {len(stable_clips_indices)} stable clips. Results may be inaccurate.")
        return

    # --- Stage 2: Feature Extraction for each clip ---
    all_fda_scores, all_snr_scores, all_signals, all_spectra, all_freqs = [], [], [], [], []
    pdtcn_fake_probs = []

    for start, end in tqdm(stable_clips_indices, desc="Extracting features from stable clips"):
        clip_frames = all_frames[start:end]
        deformations_in_clip = deformations[start:end-1] # Deformation is between frames

        # Extract rPPG and calculate features
        signal = extract_rppg_signal(np.array(clip_frames))
        all_signals.append(signal)
        
        snr = calculate_snr(signal, config.VIDEO_FPS)
        all_snr_scores.append(snr)

        fda = calculate_fda(deformations_in_clip)
        all_fda_scores.append(fda)

        # Get PD-TCN prediction
        with torch.no_grad():
            signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            pdtcn_output = pdtcn_model(signal_tensor)
            pdtcn_prob = torch.softmax(pdtcn_output, dim=1)
            pdtcn_fake_probs.append(pdtcn_prob[0, 1].item())

        # For visualization
        n = len(signal)
        fft_vals = np.abs(fft(signal))
        fft_freq = np.fft.fftfreq(n, 1.0 / config.VIDEO_FPS)
        mask = (fft_freq >= 0) & (fft_freq <= config.RPPG_FREQ_MAX)
        all_spectra.append(fft_vals[mask])
        all_freqs.append(fft_freq[mask])


    # --- Stage 3: Consolidation & Final Verdict ---
    feature_vector = np.stack([all_fda_scores, all_snr_scores, pdtcn_fake_probs], axis=1).flatten()
    feature_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        final_output = consolidator_model(feature_tensor)
        final_probs = torch.softmax(final_output, dim=1)
        confidence, prediction = torch.max(final_probs, 1)

    verdict = "FAKE" if prediction.item() == 1 else "REAL"
    confidence_score = confidence.item()

    # --- Final Output ---
    print("\n--- A.V.E.N.U.E. Prediction Result ---")
    print(f"Verdict:   {verdict}")
    print(f"Confidence: {confidence_score:.2%}")
    print("---------------------------------------")

    create_analysis_visualization(
        video_path, deformations, stable_clips_indices, 
        all_signals, all_spectra, all_freqs, 
        verdict, confidence_score
    )