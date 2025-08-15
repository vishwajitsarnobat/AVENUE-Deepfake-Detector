# /src/analysis.py (Corrected for Multiprocessing)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt
from scipy.fft import fft
import config

# --- MODIFIED: Global variables for lazy initialization ---
# We define the variables here, but we will only create the actual
# model instance inside the worker process to avoid forking issues.
mp_face_mesh = mp.solutions.face_mesh
face_mesh_instance = None # This will hold the model instance for each worker

# Define landmark indices for facial regions (excluding eyes) based on MediaPipe's 468 landmarks
FACIAL_LANDMARK_INDECES = {
    "mouth": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
    "right_eyebrow": [70, 63, 105, 66, 107],
    "left_eyebrow": [336, 296, 334, 293, 300],
    "nose": [1, 2, 98, 327],
    "face_outline": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
}

ALL_LANDMARK_IDS = sum(FACIAL_LANDMARK_INDECES.values(), [])

# --- MODIFIED: Function now handles lazy initialization ---
def get_facial_landmarks(frame_bgr):
    """
    Extracts facial landmarks from a single frame using MediaPipe.
    It lazily initializes the FaceMesh model on a per-worker basis.
    """
    global face_mesh_instance
    
    # Each worker process will initialize its own instance of FaceMesh the first time this function is called.
    if face_mesh_instance is None:
        # print(f"[Worker PID: {os.getpid()}] Initializing MediaPipe FaceMesh model...")
        face_mesh_instance = mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
        )

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh_instance.process(frame_rgb)
    
    if not results.multi_face_landmarks:
        return None
    
    landmarks = np.array(
        [(lm.x, lm.y, lm.z) for lm in results.multi_face_landmarks[0].landmark]
    )
    return landmarks[ALL_LANDMARK_IDS]


def calculate_facial_deformation(landmarks_sequence):
    """Calculates a time series of facial deformation."""
    deformations = []
    for i in range(1, len(landmarks_sequence)):
        prev_landmarks = landmarks_sequence[i - 1]
        curr_landmarks = landmarks_sequence[i]
        if prev_landmarks is None or curr_landmarks is None:
            deformations.append(1e6)
            continue
        dist = np.linalg.norm(curr_landmarks - prev_landmarks, axis=1)
        deformations.append(np.sum(dist))
    return np.array(deformations)

def find_stable_clips(deformations, num_clips, clip_len_frames):
    """Finds the most stable clips using the min s-sum subarray approach."""
    stable_clips_indices = []
    temp_deformations = -deformations.copy()
    for _ in range(num_clips):
        max_sum = -np.inf
        best_start_idx = -1
        for i in range(len(temp_deformations) - clip_len_frames + 1):
            current_sum = np.sum(temp_deformations[i : i + clip_len_frames])
            if current_sum > max_sum:
                max_sum = current_sum
                best_start_idx = i
        if best_start_idx != -1:
            stable_clips_indices.append((best_start_idx, best_start_idx + clip_len_frames))
            temp_deformations[best_start_idx : best_start_idx + clip_len_frames] = -np.inf
    stable_clips_indices.sort()
    return stable_clips_indices

def bandpass_filter(signal, lowcut, highcut, fs):
    """Applies a bandpass filter to the signal."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(1, [low, high], btype='band')
    y = filtfilt(b, a, signal)
    return y

def extract_rppg_signal(video_clip):
    """
    Extracts the rPPG signal from a video clip using the POS algorithm.
    """
    rgb_mean_over_time = np.mean(video_clip, axis=(1, 2))
    mean_color = np.mean(rgb_mean_over_time, axis=0)
    normalized_rgb = rgb_mean_over_time / (mean_color + 1e-10)
    B, G, R = normalized_rgb[:, 0], normalized_rgb[:, 1], normalized_rgb[:, 2]
    X = R - G
    Y = R + G - 2*B
    alpha = np.std(X) / (np.std(Y) + 1e-10)
    raw_signal = X - alpha * Y
    filtered_signal = bandpass_filter(
        raw_signal,
        config.RPPG_FREQ_MIN,
        config.RPPG_FREQ_MAX,
        config.VIDEO_FPS
    )
    return filtered_signal

def calculate_snr(signal, fs):
    """Calculates the Signal-to-Noise Ratio of the pulse signal."""
    n = len(signal)
    if n == 0: return 0.0
    fft_vals = np.abs(fft(signal))
    fft_freq = np.fft.fftfreq(n, 1.0 / fs)
    mask = (fft_freq >= config.RPPG_FREQ_MIN) & (fft_freq <= config.RPPG_FREQ_MAX)
    pulse_spectrum = fft_vals[mask]
    freqs = fft_freq[mask]
    if len(pulse_spectrum) == 0: return 0.0
    peak_idx = np.argmax(pulse_spectrum)
    peak_freq = freqs[peak_idx]
    window_size = 0.2 
    signal_mask = (freqs >= peak_freq - window_size) & (freqs <= peak_freq + window_size)
    signal_power = np.sum(pulse_spectrum[signal_mask] ** 2)
    total_power = np.sum(pulse_spectrum ** 2)
    noise_power = total_power - signal_power
    if noise_power <= 0: return 100.0
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def calculate_fda(deformations_in_clip):
    """Calculates the Facial Deformation-Based Attention (FDA) score."""
    if len(deformations_in_clip) == 0: return 0.0
    sum_sq_deformations = np.sum(deformations_in_clip**2)
    if sum_sq_deformations == 0: return 1.0
    fda = 1.0 / np.sqrt(sum_sq_deformations)
    return fda