# AVENUE: A Novel Deepfake Detection Method

This repository contains a Python implementation of the research paper **"AVENUE: A Novel Deepfake Detection Method Based on Temporal Convolutional Network and rPPG Information"**. The project provides a complete, modular pipeline to preprocess video data, train the deepfake detection models, and predict whether a given video is real or a deepfake, all controlled via a simple command-line interface.

The core of the method relies on extracting physiological signals (specifically, the remote photoplethysmography or rPPG signal) from stable portions of a video, as these signals are often disrupted or absent in deepfakes.

## ✨ Features

-   **Modular Architecture:** A clean separation of concerns with a central `config.py` for easy configuration and a `src/` directory for all logic.
-   **Command-Line Interface:** All operations are handled through `main.py` with distinct commands: `preprocess`, `train`, and `predict`.
-   **Automated Data Handling:** Automatically splits raw video data into training and validation sets based on a configurable ratio.
-   **Parallel Preprocessing:** Utilizes multiple CPU cores to significantly speed up the time-consuming video analysis step.
-   **Advanced Training:** Implements automatic checkpointing after each epoch, saves the best-performing model based on validation accuracy, and generates confusion matrices.
-   **Detailed Prediction Analysis:** The `predict` command not only gives a terminal verdict but also generates a comprehensive visualization image showing:
    -   A facial deformation timeline for the entire video.
    -   The specific "stable clips" selected for analysis.
    -   The rPPG signal and pulse spectrum for each selected clip.
    -   A clear, color-coded final verdict.

## 📂 Project Structure

```
AVENUE_Deepfake_Detector/
├── main.py                 # CLI entry point (preprocess, train, predict)
├── config.py               # Central configuration for paths and parameters
├── requirements.txt        # Python package dependencies
├── README.md               # This file
├── data_processed/         # Preprocessed data is saved here (auto-generated)
├── checkpoints/            # Model checkpoints are saved here (auto-generated)
├── outputs/                # Training plots and prediction images are saved here
│   ├── training_plots/
│   └── predictions/
└── src/
    ├── __init__.py
    ├── data_loader.py      # Pytorch datasets and data loading
    ├── analysis.py         # Face analysis, stable clip extraction, rPPG, features
    ├── models.py           # PyTorch models (PD-TCN, Consolidator)
    ├── trainer.py          # Training and validation loop logic
    └── predictor.py        # Inference pipeline and visualization
```

## 🚀 Setup and Installation

This guide uses `uv`, a fast, modern Python package manager. It can be used as a drop-in replacement for `pip` and `venv`.

### Prerequisites

-   Python 3.10+
-   Git

### Installation Steps

1.  **Clone the Repository**
    ```bash
    git clone <your-repository-url>
    cd AVENUE_Deepfake_Detector
    ```

2.  **Install `uv`**
    If you don't have `uv` installed, you can do so with pip, curl, or PowerShell:
    ```bash
    # Via pip (recommended)
    pip install uv

    # Or via curl (Linux/macOS)
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

3.  **Create and Activate Virtual Environment**
    Use `uv` to create a new virtual environment.
    ```bash
    uv venv
    ```
    Activate the environment.
    ```bash
    # On Linux/macOS
    source .venv/bin/activate

    # On Windows (Command Prompt)
    .venv\Scripts\activate
    ```

4.  **Install Dependencies**
    Use `uv` to install all required packages from `requirements.txt`. This will be significantly faster than a standard `pip install`.
    ```bash
    uv pip install -r requirements.txt
    ```

## ⚙️ Usage

The entire pipeline is a three-step process: configure, preprocess, and train. Prediction can be run anytime after training.

### Step 1: Configure the Project

Before running anything, you **must** edit the `config.py` file.

-   **`RAW_DATA_PATHS`**: Update the paths to point to the folders containing your real and fake video datasets. You can provide a list of multiple folders for each category.
-   **`NUM_WORKERS`**: By default, this is set to use most of your available CPU cores for preprocessing. Adjust if needed.

### Step 2: Preprocess the Data

This command will read all videos from the paths you specified, automatically split them into training and validation sets, analyze every video in parallel, and save the extracted features to the `data_processed/` directory.

```bash
uv run main.py preprocess
```
This step can take a long time, but it is resumable. If it's interrupted, you can run the same command again to pick up where it left off.

### Step 3: Train the Models

After preprocessing is complete, you can start training the neural networks. This command will execute the full two-phase training pipeline.

```bash
uv run main.py train
```
-   **Phase 1:** Trains the `PD-TCN` model on individual rPPG signals.
-   **Phase 2:** Trains the `Consolidator` model on the combined features of all stable clips.

The best models will be saved in the `checkpoints/` directory. This step is significantly faster on a CUDA-enabled GPU.

### Step 4: Predict a Video

Once the models are trained, you can run inference on any video file.

```bash
uv run main.py predict /path/to/your/video.mp4
```

**Output:**
1.  **Terminal Verdict:** You will see a clear `REAL` or `FAKE` verdict with a confidence score printed to your console.
2.  **Analysis Image:** A detailed visualization image (e.g., `analysis_video.png`) will be saved in the `outputs/predictions/` directory, showing the model's decision-making process.

## 📄 Acknowledgments

This project is an implementation of the work presented in the following paper. Please cite the original authors if you use this code in your research.

> Lokendra Birla, Trishna Saikia, and Puneet Gupta. 2025. AVENUE: A Novel Deepfake Detection Method Based on Temporal Convolutional Network and rPPG Information. *ACM Trans. Intell. Syst. Technol. 16, 1, Article 17 (January 2025), 16 pages.* https://doi.org/10.1145/3702232
