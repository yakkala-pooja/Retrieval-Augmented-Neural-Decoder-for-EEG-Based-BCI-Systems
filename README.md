# Retrieval-Augmented Neural Decoder for EEG-Based BCI Systems

This project implements a neural decoder for EEG-based Brain-Computer Interface (BCI) systems, focusing on high-gamma band activity processing and feature extraction.

## Project Structure

```
.
├── mne_data/
│   └── MNE-schirrmeister2017-data/
│       ├── train/          # Training EDF files
│       └── test/           # Test EDF files
├── events/                 # Event marker files
│   ├── train_subject_*_events.npy
│   └── test_subject_*_events.npy
├── data/
│   └── processed/         # Processed data output
│       ├── train_subject_*_processed.h5
│       └── test_subject_*_processed.h5
├── preprocess_data.py     # Main preprocessing script
└── requirements.txt       # Python dependencies
```

## Data Processing Pipeline

The preprocessing pipeline includes the following steps:

1. **Bandpass Filtering (70-150 Hz)**
   - Isolates high-gamma range relevant for motor/cognitive activity
   - Uses Butterworth filter implementation

2. **Downsampling (to 500 Hz)**
   - Reduces data size while maintaining high-gamma information
   - Satisfies Nyquist criterion for 150 Hz signals

3. **Epoching (Event-Based Segmentation)**
   - Time window: -0.5 to +1.5 seconds around events
   - Aligns data with cognitive states/actions

4. **Feature Extraction**
   - Log power features via Hilbert transform
   - Extracts amplitude envelopes
   - Memory-efficient implementation

5. **Feature Scaling**
   - Standardization across time/trials/channels
   - Stabilizes training and prevents bias

## Requirements

```
numpy>=1.20.0
scipy>=1.7.0
scikit-learn>=0.24.0
mne>=0.24.0
h5py>=3.0.0
tqdm>=4.0.0
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd Retrieval-Augmented-Neural-Decoder
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preprocessing

Run the preprocessing script to process both training and test data:

```bash
python preprocess_data.py
```

The script will:
1. Process training data first
2. Then process test data
3. Save processed files in `data/processed/`

### Output Format

Processed data is saved in HDF5 format with the following structure:

- Training files: `data/processed/train_subject_X_processed.h5`
- Test files: `data/processed/test_subject_X_processed.h5`

Each H5 file contains:
- Dataset: 'processed_data' - shape (epochs, channels, time_points)
- Attributes:
  - sampling_rate: 500 Hz
  - original_sampling_rate: Original EEG sampling rate
  - n_channels: Number of EEG channels
  - epoch_duration: 2.0 seconds
  - n_epochs: Total number of epochs

## Memory Optimization

The preprocessing pipeline includes several memory optimization features:
- Chunk-based processing for large files
- Memory-efficient Hilbert transform
- Aggressive garbage collection
- Float32 data type usage
- Compressed HDF5 storage

## Data Sources

The project uses the Schirrmeister2017 dataset:
- Training data: 14 subjects
- Test data: 14 subjects
- Each subject's data includes:
  - EDF files containing raw EEG recordings
  - Event files marking significant time points