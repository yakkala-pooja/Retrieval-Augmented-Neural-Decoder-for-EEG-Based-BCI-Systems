import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, hilbert
from sklearn.preprocessing import StandardScaler
import mne
import os
from pathlib import Path
import h5py
from tqdm import tqdm
import gc
import time

def create_bandpass_filter(lowcut=70, highcut=150, fs=1000, order=4):
    """
    Create a bandpass filter for the high-gamma range.
    
    Args:
        lowcut (float): Lower frequency bound (Hz)
        highcut (float): Upper frequency bound (Hz)
        fs (float): Sampling frequency (Hz)
        order (int): Filter order
        
    Returns:
        b, a: Filter coefficients
    """
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, fs=1000):
    """
    Apply bandpass filter to EEG data.
    
    Args:
        data (np.array): EEG data of shape (channels, samples)
        fs (float): Sampling frequency (Hz)
        
    Returns:
        np.array: Filtered data
    """
    b, a = create_bandpass_filter(fs=fs)
    filtered_data = np.zeros_like(data)
    for ch in range(data.shape[0]):
        filtered_data[ch] = filtfilt(b, a, data[ch])
    return filtered_data

def downsample_data(data, original_fs=1000, target_fs=500):
    """
    Downsample the EEG data to target frequency.
    
    Args:
        data (np.array): EEG data of shape (channels, samples)
        original_fs (float): Original sampling frequency (Hz)
        target_fs (float): Target sampling frequency (Hz)
        
    Returns:
        np.array: Downsampled data
    """
    downsample_factor = int(original_fs / target_fs)
    return signal.decimate(data, downsample_factor, axis=1)

def epoch_data(data, events, tmin=-0.5, tmax=1.5, fs=500):
    """
    Segment continuous data into epochs around events.
    
    Args:
        data (np.array): EEG data of shape (channels, samples)
        events (np.array): Event timestamps in samples
        tmin (float): Start time of epoch relative to event (seconds)
        tmax (float): End time of epoch relative to event (seconds)
        fs (float): Sampling frequency (Hz)
        
    Returns:
        np.array: Epoched data of shape (epochs, channels, samples)
    """
    samples_before = int(abs(tmin) * fs)
    samples_after = int(tmax * fs)
    epoch_samples = samples_before + samples_after
    
    epochs = np.zeros((len(events), data.shape[0], epoch_samples))
    
    for i, event in enumerate(events):
        start_idx = event - samples_before
        end_idx = event + samples_after
        if start_idx >= 0 and end_idx <= data.shape[1]:
            epochs[i] = data[:, start_idx:end_idx]
    
    return epochs

def extract_power_features_memory_efficient(data, chunk_size=100):
    """
    Extract log power features using Hilbert transform in a memory-efficient way.
    Process channels in smaller chunks to avoid memory issues.
    
    Args:
        data (np.array): EEG data of shape (epochs, channels, samples)
        chunk_size (int): Number of channels to process at once
        
    Returns:
        np.array: Log power features
    """
    n_epochs, n_channels, n_samples = data.shape
    log_power = np.zeros_like(data, dtype=np.float32)  # Use float32 instead of float64
    
    # Process channels in chunks
    for start_ch in range(0, n_channels, chunk_size):
        end_ch = min(start_ch + chunk_size, n_channels)
        
        # Process this chunk of channels
        chunk = data[:, start_ch:end_ch, :]
        
        # Apply Hilbert transform and get analytic signal
        # Process each epoch separately to save memory
        for i in range(n_epochs):
            analytic_signal = hilbert(chunk[i], axis=-1)
            amplitude_envelope = np.abs(analytic_signal)
            log_power[i, start_ch:end_ch] = np.log(amplitude_envelope ** 2 + 1e-10)
            
            # Clear memory
            del analytic_signal, amplitude_envelope
        
        # Clear memory
        del chunk
        gc.collect()
    
    return log_power

def scale_features(features):
    """
    Standardize features across time/trials/channels.
    
    Args:
        features (np.array): Feature data of shape (epochs, channels, samples)
        
    Returns:
        np.array: Scaled features
    """
    original_shape = features.shape
    # Reshape to 2D for scaling
    features_2d = features.reshape(-1, features.shape[-1])
    
    # Initialize and fit scaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_2d)
    
    # Reshape back to original dimensions
    return scaled_features.reshape(original_shape)

def preprocess_eeg(raw_data, events, fs=1000):
    """
    Complete preprocessing pipeline for EEG data.
    
    Args:
        raw_data (np.array): Raw EEG data of shape (channels, samples)
        events (np.array): Event timestamps in samples
        fs (float): Original sampling frequency (Hz)
        
    Returns:
        np.array: Preprocessed data
    """
    # 1. Bandpass filter (70-150 Hz)
    filtered_data = apply_bandpass_filter(raw_data, fs=fs)
    
    # 2. Downsample to 500 Hz
    downsampled_data = downsample_data(filtered_data, original_fs=fs, target_fs=500)
    
    # 3. Epoch the data
    epoched_data = epoch_data(downsampled_data, events, fs=500)
    
    # 4. Extract power features
    power_features = extract_power_features_memory_efficient(epoched_data)
    
    # 5. Scale features
    scaled_features = scale_features(power_features)
    
    return scaled_features

def load_events_for_subject(events_dir, subject_id, split='train'):
    """
    Load events for a specific subject.
    
    Args:
        events_dir (str): Path to events directory
        subject_id (int): Subject ID number
        split (str): Either 'train' or 'test'
        
    Returns:
        np.array: Array of event timestamps
    """
    events_file = Path(events_dir) / f"{split}_subject_{subject_id}_events.npy"
    if not events_file.exists():
        raise FileNotFoundError(f"Events file not found: {events_file}")
    return np.load(events_file)

def process_chunk_with_memory_efficiency(chunk_data, chunk_events, fs):
    """
    Process a single chunk of data with memory efficiency.
    
    Args:
        chunk_data (np.array): EEG data chunk
        chunk_events (np.array): Events in this chunk
        fs (float): Sampling frequency
        
    Returns:
        np.array: Processed chunk data
    """
    try:
        # Convert data to float32 to save memory
        chunk_data = chunk_data.astype(np.float32)
        
        # 1. Bandpass filter
        filtered_data = apply_bandpass_filter(chunk_data, fs=fs)
        del chunk_data
        gc.collect()
        
        # 2. Downsample to 500 Hz
        downsampled_data = downsample_data(filtered_data, original_fs=fs, target_fs=500)
        del filtered_data
        gc.collect()
        
        # 3. Epoch the data
        epoched_data = epoch_data(downsampled_data, chunk_events, fs=500)
        del downsampled_data
        gc.collect()
        
        if epoched_data.size == 0:  # No valid epochs in this chunk
            return None
        
        # 4. Extract power features with memory efficiency
        power_features = extract_power_features_memory_efficient(epoched_data, chunk_size=10)
        del epoched_data
        gc.collect()
        
        # 5. Scale features
        scaled_features = scale_features(power_features)
        del power_features
        gc.collect()
        
        return scaled_features.astype(np.float32)  # Return as float32 to save memory
        
    except Exception as e:
        print(f"Error in process_chunk_with_memory_efficiency: {str(e)}")
        return None

def process_edf_in_chunks(edf_file, events, chunk_size=1000, overlap=100, output_file=None):
    """
    Process an EDF file in chunks to handle large files.
    
    Args:
        edf_file (str): Path to the EDF file
        events (np.array): Event timestamps for this file
        chunk_size (int): Number of samples per chunk (reduced for memory efficiency)
        overlap (int): Number of samples to overlap between chunks
        output_file (str): Path to save processed data (optional)
    """
    # Load EDF file using MNE
    raw = mne.io.read_raw_edf(edf_file, preload=False)
    
    # Get file parameters
    n_channels = len(raw.ch_names)
    n_samples = raw.n_times
    fs = raw.info['sfreq']
    
    # Calculate epoch size in samples (2 seconds total: -0.5 to 1.5)
    epoch_samples = int(2.0 * 500)  # 2 seconds at 500Hz after downsampling
    
    # Calculate number of chunks
    n_chunks = int(np.ceil(n_samples / (chunk_size - overlap)))
    
    # Initialize output storage
    if output_file:
        h5f = h5py.File(output_file, 'w')
        # Initialize with epoch size instead of chunk size
        processed_data = h5f.create_dataset('processed_data', 
                                          shape=(0, n_channels, epoch_samples),
                                          maxshape=(None, n_channels, epoch_samples),
                                          chunks=(1, n_channels, epoch_samples),  # Optimize chunk size
                                          compression='gzip',  # Add compression
                                          compression_opts=1,  # Light compression
                                          dtype='float32')  # Use float32 instead of float64
        
        # Add metadata
        h5f.attrs['sampling_rate'] = 500  # Store downsampled rate
        h5f.attrs['original_sampling_rate'] = fs
        h5f.attrs['n_channels'] = n_channels
        h5f.attrs['epoch_duration'] = 2.0  # seconds
    else:
        processed_chunks = []
    
    # Process each chunk
    for i in tqdm(range(n_chunks), desc=f"Processing {Path(edf_file).stem}"):
        try:
            # Calculate chunk boundaries
            start = i * (chunk_size - overlap)
            end = min(start + chunk_size, n_samples)
            
            # Load chunk data
            chunk_data = raw.get_data(start=start, stop=end)
            
            # Find events that fall within this chunk
            # Add padding to ensure we don't miss events near chunk boundaries
            padded_start = max(0, start - int(0.5 * fs))  # Add 0.5s padding
            padded_end = min(n_samples, end + int(0.5 * fs))  # Add 0.5s padding
            chunk_events = events[(events >= padded_start) & (events < padded_end)]
            
            # Adjust event timestamps to be relative to chunk start
            chunk_events = chunk_events - start
            
            if len(chunk_events) > 0:  # Only process chunks with events
                # Process the chunk with memory efficiency
                processed_chunk = process_chunk_with_memory_efficiency(chunk_data, chunk_events, fs)
                
                if processed_chunk is not None:
                    if output_file:
                        # Resize dataset and append new data
                        current_size = processed_data.shape[0]
                        new_size = current_size + processed_chunk.shape[0]
                        processed_data.resize(new_size, axis=0)
                        processed_data[current_size:new_size] = processed_chunk
                        
                        # Flush to disk periodically
                        if i % 10 == 0:
                            h5f.flush()
                    else:
                        processed_chunks.append(processed_chunk)
                
                # Force garbage collection
                del processed_chunk
                gc.collect()
            
            # Clean up chunk data
            del chunk_data
            gc.collect()
            
        except Exception as e:
            print(f"Error processing chunk {i}: {str(e)}")
            continue
    
    if output_file:
        # Store final metadata
        h5f.attrs['n_epochs'] = processed_data.shape[0]
        h5f.close()
        return None
    else:
        # Combine all processed chunks
        if processed_chunks:
            combined = np.concatenate(processed_chunks, axis=0)
            del processed_chunks
            gc.collect()
            return combined
        return None

def verify_processed_file(file_path):
    """
    Verify that a processed file contains valid data.
    
    Args:
        file_path (str): Path to the HDF5 file
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    try:
        with h5py.File(file_path, 'r') as f:
            # Check if dataset exists
            if 'processed_data' not in f:
                return False
                
            # Check if dataset has data
            if f['processed_data'].shape[0] == 0:
                return False
                
            # Check if metadata exists
            required_attrs = ['sampling_rate', 'original_sampling_rate', 'n_channels', 'epoch_duration']
            if not all(attr in f.attrs for attr in required_attrs):
                return False
                
            return True
    except:
        return False

def batch_process_directory(edf_dir, events_dir, output_dir, split='train', chunk_size=2000, overlap=200):
    """
    Process all EDF files in a directory.
    
    Args:
        edf_dir (str): Directory containing EDF files
        events_dir (str): Directory containing event files
        output_dir (str): Directory to save processed data
        split (str): Either 'train' or 'test'
        chunk_size (int): Number of samples per chunk (reduced for memory efficiency)
        overlap (int): Number of samples to overlap between chunks
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Keep track of processing status
    processed_files = []
    failed_files = []
    
    # Process each EDF file
    edf_path = Path(edf_dir)
    for edf_file in sorted(edf_path.glob('*.edf')):
        # Extract subject number from filename
        subject_id = int(edf_file.stem)
        output_file = Path(output_dir) / f"{split}_subject_{subject_id}_processed.h5"
        
        # Skip if file exists and is valid
        if output_file.exists() and verify_processed_file(output_file):
            print(f"\nSkipping subject {subject_id} - already processed")
            processed_files.append(subject_id)
            continue
            
        try:
            # Load events for this subject
            events = load_events_for_subject(events_dir, subject_id, split)
            
            print(f"\nProcessing subject {subject_id}")
            
            # Remove file if it exists but is invalid
            if output_file.exists():
                output_file.unlink()
            
            process_edf_in_chunks(
                str(edf_file),
                events,
                chunk_size=chunk_size,
                overlap=overlap,
                output_file=str(output_file)
            )
            
            # Verify the processed file
            if verify_processed_file(output_file):
                print(f"Successfully processed subject {subject_id}")
                processed_files.append(subject_id)
            else:
                print(f"Error: Processed file for subject {subject_id} is invalid")
                if output_file.exists():
                    output_file.unlink()
                failed_files.append(subject_id)
            
            # Force garbage collection between subjects
            gc.collect()
            
        except Exception as e:
            print(f"Error processing subject {subject_id}: {str(e)}")
            failed_files.append(subject_id)
            # Clean up failed file
            if output_file.exists():
                output_file.unlink()
            
        # Optional: Sleep between subjects to let system recover
        time.sleep(1)
    
    # Print summary
    print("\nProcessing Summary:")
    print(f"Successfully processed subjects: {sorted(processed_files)}")
    if failed_files:
        print(f"Failed to process subjects: {sorted(failed_files)}")
    else:
        print("No failures!")

# Example usage:
if __name__ == "__main__":
    # Directory structure
    events_dir = "events"
    output_dir = "data/processed"
    
    # Step 1: Process training data
    print("\n=== Processing TRAINING data ===")
    train_dir = "mne_data/MNE-schirrmeister2017-data/train"
    batch_process_directory(train_dir, events_dir, output_dir, split='train')
    
    # Step 2: Process test data
    print("\n=== Processing TEST data ===")
    test_dir = "mne_data/MNE-schirrmeister2017-data/test"
    batch_process_directory(test_dir, events_dir, output_dir, split='test')
    
    print("\nPreprocessing completed for both training and test data!")
    print("Processed files are saved in:", output_dir)
    print("- Training files format: train_subject_X_processed.h5")
    print("- Test files format: test_subject_X_processed.h5")
