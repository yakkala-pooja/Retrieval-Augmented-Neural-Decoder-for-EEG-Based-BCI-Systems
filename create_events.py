import os
import mne
import numpy as np
from pathlib import Path
import logging
import gc
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_event_file(raw, subject_id, split, output_dir):
    """
    Create event file for a single subject's recording using actual annotations.
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data
    subject_id : int
        Subject ID
    split : str
        Either 'train' or 'test'
    output_dir : str
        Directory to save event files
    """
    try:
        # Print annotations for debugging
        logger.info(f"Annotations for {split} subject {subject_id}:")
        logger.info(raw.annotations)
        
        # Extract events from annotations
        events, event_id = mne.events_from_annotations(raw)
        
        if len(events) == 0:
            logger.error(f"No events found in annotations for {split} subject {subject_id}")
            return None
        
        # Save events
        event_file = os.path.join(output_dir, f"{split}_subject_{subject_id}_events.npy")
        np.save(event_file, events)
        
        # Save event IDs
        event_id_file = os.path.join(output_dir, f"{split}_subject_{subject_id}_event_id.npy")
        np.save(event_id_file, event_id)
        
        # Save events as CSV for easier inspection
        csv_file = os.path.join(output_dir, f"{split}_subject_{subject_id}_events.csv")
        np.savetxt(csv_file, events, delimiter=',', 
                   header='sample,previous_event_id,event_id',
                   fmt='%d')
        
        # Save event IDs as text file
        event_id_txt = os.path.join(output_dir, f"{split}_subject_{subject_id}_event_id.txt")
        with open(event_id_txt, 'w') as f:
            for key, value in event_id.items():
                f.write(f"{key}: {value}\n")
        
        logger.info(f"Created event file for {split} subject {subject_id}")
        logger.info(f"Found {len(events)} events with IDs: {event_id}")
        return events, event_id
    
    except Exception as e:
        logger.error(f"Error creating event file for {split} subject {subject_id}: {str(e)}")
        return None

def load_edf_in_chunks(edf_path, chunk_duration=30):
    """
    Load EDF file in chunks to avoid memory issues.
    
    Parameters:
    -----------
    edf_path : str
        Path to the EDF file
    chunk_duration : int
        Duration of each chunk in seconds
        
    Returns:
    --------
    raw : mne.io.Raw
        Raw EEG data
    """
    try:
        # Create a temporary directory for memory mapping
        with tempfile.TemporaryDirectory() as temp_dir:
            # First, get the file info without loading the data
            raw = mne.io.read_raw_edf(edf_path, preload=False)
            
            # Get the total duration and sampling frequency
            total_duration = raw.times[-1]
            sfreq = raw.info['sfreq']
            
            # Calculate number of chunks
            n_chunks = int(np.ceil(total_duration / chunk_duration))
            
            # Process each chunk
            for i in range(n_chunks):
                start_time = i * chunk_duration
                end_time = min((i + 1) * chunk_duration, total_duration)
                
                # Extract the chunk
                chunk = raw.copy().crop(tmin=start_time, tmax=end_time)
                chunk.load_data()
                
                # Clear memory after processing each chunk
                del chunk
                gc.collect()
            
            # Return the raw object with metadata only
            return raw
    
    except Exception as e:
        logger.error(f"Error loading {edf_path}: {str(e)}")
        return None

def process_single_subject(data_dir, output_dir, subject_id, split):
    """
    Process a single subject's data.
    
    Parameters:
    -----------
    data_dir : str
        Path to the data directory
    output_dir : str
        Directory to save event files
    subject_id : int
        Subject ID
    split : str
        Either 'train' or 'test'
    """
    edf_path = os.path.join(data_dir, split, f"{subject_id}.edf")
    logger.info(f"Processing {split} subject {subject_id}")
    
    if not os.path.exists(edf_path):
        logger.error(f"File not found: {edf_path}")
        return
    
    try:
        # Load the EDF file
        raw = load_edf_in_chunks(edf_path)
        
        if raw is not None:
            # Create event file
            result = create_event_file(raw, subject_id, split, output_dir)
            
            # Clear memory
            del raw
            gc.collect()
            
            if result is not None:
                logger.info(f"Successfully processed {split} subject {subject_id}")
            else:
                logger.error(f"Failed to create events for {split} subject {subject_id}")
    
    except Exception as e:
        logger.error(f"Error processing {edf_path}: {str(e)}")

def main():
    # Set paths
    data_dir = r"D:\Retrieval-Augmented Neural Decoder for EEG-Based BCI Systems\mne_data\MNE-schirrmeister2017-data"
    output_dir = "events"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process specific problematic files first
    problematic_files = [(2, 'train'), (3, 'train'), (4, 'train')]
    
    for subject_id, split in problematic_files:
        process_single_subject(data_dir, output_dir, subject_id, split)
        # Force garbage collection between subjects
        gc.collect()
    
    # Process remaining files
    for split in ['train', 'test']:
        for subject_id in range(1, 15):
            # Skip already processed files
            if (subject_id, split) in problematic_files:
                continue
            process_single_subject(data_dir, output_dir, subject_id, split)
            # Force garbage collection between subjects
            gc.collect()
    
    logger.info("Event file creation completed")

if __name__ == "__main__":
    main() 