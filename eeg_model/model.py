import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class EEGDataset(Dataset):
    """
    Dataset class for loading preprocessed EEG data.
    """
    def __init__(self, data_dir, subject_ids, split='train'):
        self.data_dir = Path(data_dir)
        self.subject_ids = subject_ids
        self.split = split
        
        # Initialize lists to store file paths and data shapes
        self.file_paths = []
        self.data_shapes = []
        self.cumulative_sizes = [0]
        self.channel_means = {}
        self.channel_stds = {}
        self.labels = {}
        
        # Load event mappings
        self.class_mapping = {
            'feet': 0,      # event_id = 1
            'left_hand': 1, # event_id = 2
            'rest': 2,      # event_id = 3
            'right_hand': 3 # event_id = 4
        }
        
        # Get total size and verify files
        total_epochs = 0
        for subject_id in subject_ids:
            # Load data file
            file_path = self.data_dir / f"{split}_subject_{subject_id}_processed.h5"
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            # Load events from CSV
            events_path = Path('events') / f"{split}_subject_{subject_id}_events.csv"
            if not events_path.exists():
                raise FileNotFoundError(f"Events file not found: {events_path}")
            
            # Read events CSV, skip header and comments
            events_data = []
            with open(events_path, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    sample, prev_id, event_id = map(int, line.strip().split(','))
                    events_data.append(event_id)
            
            # Convert event IDs to class indices (1-4 to 0-3)
            events_data = np.array(events_data) - 1
            
            with h5py.File(file_path, 'r') as f:
                n_epochs = f['processed_data'].shape[0]
                
                # Handle mismatch between epochs and labels
                if n_epochs > len(events_data):
                    print(f"Warning: More epochs ({n_epochs}) than labels ({len(events_data)}) for subject {subject_id}")
                    print(f"Using only the first {len(events_data)} epochs")
                    n_epochs = len(events_data)
                elif n_epochs < len(events_data):
                    print(f"Warning: Fewer epochs ({n_epochs}) than labels ({len(events_data)}) for subject {subject_id}")
                    print(f"Using only the first {n_epochs} labels")
                    events_data = events_data[:n_epochs]
                
                # Store labels
                self.labels[str(file_path)] = events_data
                
                # Calculate mean and std for each channel using chunks
                data = f['processed_data'][:n_epochs]
                chunk_size = 100  # Process 100 epochs at a time
                means_sum = np.zeros(data.shape[1])
                sq_diff_sum = np.zeros(data.shape[1])
                total_samples = 0
                
                for i in range(0, n_epochs, chunk_size):
                    chunk = data[i:min(i+chunk_size, n_epochs)]
                    chunk = chunk.reshape(-1, chunk.shape[1])  # Combine epochs and time points
                    means_sum += chunk.mean(axis=0) * chunk.shape[0]
                    sq_diff_sum += ((chunk ** 2).sum(axis=0))
                    total_samples += chunk.shape[0]
                
                # Calculate final mean and std
                channel_means = means_sum / total_samples
                channel_stds = np.sqrt(sq_diff_sum / total_samples - channel_means ** 2)
                channel_stds[channel_stds == 0] = 1  # Prevent division by zero
                
                self.channel_means[str(file_path)] = channel_means
                self.channel_stds[str(file_path)] = channel_stds
                
                total_epochs += n_epochs
                self.data_shapes.append((n_epochs, *data.shape[1:]))
            
            self.file_paths.append(file_path)
            self.cumulative_sizes.append(total_epochs)
        
        self.total_size = total_epochs
        self.n_classes = len(self.class_mapping)
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        # Find which file the index belongs to
        file_idx = 0
        while idx >= self.cumulative_sizes[file_idx + 1]:
            file_idx += 1
        
        # Calculate local index within the file
        local_idx = idx - self.cumulative_sizes[file_idx]
        file_path = self.file_paths[file_idx]
        
        # Load single epoch from the appropriate file
        with h5py.File(file_path, 'r') as f:
            data = f['processed_data'][local_idx:local_idx+1]
            
            # Standardize using pre-computed mean and std
            data = (data - self.channel_means[str(file_path)][None, :, None]) / self.channel_stds[str(file_path)][None, :, None]
            
            # Get label
            label = self.labels[str(file_path)][local_idx]
            
            return torch.FloatTensor(data.squeeze()), torch.LongTensor([label]).squeeze()

class EEGClassifier(nn.Module):
    """
    Neural network for EEG classification.
    """
    def __init__(self, n_channels, n_timepoints, n_classes, 
                 temporal_filters=[64, 128], kernel_sizes=[7, 5],
                 pool_sizes=[2, 2], dense_units=[256, 128]):
        super(EEGClassifier, self).__init__()
        
        # Initialize lists to hold layers
        self.temporal_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input shape: (batch_size, channels, timepoints)
        current_timepoints = n_timepoints
        current_filters = n_channels
        
        # Temporal Convolution Layers
        for filters, kernel_size, pool_size in zip(temporal_filters, kernel_sizes, pool_sizes):
            self.temporal_layers.append(
                nn.Conv1d(current_filters, filters, kernel_size, padding='same')
            )
            self.batch_norms.append(nn.BatchNorm1d(filters))
            
            current_timepoints = current_timepoints // pool_size
            current_filters = filters
        
        # Calculate flattened size
        self.flat_size = current_filters * current_timepoints
        
        # Dense layers
        self.dense_layers = nn.ModuleList()
        current_units = self.flat_size
        
        for units in dense_units:
            self.dense_layers.append(nn.Linear(current_units, units))
            current_units = units
        
        # Output layer
        self.output_layer = nn.Linear(current_units, n_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Temporal convolutions
        for conv, bn in zip(self.temporal_layers, self.batch_norms):
            x = F.relu(bn(conv(x)))
            x = F.max_pool1d(x, 2)
        
        # Flatten
        x = x.view(-1, self.flat_size)
        
        # Dense layers
        for dense in self.dense_layers:
            x = F.relu(dense(x))
            x = self.dropout(x)
        
        # Output layer
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)

def train_model(model, train_loader, val_loader, n_epochs=100, lr=0.001, device='cuda'):
    """
    Train the EEG classifier.
    """
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    
    # Training history
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Move model to device
    model = model.to(device)
    
    # Training loop
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        
        # Training
        for batch_data, batch_labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}'):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(batch_labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Calculate metrics
        val_acc = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average='weighted')
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}/{n_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Val Accuracy: {val_acc:.4f}, Val F1: {val_f1:.4f}')
    
    return train_losses, val_losses, val_accuracies

def plot_learning_curves(train_losses, val_losses, val_accuracies):
    """
    Plot training curves.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Training and Validation Loss')
    
    # Plot accuracy
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.set_title('Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Plot confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    # Example usage
    data_dir = "data/processed"
    train_subjects = list(range(1, 15))  # Subjects 1-14
    
    # Create datasets
    train_dataset = EEGDataset(data_dir, train_subjects, split='train')
    
    # Split into train and validation
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_dataset.data, train_dataset.labels, test_size=0.2, random_state=42
    )
    
    # Create data loaders
    train_loader = DataLoader(
        torch.utils.data.TensorDataset(train_data, train_labels),
        batch_size=32, shuffle=True
    )
    val_loader = DataLoader(
        torch.utils.data.TensorDataset(val_data, val_labels),
        batch_size=32
    )
    
    # Initialize model
    model = EEGClassifier(
        n_channels=train_data.shape[1],
        n_timepoints=train_data.shape[2],
        n_classes=len(torch.unique(train_dataset.labels)),
        temporal_filters=[64, 128],
        kernel_sizes=[7, 5],
        pool_sizes=[2, 2],
        dense_units=[256, 128]
    )
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, n_epochs=100, device=device
    )
    
    # Plot results
    plot_learning_curves(train_losses, val_losses, val_accuracies)
    
    # Get predictions for confusion matrix
    model.eval()
    val_preds = []
    with torch.no_grad():
        for batch_data, _ in val_loader:
            batch_data = batch_data.to(device)
            outputs = model(batch_data)
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
    
    # Plot confusion matrix
    classes = [f'Class {i}' for i in range(len(torch.unique(train_dataset.labels)))]
    plot_confusion_matrix(val_labels.numpy(), val_preds, classes) 