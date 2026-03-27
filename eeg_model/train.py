import argparse
from pathlib import Path
import torch
from model import EEGClassifier, EEGDataset, train_model
from torch.utils.data import DataLoader, random_split

def main(args):
    # Create data directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Create dataset
    full_dataset = EEGDataset(args.data_dir, args.train_subjects, split='train')
    
    # Split into train and validation
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    # Get data shape from a single batch
    sample_data, _ = next(iter(train_loader))
    
    # Initialize model
    model = EEGClassifier(
        n_channels=sample_data.shape[1],
        n_timepoints=sample_data.shape[2],
        n_classes=full_dataset.n_classes,
        temporal_filters=args.temporal_filters,
        kernel_sizes=args.kernel_sizes,
        pool_sizes=args.pool_sizes,
        dense_units=args.dense_units
    )
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, 
        n_epochs=args.epochs, 
        lr=args.learning_rate,
        device=device
    )
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'args': args
    }, Path(args.output_dir) / 'model_checkpoint.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train EEG Classifier')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Directory containing processed data')
    parser.add_argument('--train_subjects', type=int, nargs='+', default=list(range(1, 15)),
                        help='Subject IDs to use for training')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Fraction of data to use for validation')
    
    # Model arguments
    parser.add_argument('--temporal_filters', type=int, nargs='+', default=[64, 128],
                        help='Number of filters in temporal convolution layers')
    parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[7, 5],
                        help='Kernel sizes for temporal convolution layers')
    parser.add_argument('--pool_sizes', type=int, nargs='+', default=[2, 2],
                        help='Pool sizes for temporal convolution layers')
    parser.add_argument('--dense_units', type=int, nargs='+', default=[256, 128],
                        help='Number of units in dense layers')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA training')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save model outputs')
    
    args = parser.parse_args()
    main(args) 