import argparse
from pathlib import Path
import torch
import numpy as np
from model import EEGClassifier, EEGDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on test data.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)

def plot_confusion_matrix(y_true, y_pred, classes, output_dir):
    """
    Plot and save confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(Path(output_dir) / 'test_confusion_matrix.png')
    plt.close()

def main(args):
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    checkpoint = torch.load(args.model_checkpoint)
    checkpoint_args = checkpoint['args']
    
    # Create test dataset
    test_dataset = EEGDataset(args.data_dir, args.test_subjects, split='test')
    
    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Initialize model
    model = EEGClassifier(
        n_channels=test_dataset.data.shape[1],
        n_timepoints=test_dataset.data.shape[2],
        n_classes=len(torch.unique(test_dataset.labels)),
        temporal_filters=checkpoint_args.temporal_filters,
        kernel_sizes=checkpoint_args.kernel_sizes,
        pool_sizes=checkpoint_args.pool_sizes,
        dense_units=checkpoint_args.dense_units
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    model = model.to(device)
    
    # Evaluate model
    predictions, true_labels = evaluate_model(model, test_loader, device)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    # Print results
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test F1 Score: {f1:.4f}')
    
    # Plot confusion matrix
    classes = [f'Class {i}' for i in range(len(torch.unique(test_dataset.labels)))]
    plot_confusion_matrix(true_labels, predictions, classes, args.output_dir)
    
    # Save results
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'predictions': predictions.tolist(),
        'true_labels': true_labels.tolist()
    }
    
    # Save results to file
    import json
    with open(Path(args.output_dir) / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate EEG Classifier')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Directory containing processed data')
    parser.add_argument('--test_subjects', type=int, nargs='+', default=list(range(15, 21)),
                        help='Subject IDs to use for testing')
    
    # Model arguments
    parser.add_argument('--model_checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA evaluation')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save evaluation outputs')
    
    args = parser.parse_args()
    main(args) 