# EEG Classification Model

This repository contains code for training and evaluating a deep learning model for EEG signal classification. The model is designed to work with the Schirrmeister2017 dataset and focuses on high-gamma band (70-150 Hz) activity.

## Project Structure

```
.
├── mne_data/
│   └── MNE-schirrmeister2017-data/
│       ├── train/          # Training EDF files
│       └── test/           # Test EDF files
├── data/
│   └── processed/          # Preprocessed data in .h5 format
├── eeg_model/
│   ├── model.py           # Model architecture definition
│   ├── train.py           # Training script
│   └── evaluate.py        # Evaluation script
├── events/                # Event files (.npy)
├── outputs/               # Model outputs and results
└── preprocess_data.py     # Data preprocessing script
```

## Model Architecture

The model consists of the following components:

1. **Temporal Convolution Layers**
   - Detect temporal patterns in EEG channels
   - Multiple layers with increasing filter sizes
   - Batch normalization for training stability
   - Max pooling for dimensionality reduction

2. **Dense Layers**
   - Capture abstract patterns
   - Dropout for regularization
   - ReLU activation

3. **Output Layer**
   - Softmax activation for class probabilities
   - Multi-class classification

## Data Processing Pipeline

1. **Preprocessing**
   - Bandpass filtering (70-150 Hz) for high-gamma range
   - Downsampling to 500 Hz
   - Epoching (-0.5 to 1.5s around events)
   - Power feature extraction using Hilbert transform
   - Feature scaling/standardization

2. **Memory Optimizations**
   - Chunk-based processing
   - float32 precision
   - Efficient Hilbert transform implementation
   - Garbage collection

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing

```bash
python preprocess_data.py --data_dir mne_data/MNE-schirrmeister2017-data --output_dir data/processed
```

### 2. Training

```bash
python eeg_model/train.py \
    --data_dir data/processed \
    --train_subjects 1 2 3 4 5 6 7 8 9 10 11 12 13 14 \
    --output_dir outputs \
    --epochs 100 \
    --batch_size 32
```

### 3. Evaluation

```bash
python eeg_model/evaluate.py \
    --data_dir data/processed \
    --test_subjects 15 16 17 18 19 20 \
    --model_checkpoint outputs/model_checkpoint.pt \
    --output_dir outputs
```

## Training Strategy

- **Loss Function**: Cross-entropy loss (log-softmax + NLL)
- **Optimizer**: Adam with learning rate 0.001
- **Batch Size**: 32 (configurable)
- **Epochs**: 100 (configurable)
- **Validation Split**: 20% of training data

## Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **F1 Score**: Weighted average F1 score
- **Confusion Matrix**: Per-class performance visualization
- **Learning Curves**: Training/validation loss and accuracy plots

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MNE-Python for EEG processing tools
- Schirrmeister et al. for the original dataset
- PyTorch team for the deep learning framework

## Citation

If you use this code in your research, please cite:

```bibtex
@article{schirrmeister2017deep,
  title={Deep learning with convolutional neural networks for EEG decoding and visualization},
  author={Schirrmeister, Robin Tibor and Springenberg, Jost Tobias and Fiederer, Lukas Dominique Josef and Glasstetter, Martin and Eggensperger, Katharina and Tangermann, Michael and Hutter, Frank and Burgard, Wolfram and Ball, Tonio},
  journal={Human brain mapping},
  volume={38},
  number={11},
  pages={5391--5420},
  year={2017},
  publisher={Wiley Online Library}
}
```