# ResNet50 ImageNet Training

This project implements ResNet50 architecture from scratch and provides a complete training pipeline for the ImageNet dataset using PyTorch.

## Project Structure 
```
├── README.md
├── resnet50.py # ResNet50 model implementation
├── train_imagenet.py # Training script
├── config.py # Configuration loader
├── logger.py # Logging utility
├── requirements.txt # Project dependencies
├── .env # Environment variables (not in git)
├── .env.example # Example environment variables
└── scripts/
└── convert_val.py # Script to organize validation data
```

## Features
- Complete ResNet50 implementation from scratch
- Mixed precision training with automatic mixed precision (AMP)
- Multi-GPU support with DataParallel
- Configurable learning rate scheduling
- Checkpoint saving and loading
- Progress tracking with logging
- Top-k accuracy monitoring
- Random sampling for training data
- Automatic resumption from latest checkpoint

## Requirements
- Python 3.8+
- PyTorch 1.9.0+
- CUDA-capable GPU (recommended)

## Installation

1. Clone the repository:
```
git clone <repository-url>
cd resnet50-imagenet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your specific paths and configurations
```

## Data Preparation

1. Download ImageNet dataset
2. Organize validation data using the provided script:
```
python scripts/convert_val.py \
--dir /path/to/validation/folder \
--labels /path/to/validation/labels.csv
```

## Configuration

Edit `.env` file to configure:
- Data directories
- Training parameters
- Learning rate schedule
- Batch size and workers
- Checkpoint directory

Example configuration:
```
IMAGENET_TRAIN_DIR=/path/to/train
IMAGENET_TEST_DIR=/path/to/validation
CHECKPOINT_DIR=./checkpoints
LOG_DIR=./logs
BATCH_SIZE=256
NUM_WORKERS=8
LEARNING_RATE=0.1
MOMENTUM=0.9
WEIGHT_DECAY=1e-4
EPOCHS=100
LR_STEP_SIZE=30
LR_GAMMA=0.1
```

## Training

Start training:
```bash
python train_imagenet.py
```

The training script will:
- Automatically use available GPUs
- Load and resume from the latest checkpoint if available
- Save checkpoints periodically
- Log training progress
- Keep the top 5 best-performing models

## Monitoring

Training progress can be monitored through:
- Console output with tqdm progress bars
- Log files in the configured log directory
- Checkpoint files showing best accuracies

## Model Checkpoints

The training process saves several types of checkpoints:
- Regular epoch checkpoints
- Best model checkpoint
- Latest checkpoint for resuming training
- Top 5 models by accuracy

## Features in Detail

### Mixed Precision Training
- Uses PyTorch AMP for faster training
- Automatically handles scaling of gradients

### Learning Rate Scheduling
- Step learning rate decay
- Configurable step size and decay factor

### Data Loading
- Random sampling for training data
- Proper data augmentation
- Efficient data loading with pinned memory

### Logging
- Detailed logging of training progress
- Separate log files for each training run
- Console and file output

## License
MIT

## Acknowledgments
- ResNet PyTorch implementation by [pytorch/vision](https://github.com/pytorch/vision)
- PyTorch team
- ImageNet dataset creators
```
This README provides:
1. Clear project structure
2. Installation instructions
3. Configuration details
4. Usage instructions
5. Feature explanations
6. Monitoring and checkpoint information

You can customize it further by:
1. Adding your specific license
2. Including contribution guidelines
3. Adding more detailed setup instructions
4. Including performance benchmarks
5. Adding troubleshooting section
6. Including citation information
```



