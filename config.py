import os
from dotenv import load_dotenv

def load_config():
    # Load environment variables from .env file
    load_dotenv()
    
    config = {
        'data_dir': os.getenv('IMAGENET_DATA_DIR'),
        'checkpoint_dir': os.getenv('CHECKPOINT_DIR'),
        'batch_size': int(os.getenv('BATCH_SIZE', 256)),
        'epochs': int(os.getenv('EPOCHS', 100)),
        'learning_rate': float(os.getenv('LEARNING_RATE', 0.1)),
        'momentum': float(os.getenv('MOMENTUM', 0.9)),
        'weight_decay': float(os.getenv('WEIGHT_DECAY', 1e-4)),
        'num_workers': int(os.getenv('NUM_WORKERS', 8)),
        'resume_from': os.getenv('RESUME_FROM', None),  # Path to checkpoint to resume from
    }
    
    # Validate required environment variables
    required_vars = ['data_dir', 'checkpoint_dir']
    missing_vars = [var for var in required_vars if not config[var]]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
    return config 