import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.amp import GradScaler, autocast
from resnet50 import ResNet50
from tqdm import tqdm
from logger import Logger
import sys
from config import load_config

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class ImageNetTrainer:
    def __init__(self, config):
        self.config = config
        
        # Initialize logger first
        self.logger = Logger(log_dir=config['log_dir'])
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            self.logger.warning("CUDA is not available. Training will be done on CPU.")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda')
            self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"Number of available GPUs: {torch.cuda.device_count()}")
        
        # Log initial configuration
        self.logger.info("=== Training Configuration ===")
        for key, value in config.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info(f"Device: {self.device}")
        
        # Initialize checkpoint tracking
        self.top_checkpoints = []  # List to store top 5 checkpoints
        
        try:
            # Create model
            self.model = ResNet50(num_classes=1000).to(self.device)
            # Use DataParallel if multiple GPUs are available
            if torch.cuda.device_count() > 1:
                self.logger.info(f"Using {torch.cuda.device_count()} GPUs for training")
                self.model = nn.DataParallel(self.model)
            self.logger.info("Model created successfully")
            
            # Define loss function and optimizer
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.SGD(self.model.parameters(), 
                                     lr=config['learning_rate'],
                                     momentum=config['momentum'],
                                     weight_decay=config['weight_decay'])
            
            # StepLR scheduler
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config['lr_step_size'],  # Decay LR every n epochs
                gamma=config['lr_gamma']  # Multiplicative factor of learning rate decay
            )
            
            # Gradient scaler for mixed precision training
            self.scaler = GradScaler()
            
            # Create data loaders
            self.train_loader, self.test_loader = self.get_data_loaders()
            self.logger.info("Data loaders created successfully")
            
            # Initialize best accuracy
            self.best_acc1 = 0
            
        except Exception as e:
            self.logger.error(f"Error in initialization: {str(e)}")
            raise e
        
    def get_data_loaders(self):
        # Data augmentation and normalization for training
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), 
                                         interpolation=transforms.InterpolationMode.BILINEAR, 
                                         antialias=True),
            transforms.RandomHorizontalFlip(0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Just normalization for testing
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = datasets.ImageFolder(
            self.config['train_dir'],
            transform=train_transform
        )
        
        test_dataset = datasets.ImageFolder(
            self.config['test_dir'],
            transform=test_transform
        )
        
        # Create random sampler for training data
        train_sampler = torch.utils.data.RandomSampler(
            train_dataset,
            replacement=False,  # sample without replacement
            num_samples=None  # draw all samples
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,  # Use random sampler instead of shuffle
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        return train_loader, test_loader
    
    def train_one_epoch(self, epoch):
        try:
            batch_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            
            # Switch to train mode
            self.model.train()
            
            end = time.time()
            num_batches = len(self.train_loader)
            
            for i, (images, target) in enumerate(self.train_loader):
                # Move data to target device
                images = images.to(self.device)
                target = target.to(self.device)
                
                # Mixed precision training
                with autocast(device_type='cuda', dtype=torch.float16):
                    output = self.model(images)
                    loss = self.criterion(output, target)
                
                # Compute accuracy
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                
                # Compute gradient and do SGD step
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                
                if i % 100 == 0:  # Log every 100 batches
                    self.logger.info(
                        f'Epoch: [{epoch}][{i}/{num_batches}] '
                        f'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        f'Loss: {losses.avg:.4f} '
                        f'Acc@1: {top1.avg:.2f}% '
                        f'Acc@5: {top5.avg:.2f}%'
                    )
            
            return losses.avg, top1.avg, top5.avg
            
        except Exception as e:
            self.logger.error(f"Error in training epoch {epoch}: {str(e)}")
            raise e
    
    @torch.no_grad()
    def test(self):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        # Switch to evaluate mode
        self.model.eval()
        
        num_batches = len(self.test_loader)
        
        for i, (images, target) in enumerate(self.test_loader):
            images = images.to(self.device)
            target = target.to(self.device)
            
            # Compute output
            output = self.model(images)
            loss = self.criterion(output, target)
            
            # Measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            
            if i % 100 == 0:  # Log every 100 batches
                self.logger.info(
                    f'Test: [{i}/{num_batches}] '
                    f'Loss: {losses.avg:.4f} '
                    f'Acc@1: {top1.avg:.2f}% '
                    f'Acc@5: {top5.avg:.2f}%'
                )

        self.logger.info(f"Test: Loss: {losses.avg:.4f} Acc@1: {top1.avg:.2f}% Acc@5: {top5.avg:.2f}%")
        
        return losses.avg, top1.avg, top5.avg
    
    def train(self):
        self.logger.info("=== Starting Training ===")
        start_epoch = 0
        
        # Try to find and load the latest checkpoint
        latest_checkpoint_path = os.path.join(self.config['checkpoint_dir'], f'checkpoint_latest.pth')
        if os.path.exists(latest_checkpoint_path):
            self.logger.info(f"Found latest checkpoint: {latest_checkpoint_path}")
            checkpoint = self.load_checkpoint(latest_checkpoint_path)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.scaler.load_state_dict(checkpoint['scaler'])
            self.best_acc1 = checkpoint['best_acc1']
            self.config = checkpoint['config']
            start_epoch = checkpoint['epoch']
        
        try:
            for epoch in range(start_epoch, self.config['epochs']):
                self.logger.info(f"Starting epoch {epoch+1}/{self.config['epochs']}")
                
                # Log current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(f"Current learning rate: {current_lr}")
                
                # Train for one epoch
                train_loss, train_acc1, train_acc5 = self.train_one_epoch(epoch)
                
                # Evaluate on test set
                test_loss, test_acc1, test_acc5 = self.test()
                
                # Remember best acc@1 and save checkpoint
                is_best = test_acc1 > self.best_acc1
                self.best_acc1 = max(test_acc1, self.best_acc1)
                
                # Log epoch results
                self.logger.info(
                    f"Epoch {epoch+1} Results:\n"
                    f"Training - Loss: {train_loss:.4f}, Acc@1: {train_acc1:.2f}%, Acc@5: {train_acc5:.2f}%\n"
                    f"Test - Loss: {test_loss:.4f}, Acc@1: {test_acc1:.2f}%, Acc@5: {test_acc5:.2f}%\n"
                    f"Best Acc@1: {self.best_acc1:.2f}%"
                )
                
                # Save checkpoint
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'best_acc1': self.best_acc1,
                    'test_acc1': test_acc1,
                }, is_best)
                
                # Update learning rate
                self.scheduler.step()
                
                # Log learning rate change
                new_lr = self.optimizer.param_groups[0]['lr']
                if new_lr != current_lr:
                    self.logger.info(f"Learning rate changed from {current_lr} to {new_lr}")
                
            self.logger.info("=== Training Completed ===")
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise e
    
    def save_checkpoint(self, state, is_best):
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        
        # Save checkpoint with epoch number
        filename = os.path.join(self.config['checkpoint_dir'], f'checkpoint_epoch_{state["epoch"]}.pth')
        
        # Prepare checkpoint state
        checkpoint = {
            'epoch': state['epoch'],
            'model': self.model.state_dict(),
            'best_acc1': state['best_acc1'],
            'test_acc1': state.get('test_acc1', 0.0),  # Store test accuracy
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'config': self.config,
        }
        
        # Save the checkpoint
        self.logger.info(f"Saving checkpoint for epoch {state['epoch']}")
        torch.save(checkpoint, filename)
        torch.save(checkpoint, os.path.join(self.config['checkpoint_dir'], f'checkpoint_latest.pth'))
        
        # Update top checkpoints list
        self.top_checkpoints.append((filename, state.get('test_acc1', 0.0)))
        self.top_checkpoints.sort(key=lambda x: x[1], reverse=True)  # Sort by accuracy
        
        if is_best:
            best_filename = os.path.join(self.config['checkpoint_dir'], 'model_best.pth')
            self.logger.info(f"Saving best model with accuracy {state['best_acc1']:.2f}%")
            torch.save(checkpoint, best_filename)
        
        # Keep only top 5 checkpoints by accuracy
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self, keep_top_n=5):
        """Remove checkpoints, keeping only the top N by validation accuracy"""
        checkpoint_dir = self.config['checkpoint_dir']
        # Get list of all checkpoints except the ones in top N
        checkpoints_to_remove = []
        if len(self.top_checkpoints) > keep_top_n:
            checkpoints_to_remove = [cp[0] for cp in self.top_checkpoints[keep_top_n:]]
            self.top_checkpoints = self.top_checkpoints[:keep_top_n]
        
        # Remove checkpoints not in top N
        for checkpoint_path in checkpoints_to_remove:
            try:
                os.remove(checkpoint_path)
                self.logger.info(f"Removed checkpoint: {os.path.basename(checkpoint_path)}")
            except Exception as e:
                self.logger.warning(f"Failed to remove checkpoint {checkpoint_path}: {str(e)}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and resume training state"""
        try:
            self.logger.info(f"Loading checkpoint from {checkpoint_path}")
            # Load full checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model'])
            
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            # Load scheduler state
            if 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            
            # Load AMP scaler state
            if 'scaler' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler'])
            
            # Get epoch and best accuracy
            start_epoch = checkpoint.get('epoch', 0)
            self.best_acc1 = checkpoint.get('best_acc1', 0.0)
            
            self.logger.info(f"Loaded checkpoint from epoch {start_epoch} "
                           f"with best accuracy {self.best_acc1:.2f}%")
            
            return start_epoch
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            raise e

def main():
    # Check CUDA availability before loading configuration
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU for training.")
        # Set CUDA device for optimal performance
        torch.backends.cudnn.benchmark = True
    else:
        print("WARNING: CUDA is not available. Training will be slow on CPU.")

    # Load configuration from environment variables
    config = load_config()
    
    try:
        # Create checkpoint directory
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        
        # Create trainer and start training
        trainer = ImageNetTrainer(config)
        trainer.train()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 