import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.cuda.amp import GradScaler, autocast
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize logger
        self.logger = Logger(log_dir=os.path.join(config['checkpoint_dir'], 'logs'))
        
        # Log initial configuration
        self.logger.info("=== Training Configuration ===")
        for key, value in config.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info(f"Device: {self.device}")
        
        try:
            # Create model
            self.model = ResNet50(num_classes=1000).to(self.device)
            self.logger.info("Model created successfully")
            
            # Define loss function and optimizer
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.SGD(self.model.parameters(), 
                                     lr=config['learning_rate'],
                                     momentum=config['momentum'],
                                     weight_decay=config['weight_decay'])
            
            # Learning rate scheduler
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config['epochs'])
            
            # Gradient scaler for mixed precision training
            self.scaler = GradScaler()
            
            # Create data loaders
            self.train_loader, self.val_loader = self.get_data_loaders()
            self.logger.info("Data loaders created successfully")
            
            # Initialize best accuracy
            self.best_acc1 = 0
            
        except Exception as e:
            self.logger.error(f"Error in initialization: {str(e)}")
            raise e
        
    def get_data_loaders(self):
        # Data augmentation and normalization for training
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Just normalization for validation
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = datasets.ImageFolder(
            os.path.join(self.config['data_dir'], 'train'),
            transform=train_transform
        )
        
        val_dataset = datasets.ImageFolder(
            os.path.join(self.config['data_dir'], 'val'),
            transform=val_transform
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_one_epoch(self, epoch):
        try:
            batch_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            
            # Switch to train mode
            self.model.train()
            
            end = time.time()
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            
            for i, (images, target) in pbar:
                # Move data to target device
                images = images.to(self.device)
                target = target.to(self.device)
                
                # Mixed precision training
                with autocast():
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
                        f'Epoch: [{epoch}][{i}/{len(self.train_loader)}] '
                        f'Loss: {losses.avg:.4f} '
                        f'Acc@1: {top1.avg:.2f}% '
                        f'Acc@5: {top5.avg:.2f}%'
                    )
            
            return losses.avg, top1.avg, top5.avg
            
        except Exception as e:
            self.logger.error(f"Error in training epoch {epoch}: {str(e)}")
            raise e
    
    @torch.no_grad()
    def validate(self):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        # Switch to evaluate mode
        self.model.eval()
        
        pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
        
        for i, (images, target) in pbar:
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
            
            pbar.set_description(
                f'Test: '
                f'Loss: {losses.avg:.4f} '
                f'Acc@1: {top1.avg:.2f}% '
                f'Acc@5: {top5.avg:.2f}%'
            )
        
        return losses.avg, top1.avg, top5.avg
    
    def train(self):
        self.logger.info("=== Starting Training ===")
        try:
            for epoch in range(self.config['epochs']):
                self.logger.info(f"Starting epoch {epoch+1}/{self.config['epochs']}")
                
                # Train for one epoch
                train_loss, train_acc1, train_acc5 = self.train_one_epoch(epoch)
                
                # Evaluate on validation set
                val_loss, val_acc1, val_acc5 = self.validate()
                
                # Remember best acc@1 and save checkpoint
                is_best = val_acc1 > self.best_acc1
                self.best_acc1 = max(val_acc1, self.best_acc1)
                
                # Log epoch results
                self.logger.info(
                    f"Epoch {epoch+1} Results:\n"
                    f"Training - Loss: {train_loss:.4f}, Acc@1: {train_acc1:.2f}%, Acc@5: {train_acc5:.2f}%\n"
                    f"Validation - Loss: {val_loss:.4f}, Acc@1: {val_acc1:.2f}%, Acc@5: {val_acc5:.2f}%\n"
                    f"Best Acc@1: {self.best_acc1:.2f}%"
                )
                
                # Save checkpoint
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'best_acc1': self.best_acc1,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best)
                
                # Update learning rate
                self.scheduler.step()
                
            self.logger.info("=== Training Completed ===")
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise e
    
    def save_checkpoint(self, state, is_best):
        filename = os.path.join(self.config['checkpoint_dir'], 'checkpoint.pth')
        torch.save(state, filename)
        if is_best:
            best_filename = os.path.join(self.config['checkpoint_dir'], 'model_best.pth')
            torch.save(state, best_filename)

def main():
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