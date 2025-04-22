import os

import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm 
from typing import Tuple, Optional, Any, Callable, List, Type
import matplotlib.pyplot as plt
from utils.dataloader import get_dataloader

class Trainer:
    def __init__(self,
                model : nn.Module,
                dataroot : str, 
                batch_size : int = 16,
                num_workers : int = 4,
                num_epochs : int = 10, 
                learning_rate : float = 1e-3,
                save_freq : int = 4,
                checkpoint_path : str = "checkpoints/",
                finetune_encoder : bool = True,
                use_mixed_precision : bool = False,
                early_stopping : int = 3,
                device : Optional[torch.device] = None 
    ):
        self.model = model
        self.dataroot = dataroot
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.finetune_encoder = finetune_encoder
        self.save_freq = save_freq
        self.checkpoint_path = checkpoint_path
        self.use_mixed_precision = use_mixed_precision
        self.early_stopping = early_stopping

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5), # Data augmentation
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), # More augmentation
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x), # chuyển ảnh grayscale -> RGB
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # Initialize dataloaders
        ds = get_dataloader(
            root_dir=dataroot,
            transform=self.transform,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
            img_cache_size=100
        )
        self.train_loader, self.valid_loader, self.test_loader = ds["data"]
        self.tokenizer = ds["tokenizer"]
        
        self.scaler = torch.amp.GradScaler() if self.use_mixed_precision and torch.cuda.is_available() else None
        if self.use_mixed_precision and torch.cuda.is_available():
            print("Mixed precision training enabled.")
        
        # Initialize optimizer and loss function
        self.criterion: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.writer: Optional[SummaryWriter] = None

    def initialize_model(self):
        """
        Khởi tạo criterion, optimizer, scheduler với các learning rate khác nhau.
        - Decoder Embeddings: lr * 0.2
        - Other Decoder Params: lr
        - Encoder Params: lr * 0.1 (nếu finetune_encoder=True), ngược lại thì đóng băng.
        """
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.tokenizer.pad_token_id)


        enc_params = list(self.model.encoder.parameters())
        dec_params = [p for p in self.model.decoder.parameters() if p.requires_grad] 
 
        if not self.finetune_encoder:
            for p in enc_params:
                p.requires_grad = False
            self.optimizer = optim.Adam([
                {"params": dec_params, "lr": self.learning_rate}
            ], weight_decay=1e-4)
            print("Encoder parameters are frozen.")

        else:
            print("Encoder parameters are trainable.")
            self.optimizer = optim.Adam([
                {"params": enc_params, "lr": self.learning_rate * 0.1},
                {"params": dec_params, "lr": self.learning_rate}
                ], weight_decay=1e-4)
         
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=1, verbose=True)
        self.writer = SummaryWriter(log_dir=os.path.join(self.checkpoint_path, "logs"))
        
        print("Model components initialized:")
        print(f"  Model: {self.model.__class__.__name__}")
        print(f"  Criterion: {self.criterion.__class__.__name__}")
        print(f"  Optimizer: {self.optimizer.__class__.__name__}")
        if self.scheduler:
            print(f"  Scheduler: {self.scheduler.__class__.__name__}")


        
    def save_checkpoint(self, epoch: int, val_loss: float, best_val_loss: float, filename: str):
        """Saves the model checkpoint to a specified file."""
        if self.model is None or self.optimizer is None:
            raise ValueError("Model or optimizer not initialized.")

        os.makedirs(self.checkpoint_path, exist_ok=True)

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss, # Current validation loss
            'best_val_loss': best_val_loss, # Best validation loss seen so far
            'model_class': self.model.__class__.__name__ # Store model class name
        }

        if self.scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.scaler is not None:
            checkpoint_data['scaler_state_dict'] = self.scaler.state_dict()

        full_path = os.path.join(self.checkpoint_path, filename)
        torch.save(checkpoint_data, full_path)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at {checkpoint_path}")
            return 0
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def _prepare_batch(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepare a batch of data for training or validation.
        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch of data.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Prepared input and target tensors.
        """
        images, captions, _ = batch
        images = images.to(self.device)
        captions = captions.to(self.device)

        input_caption = captions[:, :-1]
        target_caption = captions[:, 1:]
        
        padding_mask = (input_caption == self.tokenizer.tokenizer.pad_token_id)
        

        return images, input_caption, target_caption, padding_mask
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train the model for one epoch.
        Args:
            epoch (int): Current epoch number.
            Returns:
        float: Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Train]")
        
        for batch_idx, batch in enumerate(progress_bar):
            self.optimizer.zero_grad()
            
            images, input_caption, target_caption, padding_mask = self._prepare_batch(batch) 
            
            if self.use_mixed_precision and self.scaler is not None:
                with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                    outputs = self.model(images, input_caption, padding_mask = padding_mask)
                    loss = self.criterion(
                        outputs.reshape(-1, outputs.shape[-1]), 
                        target_caption.reshape(-1)
                    )
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, input_caption, padding_mask = padding_mask)    
                loss = self.criterion(
                    outputs.reshape(-1, outputs.shape[-1]), 
                    target_caption.reshape(-1)
                )
                
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
            if batch_idx % 100 == 0:
                self.writer.add_scalar('Loss/train_step', loss.item(), 
                                      epoch * len(self.train_loader) + batch_idx)
        
        avg_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        return avg_loss
    
    def validate(self, epoch: int) -> float:
        """
        Validate the model on the validation set.
        Args:
            epoch (int): Current epoch number.
        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        total_loss = 0
        progress_bar = tqdm(self.valid_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Val]")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                images, input_caption, target_caption, padding_mask = self._prepare_batch(batch)
                
                outputs = self.model(images, input_caption, padding_mask = padding_mask)
                loss = self.criterion(
                    outputs.reshape(-1, outputs.shape[-1]), 
                    target_caption.reshape(-1)
                )
                
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(self.valid_loader)
        self.writer.add_scalar('Loss/validation', avg_loss, epoch)
        return avg_loss
    
    def generate_sample_caption(self) -> str:
        """
        Generate a caption for a given image.
        Args:
            image (torch.Tensor): Input image tensor.
        Returns:
            str: Generated caption.
        """
        self.model.eval()

        with torch.no_grad():
            for i, (image, caption, _) in enumerate(self.test_loader):
                if i >= 10:
                    break

                image = image[0].unsqueeze(0).to(self.device)
                generated_caption, _ = self.model.caption_image_greedy(image)

                token_ids = caption[0].tolist()
                pad_token_id = self.tokenizer.tokenizer.pad_token_id
                valid_token_ids = [tid for tid in token_ids if tid != pad_token_id]
                # Decode the valid sequence
                true_caption = self.tokenizer.decode(valid_token_ids, skip_special_tokens=True)

                print(f"True caption: {true_caption}")
                print(f"Generated caption: {generated_caption}")


    def train(self, resume_from: Optional[str] = None) -> None:
        """
        Train the model.
        Args:
            resume_from (Optional[str]): Path to a checkpoint to resume training from.
        """
            
        if self.criterion is None or self.optimizer is None:
            self.initialize_model()
            
        start_epoch = 0
        if resume_from is not None:
            start_epoch = self.load_checkpoint(resume_from)
            
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(start_epoch, self.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()
                
            is_best = val_loss < best_val_loss
            if is_best:
                self.save_checkpoint(epoch, val_loss, best_val_loss, filename="best_model.pth")
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if epoch % self.save_freq == 0:
                self.save_checkpoint(epoch, val_loss, best_val_loss, filename = f"checkpoint_epoch_{epoch}.pth")

            self.generate_sample_caption()
                
            if patience_counter >= self.early_stopping:
                print(f"Early stopping triggered after {epoch} epochs")
                break
                
        self.writer.close()
        print("Training complete!")