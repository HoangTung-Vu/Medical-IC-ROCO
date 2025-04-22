import sys
import os
import json
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.mic_model import CvT_PubMedBERT
from models.cnn_pubmedbert import CNN_PubMedBERT
from utils.dataloader import get_dataloader
from utils.trainer import Trainer

model = CNN_PubMedBERT(num_layers=4, num_heads=8, hidden_size=768, drop_out=0.2)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

decoder_params = sum(p.numel() for p in model.decoder.parameters())
trainable_decoder_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)

print(f"Total decoder parameters: {decoder_params:,}")
print(f"Trainable decoder parameters: {trainable_decoder_params:,}")

encoder_params = sum(p.numel() for p in model.encoder.parameters())
trainable_encoder_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)

print(f"Total encoder parameters: {encoder_params:,}")
print(f"Trainable encoder parameters: {trainable_encoder_params:,}")

trainer = Trainer(
    model = model,
    num_epochs = 30,
    dataroot = "data/03471f547bb646a1f447add638d46bb3507523e8",
    device = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size = 16,
    checkpoint_path="checkpoints_cnn/",
    finetune_encoder = False,
)

trainer.train()