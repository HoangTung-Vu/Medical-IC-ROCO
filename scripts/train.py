import sys
import os
import json
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.mic_model import CvT_PubMedBERT
from utils.dataloader import get_dataloader
from utils.trainer import Trainer

model = CvT_PubMedBERT()

trainer = Trainer(
    model = model,
    num_epochs = 30,
    dataroot = "data/mdwiratathya___roco-radiology/default/0.0.0/03471f547bb646a1f447add638d46bb3507523e8",
    device = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size = 8
)

trainer.train()