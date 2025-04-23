import torch
import torch.nn as nn
from torchvision.models import resnet50
from typing import Tuple, Any, Optional
from models.base import BaseMD
from models.decoder import Decoder, GPTDecoder
from transformers import AutoTokenizer

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = resnet50(pretrained=False)
        encoder_layers = list(base_model.children())
        self.backbone = nn.Sequential(*encoder_layers[:9])
                        
    def forward(self, x):
        return self.backbone(x)

backbone = Backbone()
backbone.load_state_dict(torch.load("models/ResNet50.pt", weights_only=True))
print("Loaded ResNet50 backbone weights successfully.")


class EncoderWrapper(nn.Module):
    """
    cnn_PubMedBERT model class.
    Combines a cnn encoder with a PubMedBERT decoder for image captioning tasks.
    """
    def __init__(self, projection : nn.Module):
        super(EncoderWrapper, self).__init__()
        self.cnn = nn.Sequential(*list(backbone.backbone.children())[:-1]) 
        self.proj = projection

        for p in self.cnn.parameters():
            p.requires_grad = True
        
    def forward(self, input_image : torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the cnn_PubMedBERT model.
        Args:
            input_image (torch.Tensor): Input image tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (B, N, hidden_size).
        """
        feature_maps = self.cnn(input_image)
        B, C, H, W = feature_maps.size()
        feature_maps = feature_maps.view(B, C, H*W)
        features = feature_maps.permute(0, 2, 1)  # shape: (B, N, C)

        encoded_seq = self.proj(features)
        return encoded_seq


class CNN_GPT(BaseMD):
    def __init__(self, freeze_encoder: bool = True):
        super(CNN_GPT, self).__init__()
        projection = nn.Linear(2048, 768)
        self.encoder = EncoderWrapper(projection)
        
        if freeze_encoder:
            for param in self.encoder.cnn.parameters():
                param.requires_grad = False
        
        self.decoder = GPTDecoder()
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    def forward(self,
                input_image : torch.Tensor,
                target_seq : torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None
    ):
        """
        Forward pass of the CNN_GPT model.
        Args:
            input_image (torch.Tensor): Input image tensor of shape (B, C, H, W).
            target_seq (torch.Tensor): Target sequence tensor of shape (B, N).
            padding_mask (Optional[torch.Tensor]): Target padding mask. Shape (B, N).
        Returns:
            torch.Tensor: Output tensor of shape (B, N, vocab_size).
        """
        encoded_seq = self.encoder(input_image)
        decoded_seq = self.decoder(
            input_ids=target_seq,
            encoder_hidden_states=encoded_seq,
            attention_mask=padding_mask
        )
        return decoded_seq.logits

