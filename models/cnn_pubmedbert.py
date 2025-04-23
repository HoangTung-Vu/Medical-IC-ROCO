import torch
import torch.nn as nn
from torchvision.models import resnet50
from typing import Tuple, Any, Optional
from models.base import BaseMD
from models.decoder import Decoder
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
    
class CNN_PubMedBERT(BaseMD):
    """
    cnn_PubMedBERT model class.
    Combines a cnn encoder with a PubMedBERT decoder for image captioning tasks.
    """
    def __init__(self, hidden_size : int = 512, num_layers : int = 2, num_heads : int = 4, drop_out : float = 0.2):
        super(CNN_PubMedBERT, self).__init__()
        projection = nn.Linear(2048, hidden_size)
        self.encoder = EncoderWrapper(projection)
        self.decoder = Decoder(hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads, drop_out=drop_out)
        self.vocab_size = self.decoder.vocab_size
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    def forward(self,
                input_image : torch.Tensor,
                target_seq : torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None
    ):
        """
        Forward pass of the cnn_PubMedBERT model.
        Args:
            input_image (torch.Tensor): Input image tensor of shape (B, C, H, W).
            target_seq (torch.Tensor): Target sequence tensor of shape (B, N).
            padding_mask (Optional[torch.Tensor]): Target padding mask. Shape (B, N).
        Returns:
            torch.Tensor: Output tensor of shape (B, N, vocab_size).
        """
        encoded_seq = self.encoder(input_image)
        decoded_seq, _ = self.decoder(
            target_seq=target_seq,
            encoder_output=encoded_seq,
            target_padding_mask=padding_mask,
            encoder_padding_mask = None
        )

        return decoded_seq
 