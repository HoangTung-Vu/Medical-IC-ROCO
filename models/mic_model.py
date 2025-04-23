import torch
import torch.nn as nn
from transformers import CvtModel, AutoModel, AutoTokenizer
from typing import Tuple, Any, Optional
from models.base import BaseMD
from models.decoder import Decoder



class EncoderWrapper(nn.Module):
    """
    CvT_PubMedBERT model class.
    Combines a CvT encoder with a PubMedBERT decoder for image captioning tasks.
    """
    def __init__(self, projection : nn.Module):
        super(EncoderWrapper, self).__init__()
        self.cvt = CvtModel.from_pretrained("microsoft/cvt-13")
        self.proj = projection

        for p in self.cvt.parameters():
            p.requires_grad = True
        
    def forward(self, input_image : torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CvT_PubMedBERT model.
        Args:
            input_image (torch.Tensor): Input image tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (B, N, hidden_size).
        """
        output = self.cvt(input_image)
        feature_maps = output.last_hidden_state
        B, C, H, W = feature_maps.size()
        feature_maps = feature_maps.view(B, C, H*W)
        features = feature_maps.permute(0, 2, 1)  # shape: (B, N, C)

        encoded_seq = self.proj(features)
        return encoded_seq
    
class CvT_PubMedBERT(BaseMD):
    """
    CvT_PubMedBERT model class.
    Combines a CvT encoder with a PubMedBERT decoder for image captioning tasks.
    """
    def __init__(self, hidden_size : int = 512, num_layers : int = 2, num_heads : int = 4, drop_out : float = 0.2):
        super(CvT_PubMedBERT, self).__init__()
        projection = nn.Linear(384, hidden_size)
        self.encoder = EncoderWrapper(projection)
        self.decoder = Decoder(hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads, drop_out=drop_out)
        self.vocab_size = self.decoder.vocab_size
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
        
    def forward(self,
                input_image : torch.Tensor,
                target_seq : torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None
    ):
        """
        Forward pass of the CvT_PubMedBERT model.
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
 