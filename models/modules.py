import torch
import torch.nn as nn
import math
from typing import Any, List, Optional, Tuple  

def extract_patches(image_tensor : torch.Tensor, patch_size : int = 16) -> torch.Tensor : 
    """
    Extract patches from an image tensor.
    Args:
        image_tensor (torch.Tensor): Input image tensor of shape (B, C, H, W).
        patch_size (int): Size of the patches to extract.
    Returns:
        torch.Tensor: Extracted patches of shape (B, num_patches, patch_dim).
    """
    bs, c, h, w = image_tensor.size()
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    num_patches = num_patches_h * num_patches_w
    patch_dim = c * patch_size * patch_size

    patches = image_tensor.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(bs, num_patches, c, patch_size, patch_size)
    patches = patches.view (bs, num_patches, patch_dim)
    return patches

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim
    
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate sinusoidal positional embeddings.
        Args:
            seq_len (int): Length of the sequence.
            device (torch.device): Device to create the tensor on.
        Returns:
            torch.Tensor: Sinusoidal positional embeddings of shape (1, seq_len, dim).
        """
        half_dim = self.dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        inv_freq = torch.exp(torch.arange(half_dim, device=device) * -emb_factor)  # shape (half_dim,)
        pos = torch.arange(seq_len, device=device).float()  # shape (seq_len,)
        # Compute outer product: (seq_len, half_dim)
        sinusoid_inp = pos.unsqueeze(1) * inv_freq.unsqueeze(0)  # shape (seq_len, half_dim)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)  # shape (seq_len, dim)
        return emb.unsqueeze(0)  # shape (1, seq_len, dim)

    
class AttentionBlock(nn.Module):
    def __init__(self, hidden_size : int = 256, num_heads :  int = 4, drop_out : int = 0.1, is_causal : bool = False, return_weights : bool = False):
        super(AttentionBlock, self).__init__()
        self.is_causal = is_causal
        self.return_weights = return_weights
        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True, dropout=drop_out)

    def forward(
            self, 
            query : torch.Tensor, 
            key : torch.Tensor, 
            value : torch.Tensor, 
            key_padding_mask : Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the attention block.
        Args:
            query (torch.Tensor): Query tensor of shape (B, N, C).
            key (torch.Tensor): Key tensor of shape (B, N, C).
            value (torch.Tensor): Value tensor of shape (B, N, C).
            key_padding_mask (Optional[torch.Tensor]): Key padding mask.
        Output : 
            Attention output tensor of shape (B, N, C).
        """
        target_seq_len = query.size(1)
        attn_mask = None

        if self.is_causal:
            attn_mask = torch.triu(torch.ones(target_seq_len, target_seq_len, device=query.device, dtype=torch.bool), diagonal=1).to(query.device)
        
        if self.return_weights :
            attn_output, attn_weights = self.mha(
                query=query, 
                key=key, 
                value=value,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                attn_mask=attn_mask
            )   
            return attn_output, attn_weights
        else :
            attn_output, _ = self.mha(
                query=query, 
                key=key, 
                value=value,
                key_padding_mask=key_padding_mask,
                need_weights=False,
                attn_mask=attn_mask
            )
            return attn_output

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size : int = 256, num_heads : int = 4, drop_out : float = 0.1, is_decoder : bool = False):
        super(TransformerBlock, self).__init__()
        self.is_decoder = is_decoder

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        if self.is_decoder : 
            self.norm3 = nn.LayerNorm(hidden_size)

        self.self_attn = AttentionBlock(hidden_size=hidden_size, num_heads=num_heads, drop_out=drop_out, is_causal=self.is_decoder, return_weights=False)
        if self.is_decoder : 
            self.cross_attn = AttentionBlock(hidden_size=hidden_size, num_heads=num_heads, drop_out=drop_out, is_causal=False, return_weights=True)
        
        self.ffn = nn.Sequential( 
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.dropout = nn.Dropout(drop_out)

    def forward(
            self, 
            x : torch.Tensor, 
            encoder_output: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,    
            cross_attn_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the transformer block.
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).
            encoder_output (Optional[torch.Tensor]): Encoder output tensor of shape (B, N, C).
            self_attn_padding_mask (Optional[torch.Tensor]): Self-attention padding mask. shape (B, N)
            cross_attn_padding_mask (Optional[torch.Tensor]): Cross-attention padding mask. shape (B, N)
            Returns:
                torch.Tensor: Output tensor of shape (B, N, C).
        """
        residual = x
        x_norm = self.norm1(x)
        self_attn_output = self.self_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            key_padding_mask=self_attn_padding_mask
        )
        x = residual + self.dropout(self_attn_output)

        if self.is_decoder : 
            if encoder_output is None:
                raise ValueError("encoder_output must be provided for decoder blocks.")
            residual = x    
            x_norm = self.norm2(x)
            cross_attn_output, attn_weights = self.cross_attn(
                query=x_norm,
                key=encoder_output,
                value=encoder_output,
                key_padding_mask=cross_attn_padding_mask
            )
            x = residual + self.dropout(cross_attn_output)
            norm_layer_ffn = self.norm3 
        else : 
            norm_layer_ffn = self.norm2
        
        residual = x
        x_norm  = norm_layer_ffn(x)
        ffn_output = self.ffn(x_norm)
        x = residual + self.dropout(ffn_output)
        if self.is_decoder : 
            return x, attn_weights
        else :
            return x

