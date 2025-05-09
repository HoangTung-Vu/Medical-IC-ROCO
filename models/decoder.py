import torch
import torch.nn as nn
from models.modules import * 
from transformers import AutoModel
from typing import Optional

class Decoder(nn.Module):
    def __init__(self, hidden_size : int = 512, num_layers : int = 2, num_heads : int = 4, drop_out : float = 0.2):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        bert = AutoModel.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
        self.bert_embed = bert.embeddings.word_embeddings
        if bert.config.hidden_size == hidden_size : 
            self.bert_proj = None
        else :
            self.bert_proj = nn.Linear(bert.config.hidden_size, hidden_size)  # Project BERT output to match hidden_size
        self.vocab_size = bert.config.vocab_size

        self.pos_emb = SinusoidalPosEmb(hidden_size)    
        self.dropout = nn.Dropout(drop_out)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size=hidden_size, num_heads=num_heads, drop_out=drop_out, is_decoder=True)
            for _ in range(num_layers)
        ])  

        self.norm_out = nn.LayerNorm(hidden_size)
        
        for params in self.bert_embed.parameters() :
            params.requires_grad = False

    def forward(
            self, 
            target_seq : torch.Tensor,
            encoder_output : torch.Tensor,
            target_padding_mask : Optional[torch.Tensor] = None,
            encoder_padding_mask : Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the decoder.
        Args:
            target_seq (torch.Tensor): Target sequence tensor of shape (B, N).
            encoder_output (torch.Tensor): Encoder output tensor of shape (B, N, hidden_size).
            target_padding_mask (Optional[torch.Tensor]): Target padding mask. Shape (B, N).
            encoder_padding_mask (Optional[torch.Tensor]): Encoder padding mask. Shape (B, N).
        Return : 
            torch.Tensor: Output tensor of shape (B, N, vocab_size).
        """
        batch_size, seq_len = target_seq.size()
        # print(f"batch: {batch_size}")
        # print(f"seqlen : {seq_len}")
        device = target_seq.device

        # if target_padding_mask is not None:
        #     attention_mask = ~target_padding_mask
        # else:
        #     attention_mask = None
    
        target_emb = self.bert_embed(target_seq)
        if self.bert_proj is not None : 
            target_emb = self.bert_proj(target_emb)
        pos_encoding = self.pos_emb(seq_len, device=device)

        x = self.dropout(target_emb + pos_encoding) 
        cross_attn_weights_layers = []
        for block in self.blocks:
            x, cross_attn_weights = block(
                x=x,
                encoder_output=encoder_output,
                self_attn_padding_mask=target_padding_mask,
                cross_attn_padding_mask=encoder_padding_mask
            )
            cross_attn_weights_layers.append(cross_attn_weights)
        x = self.norm_out(x)
        if self.bert_proj is not None :
            x = torch.matmul(x, self.bert_proj.weight)           # (B, T, 768)
        logits = torch.matmul(x, self.bert_embed.weight.T) 
        return logits, cross_attn_weights_layers
    