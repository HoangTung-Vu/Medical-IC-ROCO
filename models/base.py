
""" THIS IS NOT USED IN THE PROJECT
    This class is defined for Decoder from scratch. (from scratch vocab).
    This class provides common inference methods like greedy search and beam search.
    Subclasses are expected to implement:
    - self.encoder: A module that takes the input (e.g., image) and returns encoded features.
    - self.decoder: A module that takes the target sequence and encoder output and returns logits.
    - Necessary vocabulary information (e.g., SOS, EOS indices) should be handled by the vocabulary object
      passed to the inference methods. """

import torch
import torch.nn as nn
import math
from typing import Any, List, Optional, Tuple
from transformers import AutoTokenizer

class BaseMD(nn.Module):
    """
    Base model class providing common inference methods like greedy search and beam search
    for sequence generation tasks (e.g., image captioning).

    Subclasses are expected to implement:
    - self.encoder: A module that takes the input (e.g., image) and returns encoded features.
    - self.decoder: A module that takes the target sequence and encoder output and returns logits.
    - Necessary vocabulary information (e.g., SOS, EOS indices) should be handled by the vocabulary object
      passed to the inference methods.
    """
    def __init__(self):
        super().__init__()
   
    @torch.no_grad()
    def caption_image_greedy(self,
                             image : torch.Tensor,
                             max_len : int = 500
    ):
        tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
        assert image.shape[0] == 1, "Greedy search requires batch size 1"
        self.eval()

        device = image.device
        image = image.to(device)
        
        start_token_id = tokenizer.cls_token_id

        encoded_seq = self.encoder(image)
        current_token = torch.tensor([[start_token_id]], dtype=torch.long, device=device)

        attention_maps = []
        caption_indices = []

        for _ in range(max_len):
            logits, attn_layers = self.decoder(
                target_seq=current_token,
                encoder_output=encoded_seq,
                target_padding_mask=None,
                encoder_padding_mask=None
            )

            next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            current_token = torch.cat([current_token, next_token], dim=1)

            last_layer_attn = attn_layers[-1]  # (1, seq_len, num_patches)
            attention_vector = last_layer_attn[0, -1]  # shape: (num_patches,)
            attention_maps.append(attention_vector)

            caption_indices.append(next_token.item())

            if next_token.item() == tokenizer.sep_token_id:
                break

                            
        final_caption = tokenizer.decode(caption_indices, skip_special_tokens=True)
        filtered_attention = [attn for idx, attn in zip(caption_indices, attention_maps) if idx not in (tokenizer.pad_token_id, tokenizer.sep_token_id)]
        
        return final_caption, filtered_attention # Return the string caption

    @torch.no_grad()
    def caption_image_beam_search(self, image, beam_size=5, max_len=50, alpha=0.7):
        """
        Generate caption using beam search decoding.
        
        Args:
            image (torch.Tensor): Input image tensor of shape (1, C, H, W)
            beam_size (int): Beam size
            max_len (int): Maximum length of generated caption
            alpha (float): Length normalization parameter
            
        Returns:
            str: Best generated caption
            list: Attention weights for the best caption
        """
        tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
        device = image.device
        self.eval()

        if image.size(0) != 1:
            raise ValueError("Beam search only supports batch size 1")
        
        encoded_seq = self.encoder(image)  # (1, N, hidden_size)
        start_token_id = tokenizer.cls_token_id
        end_token_id = tokenizer.sep_token_id
        
        # Initialize beam
        # Each beam entry: (log_prob, token_ids, finished, cross_attn_weights)
        beams = [(0.0, torch.tensor([[start_token_id]], device=device), False, [])]
        
        # Generate tokens iteratively
        for step in range(max_len - 1):
            next_beams = []
            
            for beam_idx, (log_prob, token_ids, finished, prev_attn) in enumerate(beams):
                if finished:
                    if len(next_beams) < beam_size:
                        next_beams.append((log_prob, token_ids, finished, prev_attn))
                    continue
                
                logits, cross_attn_weights = self.decoder(
                    target_seq=token_ids,
                    encoder_output=encoded_seq,
                    target_padding_mask=None,
                    encoder_padding_mask=None
                )
                
                next_token_logits = logits[0, -1, :]  # (vocab_size)
                next_token_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
                
                topk_probs, topk_indices = next_token_probs.topk(beam_size)
                
                for prob, idx in zip(topk_probs, topk_indices):
                    new_token_ids = torch.cat([token_ids, idx.unsqueeze(0).unsqueeze(0)], dim=1)
                    new_log_prob = log_prob + prob.item()
                    new_finished = (idx.item() == end_token_id)
                    
                    if new_finished:
                        seq_len = new_token_ids.size(1)
                        normalized_score = new_log_prob / (seq_len ** alpha)
                    else:
                        normalized_score = new_log_prob
                    
                    new_attn = prev_attn + [cross_attn_weights]
                    
                    next_beams.append((normalized_score, new_token_ids, new_finished, new_attn))
            
            beams = sorted(next_beams, key=lambda x: x[0], reverse=True)[:beam_size]
            
            if all(finished for _, _, finished, _ in beams):
                break
        
        best_beam = max(beams, key=lambda x: x[0])
        best_score, best_tokens, _, best_attn = best_beam
        caption = tokenizer.decode(best_tokens[0].tolist(), skip_special_tokens=True)
        
        return caption, best_attn      

                



