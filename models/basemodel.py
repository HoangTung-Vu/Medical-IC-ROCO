
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

class BaseModel(nn.Module):
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
        # Subclasses should define self.encoder and self.decoder in their __init__

    @torch.no_grad()
    def caption_image_greedy(self, 
                             image : torch.Tensor,
                             vocabulary : Any,
                             max_length : int = 50) -> List[str]:
        """
        Generate a caption from the image using greedy search.
        Args:
            image: Input image tensor [1, channels, height, width]
            vocabulary: Vocabulary object (needs stoi, itos)
            max_length: Maximum length of the generated caption
        Returns:
            List of tokens (words) in the generated caption
        """

        assert image.shape[0] == 1, "Greedy search requires batch size 1"
        self.eval()
        device = image.device
        sos_idx = vocabulary.stoi["<SOS>"]
        eos_idx = vocabulary.stoi["<EOS>"]

        encoder_output = self.encoder(image)
        current_token = torch.tensor([[sos_idx]], dtype = torch.long, device=device)

        attention_maps = []
        caption_indices = []

        for _ in range(max_length):
            logits, attn_layers = self.decoder(
                target_seq=current_token,
                encoder_output=encoder_output,
                target_padding_mask=None,
                encoder_padding_mask=None
            )
            next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            current_token = torch.cat([current_token, next_token], dim=1)

            last_layer_attn = attn_layers[-1]  # (1, seq_len, num_patches)
            attention_vector = last_layer_attn[0, -1]  # shape: (num_patches,)

            # grid_size = int(math.sqrt(attention_vector.size(0)))
            # heatmap = attention_vector.view(grid_size, grid_size).detach().cpu().numpy()
            attention_maps.append(attention_vector)

            caption_indices.append(next_token.item())

            if next_token.item() == eos_idx:
                break
                
        caption = []
        filtered_attention = []
        for idx, attn in zip(caption_indices, attention_maps):
            if idx not in (vocabulary.stoi["<PAD>"], vocabulary.stoi["<EOS>"]):
                caption.append(vocabulary.itos[idx])
                filtered_attention.append(attn)

        return caption, filtered_attention

    
    @torch.no_grad()
    def caption_image_beam_search(self, 
                                  image: torch.Tensor,
                                  vocabulary: Any,
                                  max_length: int = 50,
                                  beam_size: int = 3) -> List[str]:
        """
        Generate a caption from the image using beam search.
        Args:
            image: Input image tensor [1, channels, height, width]
            vocabulary: Vocabulary object (needs stoi, itos)
            max_length: Maximum length of the generated caption
            beam_size: Number of beams to keep during search
        Returns:
            List of tokens (words) in the best generated caption
        """

        assert image.shape[0] == 1, "Beam search requires batch size 1"
        self.eval()
        device = image.device
        sos_idx = vocabulary.stoi["<SOS>"]
        eos_idx = vocabulary.stoi["<EOS>"]

        encoder_output = self.encoder(image)  # (1, num_patches, hidden_size)
        encoder_output = encoder_output.expand(beam_size, -1, -1)  # Duplicate encoder output for beam size

        sequences = [[sos_idx]] * beam_size
        scores = torch.zeros(beam_size, device=device)

        for _ in range(max_length):
            seqs_tensor = torch.tensor(sequences, dtype=torch.long, device=device)
            logits, _ = self.decoder(
                target_seq=seqs_tensor,
                encoder_output=encoder_output,
                target_padding_mask=None,
                encoder_padding_mask=None
            )  # (beam_size, seq_len, vocab_size)

            logits = logits[:, -1, :]  # Get last token logits: (beam_size, vocab_size)
            log_probs = torch.log_softmax(logits, dim=-1)  # Convert to log-probs
            total_scores = scores.unsqueeze(1) + log_probs  # (beam_size, vocab_size)

            flat_scores = total_scores.view(-1)  # (beam_size * vocab_size)
            top_scores, top_indices = flat_scores.topk(beam_size, dim=0)

            new_sequences = []
            new_scores = []

            for i in range(beam_size):
                beam_id = top_indices[i] // len(vocabulary)
                token_id = top_indices[i] % len(vocabulary)
                new_seq = sequences[beam_id] + [token_id.item()]
                new_sequences.append(new_seq)
                new_scores.append(top_scores[i])

            sequences = new_sequences
            scores = torch.stack(new_scores)

            # If all sequences have ended with <EOS>, stop early
            if all(seq[-1] == eos_idx for seq in sequences):
                break

        # Choose the sequence with the highest score
        best_seq = sequences[scores.argmax().item()]
        caption = [
            vocabulary.itos[idx] for idx in best_seq if idx not in (vocabulary.stoi["<PAD>"], vocabulary.stoi["<SOS>"], vocabulary.stoi["<EOS>"])
        ]
        return caption


