import unittest
import torch
import os
import sys
from transformers import AutoTokenizer

# Add the project root directory to the Python path
# This allows importing modules from the 'models' directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Now import the model components
from models.mic_model import CvT_PubMedBERT

class TestCvTPubMedBERT(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up common resources for all tests."""
        print("\nSetting up TestCvTPubMedBERT...")
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {cls.device}")

        # Model parameters (match default if possible, or use specific values)
        cls.hidden_size = 512 # Smaller hidden size for faster testing
        cls.num_layers = 2
        cls.num_heads = 4
        cls.drop_out = 0.1

        # Instantiate the model
        # Wrap in try-except to catch potential Hugging Face download issues
        try:
            cls.model = CvT_PubMedBERT(
                hidden_size=cls.hidden_size,
                num_layers=cls.num_layers,
                num_heads=cls.num_heads,
                drop_out=cls.drop_out
            ).to(cls.device)
            cls.model.eval() # Set model to evaluation mode
        except Exception as e:
            print(f"Error initializing model or tokenizer: {e}")
            print("Please ensure you have an internet connection for Hugging Face downloads.")
            raise e # Re-raise the exception to fail the setup

        cls.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
        cls.vocab_size = cls.model.vocab_size

        # Standard input dimensions
        cls.batch_size = 2
        cls.img_size = 224 # Standard size for CvT
        cls.seq_len = 20

        # Expected number of patches from CvT-13 for 224x224 input
        # CvT-13 stage strides: [4, 2, 2]. Final feature map size: 224 / (4*2*2) = 14.
        cls.num_patches = 14 * 14 # 196

    def _create_dummy_data(self, batch_size, seq_len, include_padding=False):
        """Helper to create dummy input data."""
        image = torch.randn(batch_size, 3, self.img_size, self.img_size, device=self.device)
        # Create realistic target sequence with CLS token at start
        target_seq = torch.randint(low=100, high=self.vocab_size - 2, size=(batch_size, seq_len - 1), device=self.device)
        cls_tokens = torch.full((batch_size, 1), self.tokenizer.cls_token_id, device=self.device)
        target_seq = torch.cat([cls_tokens, target_seq], dim=1) # Add CLS token

        padding_mask = None
        if include_padding:
            # Mask the last 5 tokens for the first batch item, last 2 for the second
            padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device)
            if batch_size >= 1:
                padding_mask[0, -5:] = True
                target_seq[0, -5:] = self.tokenizer.pad_token_id # Set masked tokens to PAD ID
            if batch_size >= 2:
                 padding_mask[1, -2:] = True
                 target_seq[1, -2:] = self.tokenizer.pad_token_id # Set masked tokens to PAD ID
        print(f"Image shape: {image.shape}")
        print(f"Target sequence shape: {target_seq.shape}")
        print(f"Padding mask shape: {padding_mask.shape if padding_mask is not None else None}")
        print(f"Padding mask: {padding_mask}")
        print(f"Target sequence: {target_seq}")
        print(f"Target sequence (after padding): {target_seq}")

        return image, target_seq, padding_mask

    def test_model_initialization(self):
        """Test if the model and its components are initialized correctly."""
        self.assertIsInstance(self.model, CvT_PubMedBERT)
        self.assertIsNotNone(self.model.encoder)
        self.assertIsNotNone(self.model.decoder)
        self.assertEqual(self.model.vocab_size, self.tokenizer.vocab_size)
        # Check projection layer dimensions
        self.assertEqual(self.model.encoder.proj.in_features, 384) # CvT-13 output dim
        self.assertEqual(self.model.encoder.proj.out_features, self.hidden_size)
        # Check decoder output layer dimensions
        self.assertEqual(self.model.decoder.fc_out.out_features, self.vocab_size)
        self.assertEqual(self.model.decoder.fc_out.in_features, self.hidden_size)

    def test_forward_pass_shape_no_padding(self):
        """Test the forward pass output shapes without padding."""
        image, target_seq, _ = self._create_dummy_data(self.batch_size, self.seq_len, include_padding=False)

        with torch.no_grad():
            logits, attn_weights_layers = self.model(image, target_seq, padding_mask=None)
            print(f"Logits shape: {logits.shape}")
            print(f"Attention weights layers shape: {[layer.shape for layer in attn_weights_layers]}")

        # Check logits shape: (B, N, vocab_size)
        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, self.vocab_size))

        # Check attention weights list length (one per decoder layer)
        self.assertIsInstance(attn_weights_layers, list)
        self.assertEqual(len(attn_weights_layers), self.num_layers)

        # Check shape of attention weights from one layer (e.g., the last one)
        # Shape: (Batch, Heads, TargetSeqLen, SourceSeqLen=NumPatches)
        last_layer_attn = attn_weights_layers[-1]
        self.assertIsInstance(last_layer_attn, torch.Tensor)
        # Note: nn.MultiheadAttention returns (Batch, TargetSeqLen, SourceSeqLen) when batch_first=True
        # Let's re-check the AttentionBlock. It returns weights directly.
        # MHA with batch_first=True returns attn_output, attn_weights
        # attn_weights shape is (batch_size, target_seq_len, source_seq_len) if average_attn_weights is True (default)
        # If average_attn_weights is False, shape is (batch_size * num_heads, target_seq_len, source_seq_len) -- less likely used here
        # Let's assume default averaging or verify AttentionBlock. It seems average_attn_weights isn't explicitly set, defaults True.
        # Expected shape: (batch_size, target_seq_len, num_patches)
        self.assertEqual(last_layer_attn.shape, (self.batch_size, self.seq_len, self.num_patches))


    def test_forward_pass_shape_with_padding(self):
        """Test the forward pass output shapes with padding."""
        image, target_seq, padding_mask = self._create_dummy_data(self.batch_size, self.seq_len, include_padding=True)

        self.assertIsNotNone(padding_mask)
        self.assertEqual(padding_mask.shape, (self.batch_size, self.seq_len))
        self.assertTrue(padding_mask[0, -1]) # Check if padding is applied
        self.assertTrue(padding_mask[1, -1])
        self.assertEqual(target_seq[0, -1], self.tokenizer.pad_token_id) # Check PAD token
        self.assertEqual(target_seq[1, -1], self.tokenizer.pad_token_id)

        with torch.no_grad():
             # Pass the padding mask to the model's forward method
             # The model should internally pass it to the decoder's target_padding_mask
            logits, attn_weights_layers = self.model(image, target_seq, padding_mask=padding_mask)
            print(f"Logits shape: {logits.shape}")
            print(f"Attention weights layers shape: {[layer.shape for layer in attn_weights_layers]}")

        # Shapes should remain the same as without padding
        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, self.vocab_size))
        self.assertIsInstance(attn_weights_layers, list)
        self.assertEqual(len(attn_weights_layers), self.num_layers)
        last_layer_attn = attn_weights_layers[-1]
        self.assertIsInstance(last_layer_attn, torch.Tensor)
        self.assertEqual(last_layer_attn.shape, (self.batch_size, self.seq_len, self.num_patches))

        # Optional: Check if logits for padded positions are different (hard to check specific values)
        # We trust the underlying MHA and BERT embedding handle the mask correctly.

    def test_greedy_captioning(self):
        """Test greedy search caption generation."""
        batch_size = 1
        max_len = 15 # Keep it short for testing
        image, _, _ = self._create_dummy_data(batch_size, self.seq_len) # target_seq not needed here

        with torch.no_grad():
            caption_tokens, attention_maps = self.model.caption_image_greedy(image, max_len=max_len)
            print(f"Caption tokens: {caption_tokens}")
            print(f"Attention maps: {[att.shape for att in attention_maps]}")

        # Check output types
        self.assertIsInstance(caption_tokens, list)
        self.assertIsInstance(attention_maps, list)

        # Check that output contains strings (decoded tokens)
        if caption_tokens: # Only check if something was generated
             self.assertIsInstance(caption_tokens[0], str)

        # Check lengths match (attention map per generated token, excluding CLS/SEP/PAD)
        self.assertEqual(len(caption_tokens), len(attention_maps))

        # Check shape of individual attention maps (Num_Patches,)
        if attention_maps:
            self.assertIsInstance(attention_maps[0], torch.Tensor)
            self.assertEqual(attention_maps[0].shape, (self.num_patches,))
            self.assertEqual(attention_maps[0].device.type, image.device.type)

        # Check if generation stopped reasonably (either EOS or max_len)
        # Note: caption_tokens list already has special tokens filtered out by the base method
        # We can inspect the raw indices if needed by modifying the base method slightly for testing,
        # but for now, we check if the length is within limits.
        self.assertLessEqual(len(caption_tokens), max_len)


    def test_beam_search_captioning(self):
        """Test beam search caption generation."""
        batch_size = 1
        max_len = 15 # Keep it short for testing
        beam_size = 3
        image, _, _ = self._create_dummy_data(batch_size, self.seq_len) # target_seq not needed here

        with torch.no_grad():
            caption_str, attention_weights_list = self.model.caption_image_beam_search(
                image,
                beam_size=beam_size,
                max_len=max_len
            )

        # Check output types
        self.assertIsInstance(caption_str, str)
        self.assertIsInstance(attention_weights_list, list)

        # Check structure of attention weights list
        # It's a list (per step) of lists (per layer) of tensors
        if attention_weights_list:
             # Check the first step's attention list
             step_attentions = attention_weights_list[0]
             self.assertIsInstance(step_attentions, list)
             # Check number of layers
             self.assertEqual(len(step_attentions), self.num_layers)
             # Check the tensor from the last layer in the first step
             last_layer_first_step_attn = step_attentions[-1]
             self.assertIsInstance(last_layer_first_step_attn, torch.Tensor)
             # Shape: (batch=1, target_len_at_step, num_patches)
             # At step 0, target_len = 1 (only CLS); step 1, target_len = 2, etc.
             # Let's check the shape pattern loosely
             self.assertEqual(last_layer_first_step_attn.dim(), 3)
             self.assertEqual(last_layer_first_step_attn.size(0), 1) # Batch size
             self.assertGreaterEqual(last_layer_first_step_attn.size(1), 1) # Target sequence length at that step
             self.assertEqual(last_layer_first_step_attn.size(2), self.num_patches) # Source sequence length (patches)

        # Check if the generated caption is plausible (not empty, unless max_len is tiny)
        if max_len > 1:
            self.assertTrue(len(caption_str) > 0)


if __name__ == '__main__':
    unittest.main()