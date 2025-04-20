from transformers import AutoModel, AutoTokenizer
import torch

pubmedbert = AutoModel.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")

embedding = pubmedbert.embeddings.word_embeddings

sample_text = "	Posteroanterior radiography immediately before reintervention showing the inferior vena caval loop of the pacemaker lead strongly attached to the endothelium. The tip of the electrode is still attached the right ventricular wall. At that time an exit block was predominantly existent."
tokens = tokenizer(sample_text, return_tensors="pt")

input_ids = tokens["input_ids"]

# Get corresponding embeddings
with torch.no_grad():
    embedded_tokens = embedding(input_ids)

print(f"Vocabsize : {pubmedbert.config.vocab_size}")
print(f"Word embedding size : {pubmedbert.config.hidden_size}")

decoded_words = tokenizer.decode(input_ids[0], skip_special_tokens=False)
print("Input IDs:", input_ids)
print("Decoded words:", decoded_words)
print("Embedding shape:", embedded_tokens.shape)

total_params = sum(p.numel() for p in embedding.parameters())
trainable_params = sum(p.numel() for p in embedding.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")