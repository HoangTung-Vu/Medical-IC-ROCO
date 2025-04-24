import sys
import os
import json
import torch
import nltk

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure NLTK resources are downloaded
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    print("Warning: Could not download NLTK resources. METEOR score might not work properly.")

from models.mic_model import CvT_PubMedBERT
from utils.evaluator import Evaluator

# Create the model
model = CvT_PubMedBERT(num_layers=4, num_heads=8, hidden_size=768, drop_out=0.2)
print(f"Model created: {model.__class__.__name__}")

# Path to best checkpoint
checkpoint_path = "checkpoints/best_model.pth"

# Create evaluator
evaluator = Evaluator(
    model=model,
    dataroot="data/03471f547bb646a1f447add638d46bb3507523e8",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size=16,
    use_beam_search=False,
    results_dir="results2"
)

print(f"Running evaluation with {'beam search' if evaluator.use_beam_search else 'greedy search'}")
if evaluator.use_beam_search:
    print(f"Beam size: {evaluator.beam_size}")

results = evaluator.run_evaluation(checkpoint_path)

print("\nEvaluation complete.")
print("Summary of test results:")
for metric, value in results["test"].items():
    if metric != "dataset" and metric != "num_samples":
        print(f"  {metric}: {value:.4f}")