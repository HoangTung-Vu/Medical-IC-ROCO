import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.dataloader import get_dataloader
import torchvision.transforms as transforms

class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        dataroot: str,
        batch_size: int = 16,
        num_workers: int = 4,
        use_beam_search: bool = True,
        beam_size: int = 5,
        device: Optional[torch.device] = None,
        results_dir: str = "results",
        eval_file_name = "eval.json"
    ):
        self.model = model
        self.dataroot = dataroot
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_beam_search = use_beam_search
        self.beam_size = beam_size
        self.results_dir = results_dir
        self.eval_file_name = eval_file_name
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # grayscale -> RGB
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        # Initialize dataloaders
        ds = get_dataloader(
            root_dir=dataroot,
            transform=self.transform,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True
        )
        
        _, self.valid_loader, self.test_loader = ds["data"]
        self.tokenizer = ds["tokenizer"]
        
        # Initialize BERT model for semantic similarity
        self.bert_model = AutoModel.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
        self.bert_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
        self.bert_model.to(self.device)
        self.bert_model.eval()
        
        # Ensure results directory exists
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize Rouge
        self.rouge = Rouge()
        
        # Smooth function for BLEU
        self.smooth = SmoothingFunction().method1
    
    def _get_bert_embedding(self, text: str) -> np.ndarray:
        """Get BERT embeddings for a text."""
        with torch.no_grad():
            inputs = self.bert_tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            outputs = self.bert_model(**inputs)
            # Use CLS token as sentence embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
        return embedding
    
    def _compute_pubmedbert_score(self, hypothesis: str, references: List[str]) -> float:
        """Compute semantic similarity using PubMedBERT embeddings."""
        hyp_embedding = self._get_bert_embedding(hypothesis)
        
        max_score = 0.0
        for ref in references:
            ref_embedding = self._get_bert_embedding(ref)
            similarity = cosine_similarity(hyp_embedding, ref_embedding)[0][0]
            max_score = max(max_score, similarity)
            
        return max_score
    
    def _compute_meteor_score(self, hypothesis: str, references: List[str]) -> float:
        """Compute METEOR score."""
        return meteor_score(references=[ref.split() for ref in references], 
                          hypothesis=hypothesis.split())
    
    def _compute_rouge_scores(self, hypothesis: str, references: List[str]) -> Dict[str, float]:
        """Compute ROUGE scores."""
        max_rouge = {"rouge-1": {"f": 0}, "rouge-2": {"f": 0}, "rouge-l": {"f": 0}}
        
        for ref in references:
            try:
                scores = self.rouge.get_scores(hypothesis, ref)[0]
                
                for metric in max_rouge:
                    if scores[metric]["f"] > max_rouge[metric]["f"]:
                        max_rouge[metric]["f"] = scores[metric]["f"]
            except Exception:
                # In case of issues with ROUGE calculation
                continue
                
        return {
            "rouge-1": max_rouge["rouge-1"]["f"],
            "rouge-2": max_rouge["rouge-2"]["f"],
            "rouge-l": max_rouge["rouge-l"]["f"]
        }
    
    def _compute_bleu_score(self, hypothesis: List[str], references: List[List[List[str]]]) -> Dict[str, float]:
        """Compute BLEU scores."""
        bleu_1 = corpus_bleu(references, hypothesis, weights=(1, 0, 0, 0), smoothing_function=self.smooth)
        bleu_2 = corpus_bleu(references, hypothesis, weights=(0.5, 0.5, 0, 0), smoothing_function=self.smooth)
        bleu_3 = corpus_bleu(references, hypothesis, weights=(0.33, 0.33, 0.33, 0), smoothing_function=self.smooth)
        bleu_4 = corpus_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smooth)
        
        return {
            "bleu-1": bleu_1,
            "bleu-2": bleu_2,
            "bleu-3": bleu_3,
            "bleu-4": bleu_4
        }
    
    def _prepare_batch(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare a batch of data for evaluation."""
        images, captions, _ = batch
        images = images.to(self.device)
        
        return images, captions
    
    def _unnormalize_image(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Unnormalize an image tensor for display."""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_tensor.device)
        
        img = img_tensor.clone()
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        
        return img
    
    def evaluate(self, dataloader: DataLoader, dataset_name: str = "test") -> Dict[str, Any]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataloader (DataLoader): Data loader for evaluation.
            dataset_name (str): Name of the dataset for logging.
            
        Returns:
            Dict[str, Any]: Evaluation metrics.
        """
        self.model.eval()
        
        all_references = []
        all_references_tokenized = []
        all_hypotheses = []
        all_hypotheses_tokenized = []
        
        pubmedbert_scores = []
        
        progress_bar = tqdm(dataloader, desc=f"Evaluating on {dataset_name}")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                images, captions = self._prepare_batch(batch)
                
                for i in range(images.size(0)):
                    image = images[i].unsqueeze(0)
                    
                    if self.use_beam_search:
                        generated_caption, _ = self.model.caption_image_beam_search(
                            image=image, 
                            beam_size=self.beam_size
                        )
                    else:
                        generated_caption, _ = self.model.caption_image_greedy(image=image)
                    
                    # Process reference caption
                    token_ids = captions[i].tolist()
                    pad_token_id = self.tokenizer.get_pad_token_id()
                    valid_token_ids = [tid for tid in token_ids if tid != pad_token_id]
                    reference_caption = self.tokenizer.decode(valid_token_ids, skip_special_tokens=True)
                    
                    # For BLEU score
                    reference_tokenized = reference_caption.lower().split()
                    hypothesis_tokenized = generated_caption.lower().split()
                    
                    all_references.append([reference_caption])  # Wrapped in list for multi-reference format
                    all_references_tokenized.append([[reference_tokenized]])
                    all_hypotheses.append(generated_caption)
                    all_hypotheses_tokenized.append(hypothesis_tokenized)
                    
                    # Calculate PubMedBERT score
                    pubmedbert_score = self._compute_pubmedbert_score(
                        hypothesis=generated_caption,
                        references=[reference_caption]
                    )
                    pubmedbert_scores.append(pubmedbert_score)
        
        # Calculate BLEU scores
        bleu_scores = self._compute_bleu_score(
            hypothesis=all_hypotheses_tokenized,
            references=all_references_tokenized
        )
        
        # Calculate METEOR and ROUGE scores
        meteor_scores = []
        rouge_1_scores = []
        rouge_2_scores = []
        rouge_l_scores = []
        
        for hyp, refs in zip(all_hypotheses, all_references):
            # METEOR
            meteor = self._compute_meteor_score(hyp, refs)
            meteor_scores.append(meteor)
            
            # ROUGE
            rouge_scores = self._compute_rouge_scores(hyp, refs)
            rouge_1_scores.append(rouge_scores["rouge-1"])
            rouge_2_scores.append(rouge_scores["rouge-2"])
            rouge_l_scores.append(rouge_scores["rouge-l"])
        
        # Aggregate results
        results = {
            "dataset": dataset_name,
            "num_samples": len(all_hypotheses),
            "bleu-1": bleu_scores["bleu-1"],
            "bleu-2": bleu_scores["bleu-2"],
            "bleu-3": bleu_scores["bleu-3"],
            "bleu-4": bleu_scores["bleu-4"],
            "meteor": sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0,
            "rouge-1": sum(rouge_1_scores) / len(rouge_1_scores) if rouge_1_scores else 0,
            "rouge-2": sum(rouge_2_scores) / len(rouge_2_scores) if rouge_2_scores else 0,
            "rouge-l": sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0,
            "pubmedbert": sum(pubmedbert_scores) / len(pubmedbert_scores) if pubmedbert_scores else 0
        }
        
        print(f"Evaluation results on {dataset_name} dataset:")
        for metric, value in results.items():
            if metric != "dataset" and metric != "num_samples":
                print(f"  {metric}: {value:.4f}")
        
        return results
    
    def visualize_samples(self, dataloader: DataLoader, num_samples: int = 10) -> None:
        """
        Visualize sample predictions from the model.
        
        Args:
            dataloader (DataLoader): Data loader for samples.
            num_samples (int): Number of samples to visualize.
        """
        self.model.eval()
        samples_seen = 0
        
        plt.figure(figsize=(15, 15))
        
        with torch.no_grad():
            for batch in dataloader:
                images, captions = self._prepare_batch(batch)
                
                for i in range(images.size(0)):
                    if samples_seen >= num_samples:
                        break
                        
                    image = images[i].unsqueeze(0)
                    
                    # Generate caption
                    if self.use_beam_search:
                        generated_caption, _ = self.model.caption_image_beam_search(
                            image=image, 
                            beam_size=self.beam_size
                        )
                    else:
                        generated_caption, _ = self.model.caption_image_greedy(image=image)
                    
                    # Get reference caption
                    token_ids = captions[i].tolist()
                    pad_token_id = self.tokenizer.get_pad_token_id()
                    valid_token_ids = [tid for tid in token_ids if tid != pad_token_id]
                    reference_caption = self.tokenizer.decode(valid_token_ids, skip_special_tokens=True)
                    
                    # Display image and captions
                    img_display = self._unnormalize_image(images[i].cpu())
                    
                    plt.subplot(5, 2, samples_seen + 1)
                    plt.imshow(img_display.permute(1, 2, 0).numpy())
                    plt.title(f"Sample {samples_seen + 1}", fontsize=12)
                    plt.axis("off")
                    
                    print(f"\nSample {samples_seen + 1}:")
                    print(f"Reference: {reference_caption}")
                    print(f"Generated: {generated_caption}")
                    print("-" * 80)
                    
                    samples_seen += 1
                    
                if samples_seen >= num_samples:
                    break
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "sample_predictions.png"))
        plt.show()
    
    def run_evaluation(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete evaluation process.
        
        Args:
            checkpoint_path (Optional[str]): Path to model checkpoint.
            
        Returns:
            Dict[str, Any]: Combined evaluation results.
        """
        # Load checkpoint if provided
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model checkpoint from epoch {checkpoint['epoch']}")
        
        self.model.to(self.device)
        
        # Evaluate on validation and test sets
        print("Evaluating model performance...")
        valid_results = self.evaluate(self.valid_loader, "validation")
        test_results = self.evaluate(self.test_loader, "test")
        
        # Visualize samples
        print("\nGenerating sample visualizations...")
        self.visualize_samples(self.test_loader)
        
        # Save results
        results = {
            "validation": valid_results,
            "test": test_results
        }
        
        results_path = os.path.join(self.results_dir, self.eval_file_name)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Results saved to {results_path}")
        
        return results