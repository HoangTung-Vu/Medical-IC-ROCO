import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
from PIL import Image


from utils.dataloader import get_dataloader
from utils.explainer import *
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

        self.bert_model = AutoModel.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
        self.bert_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
        self.bert_model.to(self.device)
        self.bert_model.eval()

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
    
    def evaluate(self, dataloader: DataLoader, dataset_name: str = "test", save_predictions: bool = True,) -> Dict[str, Any]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataloader (DataLoader): Data loader for evaluation.
            dataset_name (str): Name of the dataset for logging.
            
        Returns:
            Dict[str, Any]: Evaluation metrics.
        """
        predictions_file = os.path.join(self.results_dir, f"{dataset_name}_captions.json")
        if os.path.exists(predictions_file):
            with open(predictions_file, "r") as f:
                predictions_data = json.load(f)
            print(f"Loaded existing predictions from {predictions_file}")
        else:
            predictions_data = []

            self.model.eval()
            
            progress_bar = tqdm(dataloader, desc=f"Evaluating on {dataset_name}")
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(progress_bar):
                    # Limit the number of batches for testing
                    # if batch_idx == 3:
                    #     break
                    images, captions = self._prepare_batch(batch)
                    
                    for i in range(images.size(0)):
                        image = images[i].unsqueeze(0)
                        idx = batch_idx * self.batch_size + i    
                        
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
                        
                        
                        # Calculate PubMedBERT score
                        pubmedbert_score = self._compute_pubmedbert_score(
                            hypothesis=generated_caption,
                            references=[reference_caption]
                        )
                        predictions_data.append({
                            "image_id": idx,
                            "reference": reference_caption,
                            "hypothesis": generated_caption,
                            "reference_tokenized" : reference_tokenized,    
                            "hypothesis_tokenized": hypothesis_tokenized,
                            "pubmedbert_score": str(pubmedbert_score)
                        })

                if save_predictions:
                    with open(predictions_file, "w") as f:
                        json.dump(predictions_data, f, indent=6)
                
        all_references = [ [entry["reference"]] for entry in predictions_data ]
        all_references_tokenized = [ [entry["reference_tokenized"]] for entry in predictions_data ]
        all_hypotheses = [ entry["hypothesis"] for entry in predictions_data ]
        all_hypotheses_tokenized = [ entry["hypothesis_tokenized"] for entry in predictions_data ]
        pubmedbert_scores = [ float(entry["pubmedbert_score"]) for entry in predictions_data ]

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
        vis_dir = os.path.join(self.results_dir, 'visualize')
        os.makedirs(vis_dir, exist_ok=True)
        records = []  # to store mapping for json

        self.model.eval()
        seen = 0
        with torch.no_grad():
            for batch_id, batch in enumerate(dataloader):
                images, captions = self._prepare_batch(batch)
                for i in range(images.size(0)):
                    if seen >= num_samples:
                        break
                    image_tensor = images[i]
                    img = image_tensor.unsqueeze(0)
                    if self.use_beam_search:
                        hyp, _ = self.model.caption_image_beam_search(img, beam_size=self.beam_size)
                    else:
                        hyp, _ = self.model.caption_image_greedy(img)
                    toks = captions[i].tolist()
                    pad_id = self.tokenizer.get_pad_token_id()
                    valid = [t for t in toks if t != pad_id]
                    ref = self.tokenizer.decode(valid, skip_special_tokens=True)

                    unnorm = image_tensor.cpu() * torch.tensor([0.229,0.224,0.225]).view(3,1,1) + torch.tensor([0.485,0.456,0.406]).view(3,1,1)
                    img_arr = (unnorm.clamp(0,1).permute(1,2,0).numpy() * 255).astype(np.uint8)
                    
                    image = Image.fromarray(img_arr)
                    fname = f"sample_{seen:03d}_id_{batch_id}.png"
                    image.save(os.path.join(vis_dir, fname))

                    records.append({
                        "image_file": fname,
                        "image_id": f"{batch_id}-{i}",
                        "reference": ref,
                        "hypothesis": hyp
                    })
                    seen += 1
                if seen >= num_samples:
                    break

        # save JSON mapping
        with open(os.path.join(vis_dir, 'visualize_captions.json'), 'w') as f:
            json.dump(records, f, indent=2)
        print(f"Saved {seen} visualizations under {vis_dir}")
    
    def visualize_attention(self, dataloader: DataLoader, num_samples: int = 10) -> None:
        """
        Generate and save per-word attention visualizations for a few samples.

        Saves each word-level overlay into:
            results/visualize_attention/<image_id>/<word_index>_<word>.png
        """
        base_dir = os.path.join(self.results_dir, 'visualize_attention')
        os.makedirs(base_dir, exist_ok=True)
        seen = 0
        self.model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                images, _ = self._prepare_batch(batch)
                for i in range(images.size(0)):
                    if seen >= num_samples:
                        return
                    
                    image_tensor = images[i]
                    img = image_tensor.unsqueeze(0)
                    image_id = f"{batch_idx}-{i}"

                    words, heatmaps = explain_inference_image(img, self.model, device=self.device)
                    img_folder = os.path.join(base_dir, image_id)

                    for idx, (word, hm) in enumerate(zip(words, heatmaps)):
                        out_path = os.path.join(img_folder, f"{idx:02d}_{word}.png")
                        overlay = overlay_heatmap(image_tensor, hm, alpha = 0.2)
                        save_image(overlay, out_path)
                    seen += 1
        print(f"Saved {seen} attention visualizations under {base_dir}")        

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
        
        print("Evaluating model performance...")
        valid_results = self.evaluate(self.valid_loader, "validation")
        test_results = self.evaluate(self.test_loader, "test")
        

        print("\nGenerating sample visualizations...")
        self.visualize_samples(self.test_loader)
        print("\nGenerating attention visualizations...")
        self.visualize_attention(self.test_loader)

        results = {
            "validation": valid_results,
            "test": test_results
        }
        
        results_path = os.path.join(self.results_dir, self.eval_file_name)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Results saved to {results_path}")
        
        return results