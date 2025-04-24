import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch.utils.data import DataLoader
from datasets import Dataset, concatenate_datasets
from utils.paraphraser import paraphrase
from tqdm import tqdm
import json
import time


def data_augment(
    root_dir : str, 
    output_dir : str,
):
    def get_arrow_files(prefix : str) -> list[str]:
        return sorted([
            os.path.join(root_dir, f) for f in os.listdir(root_dir)
            if f.endswith('.arrow') and f.startswith(prefix)
        ])
    
    arrow_train = get_arrow_files("roco-radiology-train")

    for idx, arrow_file in enumerate(arrow_train):
        print(f"Processing file {idx+1}/{len(arrow_train)}: {os.path.basename(arrow_file)}")
        original_dataset = Dataset.from_file(arrow_file)

        augmented_data = {key : [] for key in original_dataset.features}

        for i, sample in enumerate(tqdm(original_dataset)):
            for key in augmented_data:
                augmented_data[key].append(sample[key])
                
            original_caption = sample['caption']

            try:
                paraphrased_caption = paraphrase(original_caption)
                
                for key in augmented_data:
                    if key == 'caption':
                        augmented_data[key].append(paraphrased_caption)
                    else:
                        augmented_data[key].append(sample[key])
            except Exception as e:
                print(f"Error paraphrasing caption {i}: {e}")
            
            print(f"original : {original_caption}")
            print(f"paraphrase : {paraphrased_caption}\n")

        augmented_dataset = Dataset.from_dict(augmented_data)
        
        output_file = os.path.join(output_dir, os.path.basename(arrow_file))
        augmented_dataset.save_to_disk(output_file)
        print(f"Saved augmented dataset to {output_file}")
        print(f"Original size: {len(original_dataset)}, Augmented size: {len(augmented_dataset)}")


if __name__ == "__main__":
    data_augment(
        root_dir='/home/hoangtungvum/CODE/MIC/data/03471f547bb646a1f447add638d46bb3507523e8',
        output_dir='/home/hoangtungvum/CODE/MIC/data/augmented_data'
    )