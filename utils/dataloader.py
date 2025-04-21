import os
import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Callable
from transformers import AutoTokenizer
from datasets import Dataset, concatenate_datasets
from torchvision import transforms
class Tokenizer:
    def __init__(self, max_length: int = 512):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
        )

    def __len__(self):
        return len(self.tokenizer)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            max_length=self.max_length,
            truncation=True
        )

    def batch_encode(
        self, texts: List[str], add_special_tokens: bool = True, 
        padding: bool = True, return_tensors: str = "pt"
    ) -> Dict:
        return self.tokenizer(
            texts,
            add_special_tokens=add_special_tokens,
            max_length=self.max_length,
            padding=padding,
            truncation=True,
            return_tensors=return_tensors
        )

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def get_pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    def get_vocabulary(self) -> Dict[str, int]:
        return self.tokenizer.get_vocab()


class Collate:
    def __init__(self, tokenizer: Tokenizer, transform : Optional[Callable] = None):
        self.tokenizer = tokenizer
        self.transform = transform

    def __call__(self, batch):
        images = [self.transform(item["image"]) if self.transform else item["image"] for item in batch]
        images = torch.stack(images)

        captions = [item["caption"] for item in batch]

        encoded = [self.tokenizer.encode(caption, add_special_tokens=True) for caption in captions]
        max_len = max(len(e) for e in encoded)

        pad_token_id = self.tokenizer.get_pad_token_id()
        padded_targets = torch.full((len(encoded), max_len), pad_token_id, dtype=torch.long)

        for i, tokens in enumerate(encoded):
            padded_targets[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)

        return images, padded_targets, max_len


def get_dataloader(
    root_dir: str,
    transform: Optional[Callable] = None,
    batch_size: int = 32, 
    num_workers: int = 4, 
    shuffle: bool = True,
    pin_memory: bool = True,
    img_cache_size: Optional[int] = 100
):
    def get_arrow_files(prefix: str) -> List[str]:
        return sorted([
            os.path.join(root_dir, f) for f in os.listdir(root_dir)
            if f.endswith('.arrow') and f.startswith(prefix)
        ])
    
    # Dataloaders
    def make_loader(dataset, shuffle_flag):
        return DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)  # chuyển ảnh grayscale -> RGB
        ])

    tokenizer = Tokenizer()
    collate_fn = Collate(tokenizer, transform)

    # Load datasets
    arrow_train = get_arrow_files("roco-radiology-train")
    arrow_valid = get_arrow_files("roco-radiology-validation")
    arrow_test = get_arrow_files("roco-radiology-test")

    assert arrow_valid and arrow_test, "Validation/Test .arrow files not found."

    datasets_train = [Dataset.from_file(f) for f in arrow_train]
    trainset = concatenate_datasets(datasets_train)
    validset = Dataset.from_file(arrow_valid[0])
    testset = Dataset.from_file(arrow_test[0])


    trainloader = make_loader(trainset, shuffle)
    validloader = make_loader(validset, False)
    testloader = make_loader(testset, False)

    return {"data" : (trainloader, validloader, testloader), "tokenizer" : tokenizer}

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    trainloader, validloader, testloader = get_dataloader(
        root_dir='../data/03471f547bb646a1f447add638d46bb3507523e8',
        batch_size=16
    )["data"]

    print(f"Trainloader: {len(trainloader)} batches")
    print(f"Validloader: {len(validloader)} batches")
    print(f"Testloader: {len(testloader)} batches")

    sample_batch = next(iter(trainloader))
    images, padded_captions, max_len = sample_batch

    print(f"\nSample batch:")
    print(f"  Images shape: {images.shape}")

    print(f"  Captions shape: {padded_captions.shape}")
    print(f"  Max caption length in this batch: {max_len}")

    tokenizer = Tokenizer()
    decoded_caption = tokenizer.decode(padded_captions[0].tolist(), skip_special_tokens=True)
    print(f"  Sample encoded caption: {padded_captions[0]}")
    print(f"  Sample decoded caption: {decoded_caption}")

    image_tensor = images[0]  # shape [C, H, W]
    
    # Giả sử ảnh đã được normalize, ta cần unnormalize để hiển thị (nếu áp dụng transform)
    def show_image(img_tensor, title=None):
        img = img_tensor.clone().detach()
        img = transforms.ToPILImage()(img)
        plt.imshow(img)
        if title:
            plt.title(title)
        plt.axis("off")
        plt.show()

    show_image(image_tensor, title=decoded_caption)

    