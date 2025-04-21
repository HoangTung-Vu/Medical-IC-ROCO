import os
import torch 
import torch.optim as optim


def load_checkpoint(model, checkpoint_path, device) -> int:
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return 0
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return model

