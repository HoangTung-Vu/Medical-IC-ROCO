import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import torch
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import math
import re

def reconstruct_heatmap(conv_layers, rf_map):
    """
    Dùng cho mô hình sử dụng Conv
    Args:
        conv_layers: list of nn.Conv2d (stride >= 1)
        rf_map: Tensor, shape (B,1,H',W'), receptive field heatmap (1 channel)
    Returns:
        heatmap: Tensor, shape (B,1,H, W), phóng ngược về kích thước gốc nhờ upsample + smoothing
    """
    x = rf_map
    for conv in reversed(conv_layers):
        stride = conv.stride[0] if isinstance(conv.stride, (tuple, list)) else conv.stride
        x = F.interpolate(x, scale_factor=stride, mode='bilinear', align_corners=False)

        k = conv.kernel_size[0] if isinstance(conv.kernel_size, (tuple, list)) else conv.kernel_size
        smoothing_kernel = torch.ones(1, 1, k, k, device=x.device) / (k * k)
        x = F.conv2d(x, weight=smoothing_kernel, padding=k // 2)
        
    return x

def reconstruct_heatmap_grid(rf_map, target_size = (224,224)):
        return rf_map


def explain_inference_image(img_tensor: torch.Tensor, model, device='cuda'):
    """
    Chạy model greedy captioning với attention map.

    Args:
        img_tensor (torch.Tensor): [1, 3, H, W] input image tensor
        model: Trained captioning model (BaseMD subclass)
        device (str): 'cuda' or 'cpu's

    Returns:
        List[str]: caption words
        List[np.ndarray]: heatmaps (H, W)
    """
    model.eval()
    img_tensor = img_tensor.to(device)

    caption, attention_maps = model.caption_image_greedy(img_tensor)

    grid_size = int(math.sqrt(attention_maps[0].size(0)))
    conv_layers = [
        nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3),
        nn.Conv2d(64, 192, kernel_size=3, stride=2, padding=1),
        nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1)
    ]

    heatmaps = [
        reconstruct_heatmap(
            conv_layers,
            attn.view(1, 1, grid_size, grid_size)
        ).squeeze(0).squeeze(0).detach().cpu().numpy()
        for attn in attention_maps
    ]

    caption_words = caption.split()
    return caption_words, heatmaps


def overlay_heatmap(image_tensor: torch.Tensor, heatmap: np.ndarray, alpha: float = 0.5) -> plt.Figure:
    """
    Overlay a heatmap onto the image and return a matplotlib figure.

    Args:
        image_tensor (torch.Tensor): [1, 3, H, W] unnormalized image tensor
        heatmap (np.ndarray): 2D heatmap array (H', W')
        alpha (float): blending factor
        cmap (str): colormap name for matplotlib

    Returns:
        plt.Figure: matplotlib figure object
    """
    unnorm = image_tensor.cpu() * torch.tensor([0.229,0.224,0.225]).view(3,1,1) + torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    img_arr = (unnorm.clamp(0,1).permute(1,2,0).numpy() * 255).astype(np.uint8)
    H, W = img_arr.shape[:2]

    fig, ax = plt.subplots()
    ax.imshow(img_arr)
    ax.imshow(heatmap, alpha=alpha, extent=(0, W, H, 0))
    ax.axis('off')  # remove axes
    fig.tight_layout(pad=0)

    return fig


def save_image(fig: plt.Figure, save_path: str) -> None:
    """
    Save a matplotlib figure to disk, creating directories as needed.

    Args:
        fig (plt.Figure): Matplotlib figure containing the image and heatmap overlay
        save_path (str): Path to save the image
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
