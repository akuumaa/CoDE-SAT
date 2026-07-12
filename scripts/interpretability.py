"""
    Shows which parts of an image were important for the model's prediction

    - ConvNeXt: Grad-CAM
    - DeiT: Attention Rollout

    The model is only evaluated, not trained

    run:
        uv run python scripts/interpretability.py \\
            --model convnext_tiny.fb_in1k \\
            --checkpoint outputs/checkpoints/convnext_tiny.fb_in1k_sgd_stage1_best.pt

    run on your own photos instead of eurosat test images:
        uv run python scripts/interpretability.py \\
            --model deit_small_distilled_patch16_224.fb_in1k \\
            --checkpoint outputs/checkpoints/deit_small_distilled_patch16_224.fb_in1k_sgd_stage1_best.pt \\
            --images-dir demo_images
"""

from __future__ import annotations

import argparse
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image

# use the same image size and normalization as during training.
IMG_SIZE = 224
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def create_model(model_name: str, num_classes: int):
    # creates the model that matches the saved checkpoint
    return timm.create_model(model_name, pretrained=False, num_classes=num_classes)


def load_classes(splits_dir: Path) -> list[str]:
    classes_df = pd.read_csv(splits_dir / "classes.txt").sort_values("label")
    return classes_df["class_name"].tolist()


def load_image(path: Path) -> tuple[torch.Tensor, np.ndarray]:
    image = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    # keep the original RGB values for visualization
    rgb = np.array(image).astype(np.float32) / 255.0  
    # change the order from [height, width, channels] to [channels, height, width]
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)
    # normalize the image in the same way as during training
    tensor = (tensor - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]

    return tensor, rgb


def pick_test_images(splits_dir: Path, classes: list[str], num_images: int, seed: int):
    test_frame = pd.read_csv(splits_dir / "test.csv")
    # first select one image per class, then shuffle and limit the result
    per_class = test_frame.groupby("label", group_keys=False).sample(n=1, random_state=seed)
    per_class = per_class.sample(frac=1.0, random_state=seed).head(num_images)

    return [
        (Path(row["path"]), classes[int(row["label"])])
        for _, row in per_class.iterrows()
    ]


def pick_demo_images(images_dir: Path, num_images: int):
    paths = sorted(images_dir.glob("*"))[:num_images]
    return [(path, None) for path in paths]


def gradcam(model, image_tensor: torch.Tensor) -> tuple[np.ndarray, int]:
    """Creates a Grad-CAM heatmap for a CNN such as ConvNeXt"""
    # add a batch dimension: [C, H, W] -> [1, C, H, W].
    x = image_tensor.unsqueeze(0)

    # compute image features from the last spatial stage of the model
    features = model.forward_features(x)
    features.retain_grad()  # not a leaf tensor, so grad isn't kept by default

    # use the features to compute the class scores
    logits = model.forward_head(features)
    predicted = int(logits.argmax(dim=1).item())

    # compute which features influenced the prediction
    model.zero_grad(set_to_none=True)
    logits[0, predicted].backward()

    gradients = features.grad[0] 
    activations = features.detach()[0]

    # give more weight to feature channels that were important
    channel_weights = gradients.mean(dim=(1, 2))
    cam = torch.relu((channel_weights[:, None, None] * activations).sum(dim=0))
    # scale the values to the range from 0 to 1
    cam = cam / (cam.max() + 1e-8)
    # resize the small heatmap to the input image size
    cam = F.interpolate(cam[None, None], size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    return cam[0, 0].detach().cpu().numpy(), predicted


def _attention_forward_with_capture(self, x, attn_mask=None, is_causal=False):
    # Create query, key, and value tensors for all attention heads
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    # compute how strongly each token attends to the other tokens
    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    # store the weights for the later attention rollout
    self.attn_weights = attn.detach()

    x = attn @ v
    x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
    x = self.norm(x)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


@torch.no_grad()
def attention_rollout(model, image_tensor: torch.Tensor) -> tuple[np.ndarray, int]:
    original_forwards = [block.attn.forward for block in model.blocks]
    # temporarily replace the attention function with the capturing version
    for block in model.blocks:
        block.attn.forward = types.MethodType(_attention_forward_with_capture, block.attn)

    try:
        # run the prediction and collect the attention weights
        logits = model(image_tensor.unsqueeze(0))
        predicted = int(logits.argmax(dim=1).item())

        num_tokens = model.blocks[0].attn.attn_weights.shape[-1]
        rollout = torch.eye(num_tokens)

        for block in model.blocks:
            # average all attention heads and add the skip connection
            # normalize the result so every row sums to 1
            layer_attn = block.attn.attn_weights[0].mean(dim=0)
            layer_attn = layer_attn + torch.eye(num_tokens)
            layer_attn = layer_attn / layer_attn.sum(dim=-1, keepdim=True)
            rollout = layer_attn @ rollout
    finally:
        for block, original_forward in zip(model.blocks, original_forwards):
            block.attn.forward = original_forward

    # get the relevance of each image patch from the CLS token
    # skip the extra prefix tokens used by distilled models
    num_prefix = model.num_prefix_tokens
    patch_relevance = rollout[0, num_prefix:]

    grid_size = int(patch_relevance.numel() ** 0.5)
    heatmap = patch_relevance.reshape(1, 1, grid_size, grid_size)
    heatmap = F.interpolate(heatmap, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    heatmap = heatmap[0, 0]
    heatmap = heatmap / (heatmap.max() + 1e-8)

    return heatmap.cpu().numpy(), predicted


def save_overlay(rgb: np.ndarray, heatmap: np.ndarray, title: str, out_path: Path):
    colored_heatmap = plt.get_cmap("jet")(heatmap)[..., :3]
    overlay = np.clip(0.5 * rgb + 0.5 * colored_heatmap, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(rgb)
    axes[0].set_title("input")
    axes[1].imshow(overlay)
    axes[1].set_title("heatmap")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="timm model name")
    parser.add_argument("--checkpoint", type=Path, required=True, help="path to a *_best.pt checkpoint")
    parser.add_argument("--images-dir", type=Path, default=None,
                         help="folder of your own photos instead of EuroSAT test images (domain-shift demo)")
    parser.add_argument("--num-images", type=int, default=6)
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures/interpretability"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    classes = load_classes(args.splits_dir)


    model = create_model(args.model, num_classes=len(classes))
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device).eval()

    is_transformer = hasattr(model, "blocks")

    if args.images_dir is not None:
        images = pick_demo_images(args.images_dir, args.num_images)
    else:
        images = pick_test_images(args.splits_dir, classes, args.num_images, args.seed)

    for path, true_class in images:
        image_tensor, rgb = load_image(path)
        image_tensor = image_tensor.to(device)

        if is_transformer:
            heatmap, predicted = attention_rollout(model, image_tensor)
            method = "attention rollout"
        else:
            heatmap, predicted = gradcam(model, image_tensor)
            method = "grad-cam"

        predicted_class = classes[predicted]
        title = f"{path.name} | pred: {predicted_class}"
        if true_class is not None:
            title += f" | true: {true_class}"

        out_name = f"{args.model}_{method.replace(' ', '')}_{path.stem}.png"
        save_overlay(rgb, heatmap, title, args.out_dir / out_name)
        print(f"{path.name}: predicted={predicted_class}" + (f" true={true_class}" if true_class else ""))


if __name__ == "__main__":
    main()
