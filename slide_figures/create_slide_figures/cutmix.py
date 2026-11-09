import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def rand_bbox(size, lam):
    """Computes random bounding box coordinates proportional to (1 - lam)."""
    W, H = size[3], size[2]  # PyTorch shape is (B, C, H, W)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0):
    """
    Applies CutMix data augmentation: cuts a rectangular region from one image 
    and pastes it onto another image, adjusting target weights proportionally.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    
    # For testing with exactly 2 images, force lambda to be at most 0.7 
    # so the bounding box is always large enough to see clearly.
    lam = min(lam, 0.7) 

    batch_size = x.size(0)
    
    # FORCED SWAP: Instead of randperm (which has a 50% chance of matching original),
    # we force the index to reverse the batch, guaranteeing Image 1 mixes with Image 2.
    index = torch.arange(batch_size - 1, -1, -1)

    mixed_x = x.clone()
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)

    # Paste bounding box patch (H, W slicing)
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    # Adjust lambda based on actual modified area proportion
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    mixed_y = lam * y + (1 - lam) * y[index]

    return mixed_x, mixed_y, lam

def load_and_prep_images(path1, path2, img_size=(256, 256)):
    """Loads two JPG files, resizes them, and converts them to PyTorch Tensors."""
    transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor()
    ])
    
    img1 = transform(Image.open(path1).convert("RGB"))
    img2 = transform(Image.open(path2).convert("RGB"))
    
    images = torch.stack([img1, img2])
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    return images, labels

def main():
    # Target input file paths
    img_path_1 = "figs/Acadian_Flycatcher.jpg"
    img_path_2 = "figs/Acadian_Flycatcher_2.jpg"

    images, labels = load_and_prep_images(img_path_1, img_path_2)

    # Apply CutMix
    cutmix_imgs, cutmix_labels, cut_lam = cutmix_data(images, labels, alpha=1.0)

    # Save synthetic JPG output
    T.ToPILImage()(cutmix_imgs[0]).save("cutmix_synthetic_fixed.jpg")
    print(f"Saved 'cutmix_synthetic_fixed.jpg' (lambda={cut_lam:.2f})")

    # Display comparison
    def to_np(tensor):
        return tensor.permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(to_np(images[0]))
    axes[0].set_title("Input JPG 1")
    
    axes[1].imshow(to_np(images[1]))
    axes[1].set_title("Input JPG 2")
    
    axes[2].imshow(to_np(cutmix_imgs[0]))
    axes[2].set_title(f"CutMix Synthetic Output\n(λ={cut_lam:.2f})")

    for ax in axes:
        ax.axis("off")
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()