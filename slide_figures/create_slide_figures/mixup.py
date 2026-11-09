import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def mixup_data(x, y, alpha=0.4):
    """
    Applies Mixup data augmentation: linearly blends two images and their target labels.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    
    # For testing with exactly 2 images, constrain lambda between 0.3 and 0.7
    # so the alpha blending is clearly visible and not overwhelmingly one image.
    lam = max(0.3, min(lam, 0.7))

    batch_size = x.size(0)
    
    # FORCED SWAP: Prevent randperm from mapping an image to itself.
    # Reverses the batch order so Image 1 is guaranteed to blend with Image 2.
    index = torch.arange(batch_size - 1, -1, -1)

    mixed_x = lam * x + (1 - lam) * x[index]
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

    # Apply Mixup
    mixup_imgs, mixup_labels, mix_lam = mixup_data(images, labels, alpha=0)

    # Save synthetic JPG output
    T.ToPILImage()(mixup_imgs[0]).save("mixup_synthetic_fixed.jpg")
    print(f"Saved 'mixup_synthetic_fixed.jpg' (lambda={mix_lam:.2f})")

    # Display comparison
    def to_np(tensor):
        return tensor.permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(to_np(images[0]))
    axes[0].set_title("Input JPG 1")
    
    axes[1].imshow(to_np(images[1]))
    axes[1].set_title("Input JPG 2")
    
    axes[2].imshow(to_np(mixup_imgs[0]))
    axes[2].set_title(f"Mixup Synthetic Output\n(λ={mix_lam:.2f})")

    for ax in axes:
        ax.axis("off")
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()