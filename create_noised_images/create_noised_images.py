import torch
from torchvision import transforms
from PIL import Image
from diffusers import DDPMScheduler

# 1. Load the image
reference_image = "Acadian_Flycatcher.jpg"
image = Image.open(reference_image).convert("RGB")

# 2. Convert image to a tensor and normalize to [-1, 1]
# Diffusion model schedulers expect inputs in this normalized range
transform = transforms.Compose([
    transforms.Resize((512, 512)), 
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
clean_image_tensor = transform(image).unsqueeze(0) # Add batch dimension

# 3. Initialize a diffusion scheduler
# This handles the exact mathematics of adding noise at different timesteps
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# 4. Define the timestep (how much noise to add)
# 0 = clean image, 999 = pure Gaussian noise
timesteps = torch.tensor([100]).long() 

# 5. Generate random Gaussian noise
noise = torch.randn_like(clean_image_tensor)

# 6. Add noise to the clean image based on the timestep
noisy_image_tensor = noise_scheduler.add_noise(clean_image_tensor, noise, timesteps)

# 7. Convert the noisy tensor back to a viewable PIL Image
# Denormalize from [-1, 1] back to [0, 1]
noisy_image_tensor = (noisy_image_tensor / 2 + 0.5).clamp(0, 1)

# Convert to array then to PIL Image
noisy_image_np = noisy_image_tensor.squeeze(0).permute(1, 2, 0).numpy()
noisy_image_pil = Image.fromarray((noisy_image_np * 255).astype("uint8"))

noisy_image_pil.save(f"noised_{reference_image}")