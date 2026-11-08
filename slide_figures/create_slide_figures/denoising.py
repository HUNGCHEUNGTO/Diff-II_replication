from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

DESIRED_CATEGORY = 'Acadian_Flycatcher' # Used to perform intra-category denoising
OTHER_CATEGORY =  'American_Goldfinch' # Used to perform inter-category denoising

# 1. Initialize the image-to-image pipeline
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16
).to("cuda")

# 2. Define the categories and load reference images
category_a_prompt = f"a photo of a {DESIRED_CATEGORY.replace('_', ' ')} bird"
category_b_prompt = f"a photo of a {OTHER_CATEGORY.replace('_', ' ')} bird"

# Load real samples (ensure you have these images locally)
image_category_a = Image.open(Path(DESIRED_CATEGORY).with_suffix(".jpg")).convert("RGB").resize((512, 512))
image_category_b = Image.open(Path(OTHER_CATEGORY).with_suffix(".jpg")).convert("RGB").resize((512, 512))

# 3. Define the noise level
# The 'strength' parameter represents the "Add noise" step. 
# 0.0 means no noise (returns original image), 1.0 means complete destruction (pure noise).
# A value around 0.6 - 0.8 allows the model to retain structural layout while changing semantics.
noise_strength = 0.7 

# ---------------------------------------------------------
# Method A: Intra-Category Denoising
# Reference Image: Category A | Prompt: Category A
# ---------------------------------------------------------
intra_result = pipe(
    prompt=category_a_prompt,
    image=image_category_a,
    strength=noise_strength,
    guidance_scale=7.5
).images[0]

intra_result.save(Path(DESIRED_CATEGORY + "_intra_output.jpg"))

# ---------------------------------------------------------
# Method B: Inter-Category Denoising
# Reference Image: Category B | Prompt: Category A
# (Generating a Category A image using Category B's structure)
# ---------------------------------------------------------
inter_result = pipe(
    prompt=category_a_prompt,
    image=image_category_b,
    strength=noise_strength,
    guidance_scale=7.5
).images[0]

inter_result.save(Path(DESIRED_CATEGORY + "_inter_output.jpg"))