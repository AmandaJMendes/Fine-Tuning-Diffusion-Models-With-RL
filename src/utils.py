from PIL import Image
import numpy as np
import os

def display_sample(latent, label = ""):
  image_processed = latent.cpu().permute(0, 2, 3, 1)
  image_processed = (image_processed + 1.0) * 127.5
  image_processed = image_processed.numpy().astype(np.uint8)
  
  # Create output directory if it doesn't exist
  os.makedirs("output", exist_ok=True)
  
  # Convert numpy array to PIL Image
  pil_image = Image.fromarray(image_processed[0])
  
  # Create filename from label or use default
  filename = f"output/{label.replace(' ', '_')}.png" if label else "output/sample.png"
  
  # Save the image to file
  pil_image.save(filename)