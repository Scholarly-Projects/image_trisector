import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# Load the MiDaS model
def load_midas_model(device):
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS", pretrained=True)
    midas.eval()
    midas.to(device)
    return midas

# Define the transformation
def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((384, 384)),  # Adjust as needed
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def estimate_depth(image_tensor, midas, device):
    with torch.no_grad():
        depth_map = midas(image_tensor.unsqueeze(0).to(device))  # Add batch dimension
    return depth_map.squeeze().cpu().numpy()  # Remove batch dimension and convert to NumPy array

def split_image(image_path, output_folder, filename, midas, transform, device):
    print(f"Processing image: {image_path}")

    # Load image
    image_np = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image_np is None:
        print(f"Error loading image: {image_path}")
        return

    # Resize while maintaining aspect ratio
    h, w, _ = image_np.shape
    new_w = 640  # Set desired width
    new_h = int((new_w / w) * h)
    image_np_resized = cv2.resize(image_np, (new_w, new_h))

    # Convert NumPy array to PIL Image
    image_pil = Image.fromarray(cv2.cvtColor(image_np_resized, cv2.COLOR_BGR2RGB))  # Convert from BGR to RGB

    # Apply transformations
    image_tensor = transform(image_pil)  # Apply transform to PIL image

    # Estimate depth
    depth_map = estimate_depth(image_tensor, midas, device)

    # Ensure depth map has the same dimensions as the resized image
    depth_map_resized = cv2.resize(depth_map, (new_w, new_h))

    # Create binary masks for foreground, middle ground, and background based on depth thresholds
    foreground_mask = depth_map_resized < 0.5
    middle_ground_mask = (depth_map_resized >= 0.5) & (depth_map_resized < 0.8)
    background_mask = depth_map_resized >= 0.8

    # Create output images with transparency
    foreground_image = np.zeros_like(image_np_resized, dtype=np.uint8)
    middle_ground_image = np.zeros_like(image_np_resized, dtype=np.uint8)
    background_image = np.zeros_like(image_np_resized, dtype=np.uint8)

    # Apply masks to create separate images
    foreground_image[foreground_mask] = image_np_resized[foreground_mask]
    middle_ground_image[middle_ground_mask] = image_np_resized[middle_ground_mask]
    background_image[background_mask] = image_np_resized[background_mask]

    # Convert to RGBA for transparency
    foreground_image = cv2.cvtColor(foreground_image, cv2.COLOR_BGR2RGBA)
    middle_ground_image = cv2.cvtColor(middle_ground_image, cv2.COLOR_BGR2RGBA)
    background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGBA)

    # Add transparency
    foreground_image[..., 3] = (foreground_mask * 255).astype(np.uint8)  # Foreground mask as alpha channel
    middle_ground_image[..., 3] = (middle_ground_mask * 255).astype(np.uint8)  # Middle ground mask as alpha channel
    background_image[..., 3] = (background_mask * 255).astype(np.uint8)  # Background mask as alpha channel

    # Save images as PNG
    fg_output_path = os.path.join(output_folder, f"{filename}_foreground.png")
    mg_output_path = os.path.join(output_folder, f"{filename}_middle_ground.png")
    bg_output_path = os.path.join(output_folder, f"{filename}_background.png")

    print(f"Saving images to {fg_output_path}, {mg_output_path}, {bg_output_path}")
    
    cv2.imwrite(fg_output_path, foreground_image)
    cv2.imwrite(mg_output_path, middle_ground_image)
    cv2.imwrite(bg_output_path, background_image)

    print("Images saved successfully.")

def process_images(input_folder, output_folder, midas, transform, device):
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff')):
            input_image_path = os.path.join(input_folder, filename)
            split_image(input_image_path, output_folder, os.path.splitext(filename)[0], midas, transform, device)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize midas and transform
midas = load_midas_model(device)
transform = get_transform()

# Call process_images with the input and output folder paths
input_folder = 'image_trisector/A'  # Assuming A is a folder in the current directory
output_folder = 'image_trisector/B'  # Assuming B is a folder in the current directory
process_images(input_folder, output_folder, midas, transform, device)
