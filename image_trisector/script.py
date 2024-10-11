import os
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, Normalize

# Load a smaller MiDaS model for depth estimation to avoid large file size issues
def load_midas_model():
    model_type = "MiDaS_small"  # Use a smaller model to reduce dependencies
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    return midas, midas_transforms, device

# Perform depth estimation using the MiDaS model
def estimate_depth(image, midas, midas_transforms, device):
    input_batch = midas_transforms(image).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    return depth_map

# Segment image into foreground, middle ground, and background based on depth
def segment_by_depth(image, depth_map):
    # Normalize depth values
    min_depth, max_depth = np.min(depth_map), np.max(depth_map)
    depth_map_normalized = (depth_map - min_depth) / (max_depth - min_depth)

    # Define thresholds for splitting based on depth
    foreground_threshold = 0.3
    middle_ground_threshold = 0.6

    # Create masks for foreground, middle ground, and background
    foreground_mask = depth_map_normalized < foreground_threshold
    middle_ground_mask = (depth_map_normalized >= foreground_threshold) & (depth_map_normalized < middle_ground_threshold)
    background_mask = depth_map_normalized >= middle_ground_threshold

    # Apply masks to the image
    foreground = np.copy(image)
    foreground[~foreground_mask] = [0, 0, 0, 0]  # Apply transparency where it's not foreground

    middle_ground = np.copy(image)
    middle_ground[~middle_ground_mask] = [0, 0, 0, 0]  # Apply transparency where it's not middle ground

    background = np.copy(image)
    background[~background_mask] = [0, 0, 0, 0]  # Apply transparency where it's not background

    return foreground, middle_ground, background

# Function to split an image using depth estimation into foreground, middle ground, and background
def split_image(image_path, output_folder, filename, midas, midas_transforms, device):
    # Load image and convert to RGB
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

    # Estimate depth
    depth_map = estimate_depth(image_np, midas, midas_transforms, device)

    # Segment image based on depth map
    foreground, middle_ground, background = segment_by_depth(image_np, depth_map)

    # Save the output images with transparency (RGBA)
    foreground_img = Image.fromarray(foreground.astype(np.uint8)).convert('RGBA')
    middle_ground_img = Image.fromarray(middle_ground.astype(np.uint8)).convert('RGBA')
    background_img = Image.fromarray(background.astype(np.uint8)).convert('RGBA')

    foreground_img.save(os.path.join(output_folder, f"{filename}_foreground.png"))
    middle_ground_img.save(os.path.join(output_folder, f"{filename}_middle_ground.png"))
    background_img.save(os.path.join(output_folder, f"{filename}_background.png"))

    print(f"Saved trisected images for {filename}")

# Function to process images in folder A and output to folder B
def process_images(input_folder, output_folder, midas, midas_transforms, device):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".tiff")):
            input_image_path = os.path.join(input_folder, filename)
            print(f"Processing {filename}...")

            # Split the image into three parts based on depth
            split_image(input_image_path, output_folder, os.path.splitext(filename)[0], midas, midas_transforms, device)

if __name__ == "__main__":
    # Folder A contains the input images, and Folder B will hold the output images
    input_folder = "A"
    output_folder = "B"

    # Load MiDaS model and transformations
    midas, midas_transforms, device = load_midas_model()

    # Process images using depth-based segmentation
    process_images(input_folder, output_folder, midas, midas_transforms, device)
