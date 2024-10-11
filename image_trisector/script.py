import os
import cv2
import numpy as np
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

def process_image(image_np, filename, output_folder):
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Thresholding to create a binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours based on area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Create blank images for output
    foreground_image = np.zeros_like(image_np, dtype=np.uint8)
    middle_ground_image = np.zeros_like(image_np, dtype=np.uint8)
    background_image = np.zeros_like(image_np, dtype=np.uint8)

    # Assign contours to foreground, middle ground, and background
    if contours:
        # Foreground: Largest contour (assumed to be the closest object)
        cv2.drawContours(foreground_image, contours[:1], -1, (255, 255, 255), thickness=cv2.FILLED)

        # Middle ground: Second largest contour
        if len(contours) > 1:
            cv2.drawContours(middle_ground_image, contours[1:2], -1, (255, 255, 255), thickness=cv2.FILLED)

    # Background is everything else
    background_image = cv2.bitwise_and(image_np, image_np, mask=cv2.bitwise_not(cv2.bitwise_or(foreground_image, middle_ground_image)))

    # Save images as PNG
    cv2.imwrite(os.path.join(output_folder, f"{filename}_foreground.png"), foreground_image)
    cv2.imwrite(os.path.join(output_folder, f"{filename}_middle_ground.png"), middle_ground_image)
    cv2.imwrite(os.path.join(output_folder, f"{filename}_background.png"), background_image)

def process_images(input_folder, output_folder, midas, transform, device):
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff')):
            input_image_path = os.path.join(input_folder, filename)
            image_np = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
            if image_np is None:
                print(f"Error loading image: {input_image_path}")
                continue
            
            process_image(image_np, os.path.splitext(filename)[0], output_folder)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize midas and transform
midas = load_midas_model(device)
transform = get_transform()

# Call process_images with the input and output folder paths
input_folder = 'image_trisector/A'  # Assuming A is a folder in the current directory
output_folder = 'image_trisector/B'  # Assuming B is a folder in the current directory
process_images(input_folder, output_folder, midas, transform, device)
