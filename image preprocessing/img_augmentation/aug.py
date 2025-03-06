import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

# Directories
input_images_dir = r"C:\shanghai\AOI_4_Shanghai\shanghai_train\cmp_new_images_8bit"  # Replace with your images folder path
input_masks_dir = r"C:\shanghai\AOI_4_Shanghai\shanghai_train\cmp_new_masks2m"    # Replace with your masks folder path
output_images_dir = r"C:\shanghai\AOI_4_Shanghai\shanghai_train\aug_new_8bit"
output_masks_dir = r"C:\shanghai\AOI_4_Shanghai\shanghai_train\aug_mask"

# Create output directories
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_masks_dir, exist_ok=True)

# Get all image and mask paths
image_paths = sorted(glob(os.path.join(input_images_dir, "*.png")))
mask_paths = sorted(glob(os.path.join(input_masks_dir, "*.png")))

# Ensure the same number of images and masks
assert len(image_paths) == len(mask_paths), "Mismatch in the number of images and masks."

# Calculate the number of images for each transformation
n_total = len(image_paths)
n_per_transform = n_total // 5

# Define transformations
def rotate_image(image, angle):
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

def flip_image(image, flip_code):
    return cv2.flip(image, flip_code)

# Apply augmentations
for i, (img_path, mask_path) in tqdm(enumerate(zip(image_paths, mask_paths)), total=len(image_paths)):
    # Read the image and mask
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # Load mask as grayscale or as is

    # Choose augmentation based on the index
    if i < n_per_transform:
        img_aug, mask_aug = rotate_image(img, 90), rotate_image(mask, 90)
    elif i < 2 * n_per_transform:
        img_aug, mask_aug = rotate_image(img, 180), rotate_image(mask, 180)
    elif i < 3 * n_per_transform:
        img_aug, mask_aug = rotate_image(img, 270), rotate_image(mask, 270)
    elif i < 4 * n_per_transform:
        img_aug, mask_aug = flip_image(img, 1), flip_image(mask, 1)  # Horizontal flip
    else:
        img_aug, mask_aug = flip_image(img, 0), flip_image(mask, 0)  # Vertical flip

    # Save augmented image and mask
    img_filename = os.path.basename(img_path)
    mask_filename = os.path.basename(mask_path)
    
    cv2.imwrite(os.path.join(output_images_dir, f"aug_{i}_{img_filename}"), img_aug)
    cv2.imwrite(os.path.join(output_masks_dir, f"aug_{i}_{mask_filename}"), mask_aug)

print("Augmentation completed.")
