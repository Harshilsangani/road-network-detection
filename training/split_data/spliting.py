import os
import shutil
import random

# Set random seed for reproducibility
random.seed(42)

# Define source directories
original_dir = r"C:\2_data\SN3_roads_train_AOI_3_Paris\AOI_3_Paris\patched\f_img"
mask_dir = r"C:\2_data\SN3_roads_train_AOI_3_Paris\AOI_3_Paris\patched\f_mask"

# Define destination directories
train_original_dir =r"C:\2_data\SN3_roads_train_AOI_3_Paris\AOI_3_Paris\patched\p_data\train\img"
train_mask_dir = r"C:\2_data\SN3_roads_train_AOI_3_Paris\AOI_3_Paris\patched\p_data\train\mask"
val_original_dir = r"C:\2_data\SN3_roads_train_AOI_3_Paris\AOI_3_Paris\patched\p_data\val\img"
val_mask_dir = r"C:\2_data\SN3_roads_train_AOI_3_Paris\AOI_3_Paris\patched\p_data\val\mask"

# Create destination directories if they don't exist
for dir_path in [train_original_dir, train_mask_dir, val_original_dir, val_mask_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Get list of files
original_files = os.listdir(original_dir)
mask_files = os.listdir(mask_dir)

# Find matching files
# Find matching files
# Find matching files
matching_files = []
for orig_file in original_files:
    base_filename = os.path.splitext(orig_file)[0]
    
    # For non-augmented images, check the mask directly
    if base_filename + ".png" in mask_files:
        matching_files.append(orig_file)
    
    # For augmented images, check the corresponding mask with the same base name
    elif "aug_" in base_filename and base_filename[4:] + ".png" in mask_files:
        matching_files.append(orig_file)




if not matching_files:
    print("ERROR: No matching files found!")
    print(f"Original files sample: {original_files[:5]}")
    print(f"Mask files sample: {mask_files[:5]}")
    exit()

# Calculate split sizes
total_images = len(matching_files)
val_size = int(total_images * 0.2)  # 20% for validation
train_size = total_images - val_size

# Randomly shuffle the file names
random.shuffle(matching_files)

# Split into train and validation sets
train_files = matching_files[:-val_size]
val_files = matching_files[-val_size:]

# Copy files to respective directories
def copy_files(file_list, src_original, src_mask, dst_original, dst_mask):
    for file_name in file_list:
        try:
            # Verify the original image file exists
            src_original_path = os.path.join(src_original, file_name)
            if not os.path.isfile(src_original_path):
                print(f"Original file not found: {src_original_path}")
                continue  # Skip this file
            
            # Copy original image
            dst_original_path = os.path.join(dst_original, file_name)
            shutil.copy2(src_original_path, dst_original_path)
            
            # Verify the corresponding mask file exists
            base_filename = os.path.splitext(file_name)[0]
            mask_filename = base_filename + ".png"
            src_mask_path = os.path.join(src_mask, mask_filename)
            if not os.path.isfile(src_mask_path):
                print(f"Mask file not found: {src_mask_path}")
                continue  # Skip this file
            
            # Copy mask
            dst_mask_path = os.path.join(dst_mask, mask_filename)
            shutil.copy2(src_mask_path, dst_mask_path)
        except Exception as e:
            print(f"Error copying {file_name}: {e}")


# Copy training files
print("Copying training files...")
copy_files(train_files, original_dir, mask_dir, train_original_dir, train_mask_dir)

# Copy validation files
print("Copying validation files...")
copy_files(val_files, original_dir, mask_dir, val_original_dir, val_mask_dir)

# Print summary
print(f"Total matching images: {total_images}")
print(f"Training images: {len(train_files)}")
print(f"Validation images: {len(val_files)}")