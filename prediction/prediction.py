import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import os

def load_model_weights(model, state_dict):
    """
    Helper function to handle state dict key mapping
    """
    # Create a new state dict with corrected keys
    new_state_dict = {}
    
    for k, v in state_dict.items():
        # Handle SE module naming difference
        if '.se_fc1.' in k:
            new_k = k.replace('.se_fc1.', '.se_module.fc1.')
        elif '.se_fc2.' in k:
            new_k = k.replace('.se_fc2.', '.se_module.fc2.')
        else:
            new_k = k
            
        new_state_dict[new_k] = v
    
    # Load the corrected state dict
    model.load_state_dict(new_state_dict)
    return model

class RoadPredictor:
    def __init__(self, model_path, device=None):
        """
        Initialize the predictor with a trained model
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Initialize model
        self.model = smp.DeepLabV3Plus(
            encoder_name='senet154',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
        )
        
        try:
            # Load the state dict
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # If state_dict is nested inside a dictionary, extract it
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
            # Remove 'module.' prefix if it exists (from DataParallel)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # Load the state dict with key mapping
            self.model = load_model_weights(self.model, state_dict)
            print("Model loaded successfully with key mapping")
            
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define preprocessing transform
        self.transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    
    def preprocess_image(self, image):
        """
        Preprocess a single image for prediction
        """
        # Resize image and apply transformations
        transformed = self.transform(image=image)
        image_tensor = transformed['image']
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor

    def predict(self, image, original_size, threshold=0.5):
        """
        Make prediction on a single image and resize mask back to original image dimensions
        """
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            
            # Move to device and make prediction
            with torch.no_grad():
                image_tensor = image_tensor.to(self.device)
                output = self.model(image_tensor)
                predicted_mask = torch.sigmoid(output) > threshold

            # Convert prediction to numpy array
            predicted_mask = predicted_mask.cpu().numpy().squeeze().astype(np.uint8) * 255

            # Resize mask back to the original image dimensions
            predicted_mask = Image.fromarray(predicted_mask)
            predicted_mask = predicted_mask.resize((original_size[1], original_size[0]), Image.NEAREST)
            predicted_mask = np.array(predicted_mask)

            return predicted_mask
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None

    def create_overlay(self, image, mask, alpha=0.5):
        """
        Create an overlay of the prediction on the original image
        """
        # Create a red mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = [255, 0, 0]  # Red color for roads
        
        # Combine original image with mask
        overlay = image.copy()
        mask_idx = mask > 0
        overlay[mask_idx] = (
            alpha * image[mask_idx] + (1 - alpha) * colored_mask[mask_idx]
        ).astype(np.uint8)
        
        return overlay

    def process_directory(self, input_dir, output_dir, save_mask_dir=None, save_overlay_dir=None):
        """
        Process all images in the directory
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if save_mask_dir and not os.path.exists(save_mask_dir):
            os.makedirs(save_mask_dir)
        
        if save_overlay_dir and not os.path.exists(save_overlay_dir):
            os.makedirs(save_overlay_dir)

        # Loop over all images in the directory
        for img_name in os.listdir(input_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                continue
                
            img_path = os.path.join(input_dir, img_name)
            
            # Load the image
            try:
                image = np.array(Image.open(img_path).convert('RGB'))
                original_size = image.shape[:2]  # Store original dimensions
            except Exception as e:
                print(f"Failed to load image {img_path}: {str(e)}")
                continue

            # Run prediction
            mask = self.predict(image, original_size)

            if mask is not None:
                # Save the mask if directory is specified
                if save_mask_dir:
                    mask_filename = os.path.splitext(img_name)[0] + "_mask.png"
                    mask_path = os.path.join(save_mask_dir, mask_filename)
                    Image.fromarray(mask).save(mask_path)

                # Create and save overlay if directory is specified
                if save_overlay_dir:
                    overlay = self.create_overlay(image, mask)
                    overlay_filename = os.path.splitext(img_name)[0] + "_overlay.png"
                    overlay_path = os.path.join(save_overlay_dir, overlay_filename)
                    Image.fromarray(overlay).save(overlay_path)

                print(f"Successfully processed {img_name}")
            else:
                print(f"Failed to process {img_name}")


# Example usage
if __name__ == '__main__':
    MODEL_PATH = r"C:\shanghai\AOI_4_Shanghai\shanghai_train\DeepLabV3Plus_senet154_best_model_val_iou0.6513.pth"
    INPUT_DIR = r"C:\Users\HARSHIL\OneDrive\Desktop\opennnn\all_missing"
    OUTPUT_DIR = r"C:\Users\HARSHIL\OneDrive\Desktop\opennnn\6513_res"
    SAVE_MASK_DIR = os.path.join(OUTPUT_DIR, 'masks')
    SAVE_OVERLAY_DIR = os.path.join(OUTPUT_DIR, 'overlays')

    # Initialize predictor and process images
    try:
        predictor = RoadPredictor(MODEL_PATH)
        predictor.process_directory(
            INPUT_DIR,
            OUTPUT_DIR,
            save_mask_dir=SAVE_MASK_DIR,
            save_overlay_dir=SAVE_OVERLAY_DIR
        )
        print("Processing completed successfully")
    except Exception as e:
        print(f"Error occurred: {str(e)}")