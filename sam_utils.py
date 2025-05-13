import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import cv2
import matplotlib.pyplot as plt
import os
import requests

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Visualization function kept for debugging purposes

def visualize_masks(image, masks, title="Masks", save_path=None, filter_info=None):
    plt.figure(figsize=(15, 15))
    plt.imshow(image)
    
    colors = np.random.randint(0, 255, size=(len(masks), 3))
    
    for idx, mask in enumerate(masks):
        x0, y0, w, h = mask['bbox']
        rect = plt.Rectangle((x0, y0), w, h, fill=False, 
                           color=colors[idx]/255, linewidth=2)
        plt.gca().add_patch(rect)
        
        ar = h / w if w > 0 else 0
        area = w * h
        label = f"Mask {idx}\nAR: {ar:.2f}\nArea: {area}"
        
        if filter_info and idx in filter_info:
            label += f"\n{filter_info[idx]}"
            
        plt.text(x0, y0-5, label, color=colors[idx]/255, 
                bbox=dict(facecolor='white', alpha=0.7))

    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()


def download_sam_checkpoint(url, output_path):
    """Download the SAM checkpoint file if it doesn't exist."""
    if not os.path.exists(output_path):
        print(f"Downloading SAM checkpoint from {url}...")
        response = requests.get(url, stream=True)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"SAM checkpoint saved to {output_path}.")

def load_sam():
    """
    Initialize SAM model with parameters optimized for Pokemon card segmentation.
    """
    sam_checkpoint = "sam_vit_b_01ec64.pth"
    model_type = "vit_b"

    # URL to download the SAM checkpoint
    sam_checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"  # Replace FILE_ID with the actual file ID

    # Download the checkpoint if it doesn't exist
    download_sam_checkpoint(sam_checkpoint_url, sam_checkpoint)

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cpu")
    
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=16,
        min_mask_region_area=600000,
        pred_iou_thresh=0.92,
        stability_score_thresh=0.92,
        box_nms_thresh=0.3,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        crop_overlap_ratio=0.1
    )
    return mask_generator

mask_generator = load_sam()

def preprocess_image(pil_img, max_size=1024):
    """
    Preprocess image for optimal SAM performance while maintaining aspect ratio
    Returns:
        preprocessed image, scale factor
    """
    # Get original size
    w, h = pil_img.size
    
    # Calculate new size maintaining aspect ratio
    scale = min(max_size/w, max_size/h)
    if scale < 1:
        new_w, new_h = int(w*scale), int(h*scale)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return np.array(pil_img), scale
    return np.array(pil_img), 1.0

def get_card_crops(pil_img, visualize=True):
    """
    Segment Pokemon cards from an image and return individual card crops.
    """
    # Preprocess image
    image, scale = preprocess_image(pil_img)
    img_height, img_width = image.shape[:2]
    
    # Scale the min_mask_region_area based on the preprocessing scale
    original_area = 600000 if img_height > img_width else 500000
    scaled_area = int(original_area * (scale * scale))
    
    # Adjust min_mask_region_area based on scaled image
    mask_generator.min_mask_region_area = scaled_area
    
    # Generate masks using SAM
    masks = mask_generator.generate(image)
    
    
    def filter_card_masks(masks, img_height, img_width):
        """
        Filter masks based on:
        - Standard aspect ratio (1.4)
        - Reasonable size relative to image
        - Minimal overlap between cards
        """
        filter_info = {}
        
        # Calculate expected card size based on image dimensions
        image_area = img_height * img_width
        image_diagonal = np.sqrt(img_height*img_height + img_width*img_width)
        
        # Cards should be between 1/8 and 1 of the image diagonal
        min_card_diagonal = image_diagonal / 8
        max_card_diagonal = image_diagonal
        
        # Convert diagonal to area using typical card aspect ratio (1.4)
        min_card_area = (min_card_diagonal ** 2) / (1 + 1.4**2)
        max_card_area = (max_card_diagonal ** 2) / (1 + 1.4**2)
        
        # Collect potential card masks
        card_candidates = []
        for idx, mask in enumerate(masks):
            x0, y0, w, h = mask['bbox']
            area = w * h
            ar = h / w if w > 0 else 0
            
            # Check aspect ratio (allowing for both orientations)
            if (1.15 <= ar <= 1.55) or (0.65 <= ar <= 0.87):
                card_candidates.append({
                    'mask': mask,
                    'area': area,
                    'bbox': (x0, y0, x0 + w, y0 + h),
                    'idx': idx
                })
            else:
                filter_info[idx] = f"Rejected: AR {ar:.2f} outside range"
        
        if not card_candidates:
            return [], filter_info

        # Filter by area
        valid_cards = []
        for card in card_candidates:
            area = card['area']
            if min_card_area <= area <= max_card_area:
                valid_cards.append(card)
            else:
                filter_info[card['idx']] = f"Rejected: Area {area:.0f} outside range"
        
        # Sort by area and remove overlaps
        valid_cards.sort(key=lambda x: x['area'], reverse=True)
        final_cards = []
        
        for card in valid_cards:
            x0, y0, x1, y1 = card['bbox']
            overlap = False
            
            for accepted_card in final_cards:
                ax0, ay0, ax1, ay1 = accepted_card['bbox']
                ix0 = max(x0, ax0)
                iy0 = max(y0, ay0)
                ix1 = min(x1, ax1)
                iy1 = min(y1, ay1)
                
                if ix0 < ix1 and iy0 < iy1:
                    intersection_area = (ix1 - ix0) * (iy1 - iy0)
                    current_area = (x1 - x0) * (y1 - y0)
                    
                    if intersection_area / current_area > 0.15:
                        overlap = True
                        filter_info[card['idx']] = f"Rejected: Overlap {intersection_area/current_area:.2f}"
                        break
            
            if not overlap:
                final_cards.append(card)
                filter_info[card['idx']] = "Accepted"
        
        return final_cards, filter_info

    filtered_cards, filter_info = filter_card_masks(masks, img_height, img_width)
    
    
    def visualize_with_params(image, masks, title, save_path, filter_info=None):
        """
        Helper function to visualize masks with given parameters.
        """
        visualize_masks(image, masks, title=title, save_path=save_path, filter_info=filter_info)

    # Refactor visualization calls
    if visualize:
        visualize_with_params(image, masks, "Original SAM Masks", "original_masks.png")

        filtered_masks = [card['mask'] for card in filtered_cards]
        visualize_with_params(image, filtered_masks, "Filtered Masks", "filtered_masks.png")
        visualize_with_params(image, masks, "All Masks with Filtering Info", "all_masks_with_info.png", filter_info=filter_info)
    

    # Create card crops
    card_crops = []
    for card in filtered_cards:
        x0, y0, x1, y1 = card['bbox']
        crop = image[y0:y1, x0:x1]
        card_crops.append(Image.fromarray(crop))

    return card_crops