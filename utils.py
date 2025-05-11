# utils.py
import torch
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import cv2

# Load SAM
def load_sam_model():
    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
    predictor = SamPredictor(sam)
    return predictor

# Load CLIP
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def get_clip_embedding(image, model, processor):
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

    # Normalize the embeddings
    embedding = outputs[0]
    embedding = embedding / embedding.norm(p=2)
    return embedding.cpu().numpy()