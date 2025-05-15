# utils.py
import sys
import os

# Fix torch.compiler before any imports
import torch
if not hasattr(torch, "compiler"):
    class MockCompiler:
        def __init__(self):
            self.compile = lambda *args, **kwargs: args[0] if args else None
            self.cudagraph = False
    torch.compiler = MockCompiler()

# Mock dynamo if needed
if not hasattr(torch, "_dynamo"):
    class MockDynamo:
        def __init__(self):
            self.is_compiling = lambda: False
            self.reset = lambda: None
            self.optimize = lambda *args, **kwargs: args[0] if args else None
    torch._dynamo = MockDynamo()

from PIL import Image
import numpy as np
import cv2
from transformers import CLIPProcessor, CLIPModel
from segment_anything import sam_model_registry, SamPredictor
import streamlit as st  # Import streamlit last

# Load SAM
def load_sam_model():
    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
    predictor = SamPredictor(sam)
    return predictor

# Load CLIP
@st.cache_resource  # Add caching to avoid reloading models
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