import streamlit as st
from PIL import Image
import torch
import numpy as np
import pickle
import faiss
import os
from utils import load_clip_model, get_clip_embedding
from sam_utils import get_card_crops

# Load everything at startup
st.set_page_config(layout="wide")
st.title("Multi-item image recognition scanner")
st.text("This is powered by SAM, CLIP, and FAISS, specifically trained on Pokémon cards.")

@st.cache_resource
def load_index():
    index = faiss.read_index("clip.index")
    with open("card_db.pkl", "rb") as f:
        card_db = pickle.load(f)
    return index, card_db

@st.cache_resource
def load_models():
    return load_clip_model()

index, card_db = load_index()
clip_model, clip_processor = load_models()

# Upload image
uploaded_file = st.file_uploader("Upload an image with multiple cards", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    # st.write("### Uploaded Image")
    # st.image(image, caption="Uploaded Image", use_column_width=True)


    # Display the updated all_masks_with_info.png
    st.write("### Masks Visualization")
    cols = st.columns(2)

    with cols[0]:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # # Segment and crop cards using SAM
    # st.write("🪄 Segmenting cards...")
    # crops = get_card_crops(image)

    

    with cols[1]:
        # Segment and crop cards using SAM
        st.write("🪄 Segmenting cards...")
        crops = get_card_crops(image)
        if os.path.exists("all_masks_with_info.png"):
            masks_image = Image.open("all_masks_with_info.png")
            st.image(masks_image, caption="All Masks with Info", use_column_width=True)
        else:
            st.warning("⚠️ Masks visualization not found.")

    num_crops = len(crops)
    st.write(f"📦 Detected {num_crops} card(s)")

    if num_crops > 0:
        cols = st.columns(num_crops)

        for i, crop in enumerate(crops):
            with cols[i]:
                st.image(crop, caption=f"Card {i+1 }", width=180)

                try:
                    emb = get_clip_embedding(crop.resize((224, 224)), clip_model, clip_processor)
                    D, I = index.search(np.array([emb], dtype="float32"), k=1)
                    match = card_db.iloc[I[0][0]]
                    st.success(f"✅ {match['name']} ({match['set']}) - ${match['value']}")
                except Exception as e:
                    st.error(f"❌ Match failed: {e}")
    else:
        st.warning("⚠️ No cards detected. Please try another image.")