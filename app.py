import streamlit as st
from PIL import Image
import torch
import numpy as np
import pickle
import faiss
import os
from utils import load_clip_model, get_clip_embedding
from sam_utils import get_card_crops
import plotly.graph_objects as go
from io import BytesIO
import pandas as pd

# Load card database with price history
def load_card_db():
    card_db = pd.read_csv("cards.csv")
    card_db["price_history"] = card_db["price_history"].apply(lambda x: eval(x))  # Convert JSON string to list
    return card_db

# Update the existing card_db loading logic
@st.cache_resource
def load_index():
    index = faiss.read_index("clip.index")
    card_db = load_card_db()
    return index, card_db


# Function to generate an interactive price history chart using Plotly
def generate_price_chart(card_name, card_db):
    card = card_db[card_db["name"] == card_name]
    if card.empty:
        return None

    prices = card.iloc[0]["price_history"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(prices) + 1)),
        y=prices,
        mode='lines+markers',
        line=dict(color='blue'),
        marker=dict(size=8),
        name='Price'
    ))
    fig.update_layout(
        title=f"Price History: {card_name}",
        xaxis_title="Time",
        yaxis_title="Price ($)",
        template="plotly_white",
        height=300
    )
    return fig

# Load everything at startup
st.set_page_config(layout="wide")
st.title("TCG Card Recognition and Matching Tool")
st.markdown("""
### Powered by:
- **SAM (Segment Anything Model)**: Used for precise segmentation and cropping of cards from the uploaded image.
- **CLIP (Contrastive Language–Image Pretraining)**: Utilized for generating embeddings to match card images with the database.
- **FAISS (Facebook AI Similarity Search)**: Enables efficient similarity search to find the closest match in the card database.

### Key Features:
- **Multi-Item Detection**: Capable of detecting and processing multiple cards in a single image.
- **Versatile Product Matching**: Supports recognition and matching of various types of trading cards and collectibles.
""")

# @st.cache_resource
# def load_index():
#     index = faiss.read_index("clip.index")
#     with open("card_db.pkl", "rb") as f:
#         card_db = pickle.load(f)
#     return index, card_db

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
    cols = st.columns(2)

    with cols[0]:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # # Segment and crop cards using SAM
    # st.write("🪄 Segmenting cards...")
    # crops = get_card_crops(image)
    with cols[1]:
        # Segment and crop cards using SAM
        # Center the spinner and segmenting cards message horizontally and vertically
        with st.container():
            st.write("\n" * 10)  # Add vertical spacing to center the spinner
            cols = st.columns([1, 2, 1])  # Add horizontal spacing with columns
            with cols[1]:
                with st.spinner("🪄 Segmenting cards..."):
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

                    # Generate and display interactive price history chart
                    chart_fig = generate_price_chart(match['name'], card_db)
                    if chart_fig:
                        st.plotly_chart(chart_fig, use_container_width=True)
                    else:
                        st.warning("⚠️ No price history available.")
                except Exception as e:
                    st.error(f"❌ Match failed: {e}")
    else:
        st.warning("⚠️ No cards detected. Please try another image.")