import streamlit as st
from PIL import Image
import torch
import numpy as np
import pickle
import faiss
import os
from utils import load_clip_model, get_clip_embedding
import plotly.graph_objects as go
from io import BytesIO
import pandas as pd
import json

# Disable Streamlit's file watcher to prevent unnecessary re-runs
os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"

# Disable PyTorch JIT for Streamlit Cloud compatibility
if hasattr(torch.jit, "_enabled"):
    torch.jit._enabled = False
if hasattr(torch.jit, "disable_jit"):
    torch.jit.disable_jit()

# Explicitly set the directory for the checkpoint file
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["SAM_CHECKPOINT_PATH"] = os.path.join(MODEL_DIR, "sam_vit_b_01ec64.pth")

# Import sam_utils with proper error handling
try:
    from sam_utils import get_card_crops
    sam_loaded = True
    st.sidebar.success("🎮 SAM model loaded successfully! Automatic card detection available.")
except Exception as e:
    sam_loaded = False
    st.sidebar.error(f"Error loading SAM model: {e}")
    st.sidebar.info("Using fallback mode without automatic card detection. Please upload cropped card images.")

# Load card database with price history and parse JSON
def load_card_db():
    card_db = pd.read_csv("cards.csv")
    # Ensure price_history is parsed correctly as a dictionary
    card_db["price_history"] = card_db["price_history"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )
    return card_db

# Update the existing card_db loading logic
@st.cache_resource
def load_index():
    try:
        index = faiss.read_index("clip.index")
        card_db = load_card_db()
        return index, card_db
    except FileNotFoundError:
        st.error("Required files (clip.index or card_db.pkl) are missing. Please upload them.")
        return None, None


# Function to generate an interactive price history chart using Plotly
def generate_price_chart(card_name, card_db):
    card = card_db[card_db["name"] == card_name]
    if card.empty:
        return None

    price_history = card.iloc[0]["price_history"]
    dates = price_history["dates"]
    prices = price_history["prices"]

    # Ensure all data points are included in the chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode='lines',  # Use smooth lines without markers
        line=dict(shape='spline',color='orange', width=2),  # Set line color to orange
        name='Price',
        hovertemplate='<b>Date:</b> %{x}<br><b>Price:</b> $%{y}<extra></extra>'
    ))
    fig.update_layout(
        title={
            'text': f"Price History: {card_name}",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Date",
        yaxis_title="Price ($)",
        xaxis=dict(showgrid=True, gridcolor='lightgrey', tickformat='%b %d, %Y'),
        yaxis=dict(showgrid=True, gridcolor='lightgrey'),
        template="plotly_white",
        height=400,
        width=400,  # Set the chart width to 400
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified"
    )
    return fig

# Function to display card details in a list style
def display_card_details(card):
    # Add a title for the card details
    st.markdown("### Card Details")

    # Display each card detail with the attribute in orange
    st.markdown(f"<span style='color: orange; font-weight: bold;'>Card Name:</span> {card['name']}", unsafe_allow_html=True)
    st.markdown(f"<span style='color: orange; font-weight: bold;'>Set:</span> {card['set']}", unsafe_allow_html=True)
    st.markdown(f"<span style='color: orange; font-weight: bold;'>Current Market Price:</span> CAD$ {card['value']}", unsafe_allow_html=True)
    st.markdown(f"<span style='color: orange; font-weight: bold;'>Condition:</span> {card['condition']}", unsafe_allow_html=True)

# Load everything at startup
st.set_page_config(layout="wide")
st.title("TCG Card Recognition and Matching Tool")
st.markdown("""
### Powered by:
- **:orange[SAM (Segment Anything Model)]**: Used for precise segmentation and cropping of cards from the uploaded image.
- **:orange[CLIP (Contrastive Language–Image Pretraining)]**: Utilized for generating embeddings to match card images with the database.
- **:orange[FAISS (Facebook AI Similarity Search)]**: Enables efficient similarity search to find the closest match in the card database.

### Key Features:
- **:orange[Multi-Item Detection]**: Capable of detecting and processing multiple cards in a single image.
- **:orange[Versatile Product Matching]**: Supports recognition and matching of various types of trading cards and collectibles.
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
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Display the uploaded image
    cols = st.columns(2)
    with cols[0]:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # Segment and crop cards using SAM
    with cols[1]:
        with st.container():
            st.write("\n" * 10)  # Add vertical spacing to center the spinner
            cols = st.columns([1, 2, 1])  # Add horizontal spacing with columns
            with cols[1]:
                with st.spinner("#### 🪄 Segmenting cards..."):
                    crops = get_card_crops(image)
        if os.path.exists("all_masks_with_info.png"):
            masks_image = Image.open("all_masks_with_info.png")
            st.image(masks_image, caption="All Masks with Info", use_column_width=True)
        else:
            st.warning("⚠️ Masks visualization not found.")

    num_crops = len(crops)
    st.write(f"#### 📦 Detected {num_crops} card(s)")

    if num_crops > 0:
        for i, crop in enumerate(crops):
            # Create two columns for the images (cropped card and matched card)
            img_col1, img_col2 = st.columns(2)

            with img_col1:
                st.write(f"**Card {i+1}**")
                st.image(crop, use_column_width=True)

            try:
                # Get the embedding and search for the match
                emb = get_clip_embedding(crop.resize((224, 224)), clip_model, clip_processor)
                D, I = index.search(np.array([emb], dtype="float32"), k=1)
                match = card_db.iloc[I[0][0]]

                with img_col2:
                    st.write(f"matched with {match['name']}")
                    st.image(match['image_path'], use_column_width=True)

                # Create two columns for card details and price history chart
                details_col, chart_col = st.columns(2)

                with details_col:
                    display_card_details(match)

                with chart_col:
                    try:
                        chart_fig = generate_price_chart(match['name'], card_db)
                        if chart_fig:
                            st.plotly_chart(chart_fig, use_container_width=True)
                        else:
                            st.warning("⚠️ No price history available.")
                    except Exception as e:
                        st.error(f"❌ Chart generation failed: {e}")

            except Exception as e:
                st.error(f"❌ Match failed: {e}")
    else:
        st.warning("⚠️ No cards detected. Please try another image.")