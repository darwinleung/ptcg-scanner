import os
import pandas as pd
import numpy as np
import faiss
import pickle
from PIL import Image
import torch
import transformers
import warnings
warnings.filterwarnings("ignore")

# 🔧 Limit CPU threading to avoid segfaults
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import CLIPProcessor, CLIPModel

# 🧠 Load model and processor (force CPU)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cpu()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_clip_embedding(image):
    inputs = processor(images=image, return_tensors="pt").to("cpu")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    emb = outputs[0]
    emb = emb / emb.norm(p=2)
    return emb.cpu().numpy()

# 🗂️ Load metadata
df = pd.read_csv("cards.csv")
print(f"📄 Loaded {len(df)} rows from cards.csv")

embeddings = []
valid_rows = []

# 🌀 Process each image one by one
for i, row in df.iterrows():
    path = row["image_path"]
    print(f"🔍 Processing {i+1}/{len(df)}: {path}")

    if not os.path.exists(path):
        print(f"[❌] Not found: {path}")
        continue

    try:
        img = Image.open(path).convert("RGB").resize((224, 224))
        emb = get_clip_embedding(img)

        if emb.shape != (512,):
            print(f"[⚠️] Skipping due to unexpected shape: {emb.shape}")
            continue

        embeddings.append(emb)
        valid_rows.append(row)

        # 🔥 Clear memory manually per loop
        del img, emb
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"[⚠️] Failed on {path}: {e}")

# 🚧 Final check
if not embeddings:
    raise RuntimeError("No valid embeddings. Aborting.")

# 💾 Save index
print("📦 Building FAISS index...")
embeddings_np = np.vstack(embeddings).astype("float32")
index = faiss.IndexFlatL2(512)
index.add(embeddings_np)
faiss.write_index(index, "clip.index")
print("✅ Saved FAISS index")

# 💾 Save metadata
valid_df = pd.DataFrame(valid_rows)
with open("card_db.pkl", "wb") as f:
    pickle.dump(valid_df, f)
print("✅ Saved metadata to card_db.pkl")


# # --- one-liner test ---

# from PIL import Image
# from utils import load_clip_model, get_clip_embedding

# model, processor = load_clip_model()
# img = Image.open("data/pikachu.png").convert("RGB").resize((224, 224))
# emb = get_clip_embedding(img, model, processor)
# print(emb.shape)

# # ---

# import numpy as np

# # Example with 1 embedding
# import pickle

# embedding = emb.reshape(1, -1).astype("float32")  # emb from your previous test
# print(embedding.shape)

# import faiss
# index = faiss.IndexFlatL2(embedding.shape[1])
# index.add(embedding)
# faiss.write_index(index, "clip_test.index")

# print("✅ FAISS index built and saved.")