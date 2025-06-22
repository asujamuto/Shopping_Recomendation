import gradio as gr
import pandas as pd
import numpy as np
import json
import ast
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load data ---
train_df = pd.read_csv("../data/train.csv")
meta_df = pd.read_csv("../data/item_metadata_filtered.csv")
with open("../data/id_mappings.json") as f:
    id_map = json.load(f)

asin_to_id = {asin: int(item_id) for asin, item_id in id_map["item_mapping"].items()} # asin (str) → item_id (int)
item_id_to_asin = {int(item_id): asin for item_id, asin in id_map["item_reverse_mapping"].items()} # item_id (int) → asin (str)
# item_reverse_map = {k: k for k, v in id_map["item_reverse_mapping"].items()}  # int item_id → asin
# item_id_to_asin = {v: k for k, v in item_reverse_map.items()}
# asin_to_id = {v: k for k, v in id_map["item_mapping"].items()}  # asin → item_id

# Preprocess metadata
meta_df["title"] = meta_df["title"].fillna("")
meta_df["store"] = meta_df["store"].fillna("")
meta_df["description"] = meta_df["description"].fillna("")
meta_df["average_rating"] = meta_df["average_rating"].fillna("")
meta_df["price"] = meta_df["price"].fillna("")
meta_df["image_urls"] = meta_df["image_urls"].fillna("[]")

# Index by ASIN for lookup
meta_df = meta_df.set_index("parent_asin")

# --- TF-IDF ---
all_items = sorted(asin_to_id.keys())  # all ASINs
texts = [
    meta_df.loc[asin]["title"] + " " +
    meta_df.loc[asin]["store"] + " " +
    meta_df.loc[asin]["description"]
    if asin in meta_df.index else ""
    for asin in all_items
]

vectorizer = TfidfVectorizer(max_features=10_000)
tfidf_matrix = vectorizer.fit_transform(texts)

# --- Recommender ---
def build_user_profile(user_id):
    user_ratings = train_df[train_df["user_id"] == user_id]
    indices = user_ratings["item_id"].values
    ratings = user_ratings["rating"].values
    item_vectors = tfidf_matrix[indices]
    weighted = item_vectors.multiply(ratings[:, None])
    profile = weighted.mean(axis=0)
    return np.asarray(profile)

def recommend_for_user(user_id, top_k=10):
    profile = build_user_profile(user_id)
    scores = cosine_similarity(profile, tfidf_matrix).flatten()
    seen_items = set(train_df[train_df["user_id"] == user_id]["item_id"])
    recs = [i for i in scores.argsort()[::-1] if i not in seen_items]
    return recs[:top_k]

# --- Display helpers ---
def get_product_card(item_id):
    asin = item_id_to_asin.get(item_id)
    if not asin or asin not in meta_df.index:
        return None
    row = meta_df.loc[asin]
    try:
        img_url = ast.literal_eval(row["image_urls"])[0]
    except:
        img_url = None
    name = row["title"]
    store = row["store"]
    rating = row["average_rating"]
    price = row["price"]
    caption = f"{name}\n{store}\n⭐ {rating}" + (f" – ${price}" if price else "")
    # return gr.Image(value=img_url, label=caption, width=200, height=200)
    return (img_url, caption)

def build_cards(item_ids):
    cards = []
    for iid in item_ids:
        card = get_product_card(iid)
        if card:
            cards.append(card)
    return cards

# --- User Picker ---
def pick_random_user(min_interactions):
    counts = train_df["user_id"].value_counts()
    valid = counts[counts >= min_interactions].index.tolist()
    return random.choice(valid)

# --- Gradio Functions ---
def run_dashboard(min_interactions):
    user_id = pick_random_user(min_interactions)
    history_ids = train_df[train_df["user_id"] == user_id].sort_values("timestamp", ascending=False)["item_id"].tolist()
    rec_ids = recommend_for_user(user_id)
    return str(user_id), build_cards(history_ids[:10]), build_cards(rec_ids)

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Recommender Dashboard with Images")
    with gr.Row():
        min_input = gr.Number(label="Min Past Interactions", value=5)
        run_button = gr.Button("Pick Random User")

    user_out = gr.Textbox(label="User ID")
    with gr.Column():
        gr.Markdown("### 🔁 Past Interactions")
        hist_gallery = gr.Gallery(label="History", columns=5, height="auto")
    with gr.Column():
        gr.Markdown("### 🌟 Recommendations")
        rec_gallery = gr.Gallery(label="Recommendations", columns=5, height="auto")

    run_button.click(run_dashboard, inputs=min_input, outputs=[user_out, hist_gallery, rec_gallery])

demo.launch()
