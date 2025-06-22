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

# user id and asin to int id mapping
asin_to_id = {asin: int(item_id) for asin, item_id in id_map["item_mapping"].items()} # asin (str) → item_id (int)
item_id_to_asin = {int(item_id): asin for item_id, asin in id_map["item_reverse_mapping"].items()} # item_id (int) → asin (str)

all_categories = sorted(meta_df["main_category"].dropna().unique().tolist())
category_options = ["Any"] + all_categories


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
def pick_random_user_simple(min_interactions, max_interactions):
    counts = train_df["user_id"].value_counts()
    if max_interactions <= 0:
        valid = counts[counts >= min_interactions].index.tolist()
    else:
        valid = counts[min_interactions <= counts <= max_interactions].index.tolist()
    return random.choice(valid), len(valid), len(counts) if valid else None

def pick_random_user_complex(min_interactions, max_interactions, selected_category, category_interactions_required):
    eligible_users = []

    # Precompute product_id → category mapping
    item_id_to_category = {
        iid: meta_df.loc[asin]["main_category"]
        for iid, asin in item_id_to_asin.items()
        if asin in meta_df.index and pd.notna(meta_df.loc[asin]["main_category"])
    }

    grouped = train_df.groupby("user_id")
    total_users = len(grouped)
    for user_id, group in grouped:
        if len(group) < min_interactions:
            continue

        if max_interactions <= 0 and len(group) > max_interactions:
            continue

        if selected_category != "Any":
            item_ids = group["item_id"].tolist()
            user_categories = [item_id_to_category.get(iid) for iid in item_ids]
            user_category_count = sum(1 for cat in user_categories if cat == selected_category)
            if user_category_count < category_interactions_required:
                continue

        eligible_users.append(user_id)

    if len(eligible_users) == 0:
        return None, 0, total_users

    return random.choice(eligible_users), len(eligible_users), total_users if eligible_users else None

def pick_random_user(min_interactions, max_interactions, selected_category, category_interactions_required):
    if selected_category == "Any":
        return pick_random_user_simple(min_interactions, max_interactions)
    else:
        return pick_random_user_complex(min_interactions, max_interactions, selected_category, category_interactions_required)

# --- Gradio Functions ---
def run_dashboard(min_interactions, max_interactions, category_dropdown, category_min_count):
    # user_id = pick_random_user(min_interactions)
    user_id, found_count, total_users = pick_random_user(
        min_interactions=min_interactions,
        max_interactions=max_interactions,
        selected_category=category_dropdown,
        category_interactions_required=category_min_count
    )
    user_label = f"🔍 Matched users: {found_count} / {total_users}"
    if user_id is None:
        return user_label + "⚠️ No user found with given filters.", [], []

    # history_ids = train_df[train_df["user_id"] == user_id].sort_values("timestamp", ascending=False)["item_id"].tolist()
    # rec_ids = recommend_for_user(user_id)
    # return str(user_id), build_cards(history_ids[:10]), build_cards(rec_ids)

    history_items = train_df[train_df["user_id"] == user_id].sort_values("timestamp", ascending=False)["item_id"].tolist()
    recommended_items = recommend_for_user(user_id)

    history_cards = build_cards(history_items)
    recommendation_cards = build_cards(recommended_items)

    return f"{user_label}\n✅ Selected User ID: {user_id}", history_cards, recommendation_cards

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Recommender Dashboard with Images")
    with gr.Row():
        min_input = gr.Number(label="Min Past Interactions", value=5)
        max_input = gr.Number(label="Max Past Interactions (zero for no limit)", value=0, precision=1)
    with gr.Row():
        category_dropdown = gr.Dropdown(label="Main Category Filter (Any for no filtering - much faster)", choices=category_options, value="Any")
        category_min_count = gr.Number(label="Min Interactions With Category", value=1, precision=0)
        run_button = gr.Button("Pick Random User")

    user_out = gr.Textbox(label="User ID")
    with gr.Column():
        gr.Markdown("### 🔁 Past Interactions")
        hist_gallery = gr.Gallery(label="History", columns=5, height="auto")
    with gr.Column():
        gr.Markdown("### 🌟 Recommendations")
        rec_gallery = gr.Gallery(label="Recommendations", columns=5, height="auto")

    run_button.click(run_dashboard,
                     inputs=[min_input, max_input, category_dropdown, category_min_count],
                     outputs=[user_out, hist_gallery, rec_gallery])

demo.launch()
