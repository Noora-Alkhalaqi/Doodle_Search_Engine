import json

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from keras.applications import MobileNetV2
from keras.layers import Input, GlobalAveragePooling2D
from keras.models import Model
from streamlit_drawable_canvas import st_canvas

from search_core import preprocess_query, embed, search

st.set_page_config(page_title="Doodle Search", layout="wide")


# ---------- 1. Model loading (once) ----------
@st.cache_resource
def load_model():
    backbone = MobileNetV2(
        input_shape=(160, 160, 3),
        include_top=False,
        weights="imagenet"
    )
    backbone.trainable = False

    inputs = Input(shape=(160, 160, 3))
    x = backbone(inputs)
    outputs = GlobalAveragePooling2D()(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


# ---------- 2. Index loading (once) ----------
@st.cache_resource
def load_index():
    embeddings = np.load("index/embeddings.npy")
    gallery_images = np.load("index/gallery_images.npy")
    with open("index/config.json") as f:
        config = json.load(f)
    return embeddings, gallery_images, config


model = load_model()
embeddings, gallery_images, config = load_index()


def enlarge(img_28, size=112):
    return np.array(Image.fromarray(img_28).resize((size, size), Image.NEAREST))


# ---------- 3. Inference + rendering ----------
def run_search(image, show_input=None):
    query = preprocess_query(image)

    left, right = st.columns(2)
    if show_input is not None:
        left.image(show_input, caption="your input", width=280)

    if query is None:
        right.info("Nothing drawn yet.")
        return

    right.image(enlarge(query, 280), caption="what the model sees (28x28)", width=280)

    vec = embed(model, query[np.newaxis], invert=config.get("invert", False))[0]
    k = min(config.get("top_k", 25), len(embeddings))
    if k < 1:
        st.error("The gallery is empty.")
        return

    idx, scores = search(vec, embeddings, k)

    if scores[0] < config.get("threshold", -1.0):
        st.warning("Nothing similar found")
        return

    st.subheader(f"Top {k} most similar doodles")
    for start in range(0, k, 5):
        cols = st.columns(5)
        for c in range(min(5, k - start)):
            j = start + c
            cols[c].image(
                enlarge(gallery_images[idx[j]]),
                caption=f"{scores[j]:.2f}",
                width=112
            )


# ---------- 4. UI ----------
st.title("Doodle Search")

tab_draw, tab_upload, tab_camera = st.tabs(["Draw", "Upload", "Camera"])

with tab_draw:
    canvas_args = dict(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=8,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=280,
        width=280,
        drawing_mode="freedraw",
        update_streamlit=True,
        key="canvas",
    )
    try:
        canvas = st_canvas(**canvas_args, return_image_data=True)
    except TypeError:
        canvas = st_canvas(**canvas_args)

    if canvas.image_data is not None:
        run_search(canvas.image_data.astype("uint8"))
