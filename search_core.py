import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter
from keras.applications.mobilenet_v2 import preprocess_input


# ---------- Task 2: model input and embedding ----------

def to_model_input(images_28, invert=False):
    # Prepare grayscale images for MobileNetV2
    images = np.array(images_28, dtype="float32")
    if invert:
        images = 255.0 - images
    images = np.expand_dims(images, axis=-1)
    images = tf.image.resize(images, (160, 160)).numpy()
    images = np.repeat(images, 3, axis=-1)
    images = preprocess_input(images)
    return images


def embed(model, images_28, invert=False, batch_size=64):
    features = []
    for i in range(0, len(images_28), batch_size):
        batch = images_28[i:i + batch_size]
        batch = to_model_input(batch, invert=invert)
        predictions = model.predict(batch, verbose=0)
        features.append(predictions)

    features = np.concatenate(features, axis=0)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    return features


# ---------- Task 3: make every query look like the gallery ----------

def to_gray(image):
    if hasattr(image, "mode") and image.mode not in ("RGB", "RGBA", "L"):
        image = image.convert("RGBA")
        
    img = np.asarray(image).astype("float32")

    if img.ndim == 2:
        return img

    if img.shape[2] == 4:
        alpha = img[..., 3:4] / 255.0
        img = img[..., :3] * alpha + 255.0 * (1.0 - alpha)
    else:
        img = img[..., :3]

    return img @ np.array([0.299, 0.587, 0.114], dtype="float32")


def get_strokes(gray, threshold=128):
    return ((gray < threshold) * 255).astype("uint8")


def crop_square(strokes, margin=0.1):
    ys, xs = np.where(strokes > 0)
    if len(ys) == 0:
        return None

    cropped = strokes[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = cropped.shape

    side = max(h, w)
    pad = int(side * margin)
    out = np.zeros((side + 2 * pad, side + 2 * pad), dtype=strokes.dtype)
    top = (out.shape[0] - h) // 2
    left = (out.shape[1] - w) // 2
    out[top:top + h, left:left + w] = cropped
    return out


def resize_28(squared):
    img = Image.fromarray(squared)
    out = np.array(img.resize((28, 28), Image.BOX))

    if out.max() < 230:
        size = max(3, (squared.shape[0] // 28) | 1)
        thick = img.filter(ImageFilter.MaxFilter(size))
        out = np.array(thick.resize((28, 28), Image.BOX))

    return (out.astype("float32") / out.max() * 255).astype("uint8")


def preprocess_query(image, threshold=128):
    gray = to_gray(image)
    strokes = get_strokes(gray, threshold)
    squared = crop_square(strokes)
    if squared is None:
        return None
    return resize_28(squared)


# ---------- Task 4: search ----------

def search(query_vector, embeddings, k):
    query_vector = np.array(query_vector).flatten()
    scores = np.dot(embeddings, query_vector)
    sorted_indices = np.argsort(scores)[::-1]
    top_indices = sorted_indices[:k]
    top_scores = scores[top_indices]
    return top_indices, top_scores
