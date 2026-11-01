# Doodle Search Engine

A visual search engine for hand-drawn doodles. Draw a sketch, upload a picture, or take a photo of a drawing on paper, and the app finds the 25 most similar doodles in a gallery of 6,750 images.

Search works by comparing image embeddings from a pretrained **MobileNetV2** (ImageNet weights) using cosine similarity. The model is not fine-tuned.

## Features

- **Three input modes**: a drawing canvas, image upload (PNG/JPG), and camera capture.
- **Query preprocessing**: every input is converted to the same format as the gallery (white strokes on black, cropped, centered, 28×28), so a phone photo of a pen sketch can be compared with canvas drawings.
- **"What the model sees" preview**: the app shows the processed 28×28 query next to your input.
- **Similarity threshold**: if the best match scores below the configured threshold, the app shows "Nothing similar found" instead of weak results.

## How it works

1. **Preprocess the query** (`preprocess_query` in [search_core.py](search_core.py))
   - Convert to grayscale. Transparent pixels are composited onto white.
   - Threshold to a binary stroke mask: dark pixels become white strokes on a black background.
   - Crop to the bounding box of the strokes, pad to a square with a 10% margin, and center.
   - Downsample to 28×28. If the strokes become too faint at that size, they are thickened with a max filter first.
2. **Embed** (`embed`): resize to 160×160, repeat to 3 channels, apply MobileNetV2 preprocessing, run the model with global average pooling to get a 1280-dimensional vector, and L2-normalize it.
3. **Search** (`search`): take the dot product with all gallery embeddings (cosine similarity, since all vectors are normalized) and return the top-k matches.

## Project structure

```
Doodle_Search_Engine/
├── app.py              # Streamlit UI: model/index loading, input tabs, results grid
├── search_core.py      # Preprocessing, embedding, and search functions
└── index/
    ├── config.json         # Search settings (threshold, top_k, invert)
    ├── embeddings.npy      # (6750, 1280) L2-normalized gallery embeddings
    ├── gallery_images.npy  # (6750, 28, 28) gallery doodles
    └── gallery_labels.npy  # (6750,) class labels (15 classes)
```

## Setup

Requires Python 3.9+.

```bash
pip install streamlit streamlit-drawable-canvas tensorflow numpy pillow
```

The MobileNetV2 ImageNet weights are downloaded automatically by Keras on first run.

## Running the app

Run from the project root so the `index/` folder is found:

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## Configuration

`index/config.json`:

| Key         | Default  | Description                                                           |
|-------------|----------|-----------------------------------------------------------------------|
| `threshold` | `0.8012` | Minimum cosine similarity of the top match for results to be shown.    |
| `top_k`     | `25`     | Number of results returned, shown in rows of 5 (a 5×5 grid by default). |
| `invert`    | `false`  | Whether to invert pixel values before embedding. Must match how the gallery embeddings were built. |

## Rebuilding the index

The `index/` files were produced ahead of time. To rebuild them with your own 28×28 gallery (white strokes on black, `uint8`), use the same `embed` function so queries and gallery are embedded the same way:

```python
import numpy as np
from keras.applications import MobileNetV2
from search_core import embed

model = MobileNetV2(input_shape=(160, 160, 3), include_top=False,
                    weights="imagenet", pooling="avg")

embeddings = embed(model, gallery_images, invert=False)
np.save("index/embeddings.npy", embeddings)
np.save("index/gallery_images.npy", gallery_images)
np.save("index/gallery_labels.npy", gallery_labels)
```

If you change the gallery, you may also need to adjust `threshold` in `config.json`.

## Tips for better results

- Draw a single, clear object with dark strokes on a light background.
- For photos, use plain paper and even lighting. Shadows darker than the threshold (128) show up as strokes.
- Check the "what the model sees" preview. If it looks wrong, the matches will be too.
