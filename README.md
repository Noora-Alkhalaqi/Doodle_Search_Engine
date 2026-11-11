# Doodle Search Engine

A visual search engine for hand-drawn doodles. Draw a sketch and the app finds the 25 most similar doodles in a gallery of 6,750 images from [Google Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset).

**Live demo:** https://doodlesearchengine.streamlit.app/

Search works by comparing image embeddings from a pretrained **MobileNetV2** (ImageNet weights) using cosine similarity. The model is not fine-tuned.

## Features

- **Drawing canvas**: sketch directly in the browser and results update as you draw.
- **Query preprocessing**: every drawing is converted to the same format as the gallery (white strokes on black, cropped, centered, 28×28).
- **"What the model sees" preview**: the app shows the processed 28×28 query.
- **Similarity threshold**: if the best match scores below the configured threshold, the app shows "Nothing similar found" instead of weak results.

## Dataset

The gallery comes from the Quick, Draw! 28×28 numpy bitmaps. It covers 15 categories:

cat, dog, sun, smiley face, moon, cup, bucket, apple, tree, house, car, bicycle, clock, flower, fish

For each category, 500 drawings are sampled with a fixed seed (42). Of these, 450 go into the gallery (15 × 450 = 6,750) and 50 are held out as test queries, which are used to calibrate the similarity threshold.

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
├── app.py                    # Streamlit UI: model/index loading, drawing canvas, results grid
├── search_core.py            # Preprocessing, embedding, and search functions
├── doodleSearchEngine.ipynb  # Notebook: data download, embedding, threshold calibration
├── requirements.txt
└── index/
    ├── config.json         # Search settings (threshold, top_k, invert)
    ├── embeddings.npy      # (6750, 1280) L2-normalized gallery embeddings
    ├── gallery_images.npy  # (6750, 28, 28) gallery doodles
    └── gallery_labels.npy  # (6750,) class labels (15 classes)
```

## Setup

Requires Python 3.9+.

```bash
pip install -r requirements.txt
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
| `threshold` | `0.8012` | Minimum cosine similarity of the top match for results to be shown. Set to the 5th percentile of the top-1 scores of the held-out queries. |
| `top_k`     | `25`     | Number of results returned, shown in rows of 5 (a 5×5 grid by default). |
| `invert`    | `false`  | Whether to invert pixel values before embedding. Must match how the gallery embeddings were built. |

## Rebuilding the index

The `index/` files were produced ahead of time by [doodleSearchEngine.ipynb](doodleSearchEngine.ipynb). The notebook was written for Google Colab and:

1. Downloads the 15 categories from the Quick, Draw! dataset into `data/raw/`.
2. Samples and splits the drawings into the gallery and held-out queries.
3. Embeds the gallery with MobileNetV2 and saves `embeddings.npy`, `gallery_images.npy` and `gallery_labels.npy`.
4. Embeds the held-out queries, sets `threshold` to the 5th percentile of their top-1 similarity scores, and writes `config.json`.
5. Zips `index/` and downloads it (Colab only; skip this cell when running locally).

To build the index from your own 28×28 gallery (white strokes on black, `uint8`), use the same `embed` function so queries and gallery are embedded the same way:

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
- Draw strokes large enough to fill most of the canvas. Small details are lost when the drawing is shrunk to 28×28.
- Check the "what the model sees" preview. If it looks wrong, the matches will be too.
