# VAAS Usage and API

VAAS provides an inference-first interface for image anomaly detection and localisation.

The library exposes a unified pipeline with APIs for:

* Standard inference
* Visualisation
* Detailed outputs
* Patch-level analysis
* Feature extraction
* Batch processing

All APIs accept either:

* `PIL.Image`
* image file path (`str`)

---

## Installation

```bash
pip install vaas
```

VAAS requires PyTorch at runtime:

https://pytorch.org/get-started/locally/   Or   install available latest version `pip install -q torch torchvision`

---

## Loading the Pipeline

```python
from vaas.inference.pipeline import VAASPipeline

pipeline = VAASPipeline.from_pretrained(
    repo_id="OBA-Research/vaas",
    model_variant="v2-base-df2023",
    device="cpu",   # or "cuda"
    alpha=0.5,
)
```

---

## Standard Inference

```python
result = pipeline(image)
```

### Output

```python
{
  "S_F": float,
  "S_P": float,
  "S_H": float,
  "anomaly_map": numpy.ndarray  # shape (224, 224)
}
```

### Description

* `S_F` — global attention anomaly score
* `S_P` — patch-level anomaly score
* `S_H` — hybrid anomaly score
* `anomaly_map` — dense localisation map

---

## Visualisation

```python
pipeline.visualize(
    image=image,
    save_path="output.png",
    mode="all",
    threshold=0.5,
)
```

### Modes

* `"all"` — full composite
* `"px"` — patch heatmap
* `"fx"` — attention map
* `"binary"` — thresholded mask

### Output

Saved figure containing:

* original image
* anomaly overlays
* hybrid score

---

## Detailed Inference

```python
out = pipeline.forward_detailed(image)
```

### Output

```python
{
  "S_F": float,
  "S_P": float,
  "S_H": float,
  "anomaly_map": numpy.ndarray,
  "variant": str,
  "metadata": dict
}
```

### Use Cases

* experiment tracking
* reproducibility
* model inspection

---

## Patch-Level Scores

```python
scores = pipeline.extract_patch_scores(image)
```

### Output

```python
numpy.ndarray  # shape (224, 224)
```

### Description

* Direct patch-level anomaly probabilities
* Equivalent to segmentation output

### Use Cases

* region extraction
* anomaly segmentation
* threshold-based filtering

---

## Feature Extraction

```python
features = pipeline.extract_features(image)
```

### Output

```python
{
  "fx_cls_embedding": ndarray,
  "fx_patch_tokens": ndarray,
  "px_cross_attention_tokens": ndarray,
}
```

### Description

* `fx_cls_embedding` — global representation
* `fx_patch_tokens` — ViT patch tokens
* `px_cross_attention_tokens` — cross-attended Px tokens

### Use Cases

* retrieval
* clustering
* representation analysis
* downstream modelling

---

## Batch Processing

```python
results = pipeline.batch(images)
```

### Input

```python
images = [img1, img2, ...]
```

### Output

```python
List[Dict]
```

Each element follows the standard inference output format.

### Use Cases

* dataset scanning
* ranking images by anomaly
* pipeline integration

---

## Example: Ranking by Anomaly Score

```python
results = pipeline.batch(images)

ranked = sorted(
    results,
    key=lambda x: x["S_H"],
    reverse=True,
)
```

---

## Model Variants

```python
model_variant = "v2-base-df2023"
```

Available variants:

* `v2-base-df2023`
* `v2-medium-df2023`
* `v2-large-df2023`

---

## Notes

* PyTorch is loaded lazily at runtime
* CPU inference is supported
* GPU improves performance but is optional

---

## Recommended Usage

### For practitioners

* `pipeline(image)`
* `pipeline.visualize(...)`

### For researchers

* `forward_detailed`
* `extract_patch_scores`
* `extract_features`
* `batch`

---

## Colab Notebooks

See the `examples/notebooks/` directory for:

* Quick start (v2)
* Model comparison
* Detailed outputs
* Patch-level analysis
* Feature extraction and batch workflows
* Structural vs AI-generated examples

---

## Summary

VAAS provides a compact and expressive inference interface for:

* anomaly detection
* localisation
* interpretability
* feature-level analysis

All functionality is exposed through a single pipeline abstraction.
