# VAAS: Vision-Attention Anomaly Scoring

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18064355.svg)](https://doi.org/10.5281/zenodo.18064355)
[![CI](https://github.com/OBA-Research/VAAS/actions/workflows/test.yaml/badge.svg)](https://github.com/OBA-Research/VAAS/actions/workflows/test.yaml)
[![PyPI](https://img.shields.io/pypi/v/vaas.svg)](https://pypi.org/project/vaas/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/OBA-Research/vaas-v1-df2023)
[![Python](https://img.shields.io/pypi/pyversions/vaas.svg)](https://pypi.org/project/vaas/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)



VAAS (Vision-Attention Anomaly Scoring) is a dual-module vision framework for image anomaly detection and localisation.  
It combines global attention-based reasoning with patch-level self-consistency analysis to produce interpretable anomaly scores and dense anomaly maps.

This repository provides the **inference-ready implementation** of VAAS for research engineers and practitioners.

---

## Read Paper
- [Arxiv version](https://arxiv.org/abs/2512.15512)
- [Conference version](https://arxiv.org/abs/2512.15512)

---

## Architecture

![VAAS Methodology](docs/architecture_diagrams/methodology.png)

VAAS integrates two complementary components:

- **Fx — Global Attention Module**  
  A Vision Transformer capturing semantic/global irregularities from attention patterns.

- **Px — Patch Consistency Module**  
  A SegFormer-based model capturing local inconsistencies across image patches.

These combine to produce:

- `S_F` — global attention fidelity  
- `S_P` — patch-level plausibility  
- `S_H` — hybrid anomaly score (final)

`S_H` is continuous and reflects anomaly **intensity**, not just presence.

---

## Installation

```
pip install vaas
```
OR

```
uv pip install vaas
```

**Important:** VAAS requires **PyTorch + torchvision**.  
Install the correct PyTorch build for your system (CPU / CUDA / ROCm):

https://pytorch.org/get-started/locally/

---

## Usage


### 1. Basic inference on local and online images

```python
from vaas.inference.pipeline import VAASPipeline
from PIL import Image
import requests
from io import BytesIO

pipeline = VAASPipeline.from_pretrained(
    "OBA-Research/vaas-v1-df2023",
    device="cpu",
    alpha=0.5
)

# # Option A: Using a local image
# image = Image.open("example.jpg").convert("RGB")
# result = pipeline(image)

# Option B: Using an online image
url = "https://raw.githubusercontent.com/OBA-Research/VAAS/main/examples/images/COCO_DF_C110B00000_00539519.jpg"
image = Image.open(BytesIO(requests.get(url).content)).convert("RGB")
result = pipeline(image)

print(result)
anomaly_map = result["anomaly_map"]

```

#### Output Format

```python
{
  "S_F": float,
  "S_P": float,
  "S_H": float,
  "anomaly_map": numpy.ndarray  # shape (224, 224)
}
```

### 2. Inference with visual explanation

VAAS can also generate a qualitative visualization combining:

* Patch-level anomaly heatmaps (Px)
* Global attention maps (Fx)
* Final hybrid anomaly score (S_H)

```python

pipeline.visualize(
    image=image,
    save_path="vaas_visualization.png",
    mode="all",        # options: "all", "px", "binary", "fx"
    threshold=0.5,
)
```

This will save a figure containing:

* Original image
* Patch-level anomaly overlays
* Global attention overlays
* A gauge-style visualization of the hybrid anomaly score

For examples:

![Inference with visual example](docs/visualizations/COCO_DF_I000B00000_00966250_vaas.png)
![Inference with visual example](docs/visualizations/COCO_DF_S000B00000_00120651_vaas.png)
![Inference with visual example](docs/visualizations/COCO_DF_C110B00000_00539519_vaas.png)

---

## Model Variants (Planned & Released)

| Version | Training Data | Description | Hugging Face Model |
|--------|----------------|-------------|--------------------|
| v1     | DF2023 (10%)   | Initial public inference release | [vaas-v1-df2023](https://huggingface.co/OBA-Research/vaas-v1-df2023) |
| v2     | DF2023 (≈50%)  | Planned scale-up experiment | TBD |
| v3     | DF2023 (100%)  | Full-dataset training (planned) | TBD |
| v4     | DF2023 + CASIA2.0 | Cross-dataset study (planned) | TBD |
| v5     | Other datasets | Exploratory generalisation study | TBD |


These planned variants aim to study the effect of training scale, dataset diversity, and cross-dataset benchmarking on generalisation and score calibration.

## Roadmap: Inference API 

- Batch inference and folder-level CLI  
- Richer visualisation modes  
- More efficient backbones  
- Expose rich image embeddings
- Cross-dataset inferencing 
- Model compression  
- Extended anomaly-map visualisation  
- ONNX / TorchScript export  
- Use cases with Streamlit / Gradio

---

---

## Example Notebooks

A set of runnable example notebooks covering:

- Quick-start inference with VAAS  
- Visualising anomaly maps  
- Batch / folder inference  
- How S_F, S_P, and S_H are combined  
- Device selection (CPU, CUDA, MPS)

If you would like to contribute a notebook, see  **[CONTRIBUTING.md](CONTRIBUTING.md)** for guidelines.

---

## Contributing

We welcome contributions that improve the usability, robustness, and extensibility of VAAS.

Please see the full guidelines in **[CONTRIBUTING.md](CONTRIBUTING.md)**.

---

## Citation

If you use VAAS in your research, please cite both the software and the associated paper as appropriate.

```python
@software{vaas,
  title        = {VAAS: Vision-Attention Anomaly Scoring},
  author       = {Bamigbade, Opeyemi and Scanlon, Mark and Sheppard, John},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18064355},
  url          = {https://doi.org/10.5281/zenodo.18064355}
}

```

```python
@article{bamigbade2025vaas,
  title={VAAS: Vision-Attention Anomaly Scoring for Image Manipulation Detection in Digital Forensics},
  author={Bamigbade, Opeyemi and Scanlon, Mark and Sheppard, John},
  journal={arXiv preprint arXiv:2512.15512},
  year={2025}
}
```

---

## License

MIT License.

---

## Maintainers

**OBA-Research**  
https://github.com/OBA-Research  
https://huggingface.co/OBA-Research
