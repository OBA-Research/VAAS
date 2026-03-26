import pytest

pytest.importorskip("torch")
pytestmark = pytest.mark.integration


def _get_pipeline():
    from vaas.inference.pipeline import VAASPipeline

    repo_id = "OBA-Research/vaas"
    model_variant = "v1-base-df2023"

    return VAASPipeline.from_pretrained(
        repo_id,
        device="cpu",
        alpha=0.5,
        model_variant=model_variant,
    )


def _get_image():
    import numpy as np
    from PIL import Image

    return Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        mode="RGB",
    )


def test_vaas_pipeline_smoke():
    import numpy as np

    pipeline = _get_pipeline()
    img = _get_image()

    result = pipeline(img)

    assert isinstance(result, dict)
    assert {"S_F", "S_P", "S_H", "anomaly_map"} <= result.keys()

    assert isinstance(result["S_F"], float)
    assert isinstance(result["S_P"], float)
    assert isinstance(result["S_H"], float)

    anomaly_map = result["anomaly_map"]
    assert isinstance(anomaly_map, np.ndarray)
    assert anomaly_map.ndim == 2


def test_forward_detailed():
    pipeline = _get_pipeline()
    img = _get_image()

    out = pipeline.forward_detailed(img)

    assert isinstance(out, dict)
    assert {"S_F", "S_P", "S_H", "anomaly_map"} <= out.keys()
    assert "metadata" in out
    assert "variant" in out


def test_extract_patch_scores():
    import numpy as np

    pipeline = _get_pipeline()
    img = _get_image()

    scores = pipeline.extract_patch_scores(img)

    assert isinstance(scores, np.ndarray)
    assert scores.ndim == 2


def test_extract_features():
    import numpy as np

    pipeline = _get_pipeline()
    img = _get_image()

    feats = pipeline.extract_features(img)

    assert isinstance(feats, dict)

    assert {
        "fx_cls_embedding",
        "fx_patch_tokens",
        "px_cross_attention_tokens",
    } <= feats.keys()

    assert isinstance(feats["fx_cls_embedding"], np.ndarray)
    assert isinstance(feats["fx_patch_tokens"], np.ndarray)
    assert isinstance(feats["px_cross_attention_tokens"], np.ndarray)


def test_batch():
    pipeline = _get_pipeline()
    imgs = [_get_image() for _ in range(3)]

    results = pipeline.batch(imgs)

    assert isinstance(results, list)
    assert len(results) == 3
    assert all(isinstance(r, dict) for r in results)
