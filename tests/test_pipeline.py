import pytest

pytest.importorskip("torch")
pytestmark = pytest.mark.integration


def test_vaas_pipeline_smoke():
    import numpy as np
    from PIL import Image

    from vaas.inference.pipeline import VAASPipeline

    repo_id = "OBA-Research/vaas-v1-df2023"

    pipeline = VAASPipeline.from_pretrained(
        repo_id,
        device="cpu",
        alpha=0.5,
    )

    img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        mode="RGB",
    )

    result = pipeline(img)

    assert isinstance(result, dict)
    assert {"S_F", "S_P", "S_H", "anomaly_map"} <= result.keys()

    assert isinstance(result["S_F"], float)
    assert isinstance(result["S_P"], float)
    assert isinstance(result["S_H"], float)

    anomaly_map = result["anomaly_map"]
    assert isinstance(anomaly_map, np.ndarray)
    assert anomaly_map.ndim == 2
