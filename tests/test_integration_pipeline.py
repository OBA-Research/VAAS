import numpy as np
import pytest
from PIL import Image

from vaas.inference.pipeline import VAASPipeline


@pytest.mark.integration
def test_hf_model_loading_and_inference():
    pipe = VAASPipeline.from_pretrained(
        repo_id="OBA-Research/vaas",
        model_variant="v1-base-df2023",
        device="cpu",
    )

    img = Image.new("RGB", (224, 224), color="white")

    result = pipe(img)

    assert "S_H" in result
    assert isinstance(result["S_H"], float)
    assert isinstance(result["anomaly_map"], np.ndarray)
    assert result["anomaly_map"].shape == (224, 224)


@pytest.mark.integration
def test_visualization_pipeline(tmp_path):
    pipe = VAASPipeline.from_pretrained(
        repo_id="OBA-Research/vaas",
        model_variant="v1-base-df2023",
        device="cpu",
    )

    img = Image.new("RGB", (224, 224), color="white")
    output_path = tmp_path / "viz.png"

    pipe.visualize(
        image=img,
        save_path=str(output_path),
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
