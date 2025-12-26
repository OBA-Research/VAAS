import os

import pytest

pytest.importorskip("torch")
pytestmark = pytest.mark.integration

if not os.path.exists("checkpoints"):
    pytest.skip("Local checkpoints not available", allow_module_level=True)


def test_local_pipeline():
    import numpy as np
    from PIL import Image

    from vaas.inference.pipeline import VAASPipeline

    checkpoint_dir = "checkpoints/DF2023_VAAS_DF2023_20251217_163102"

    pipeline = VAASPipeline.from_checkpoint(
        checkpoint_dir=checkpoint_dir,
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
