import os

import pytest

pytest.importorskip("torch")
pytestmark = pytest.mark.integration


def test_visualization():
    import numpy as np
    from PIL import Image

    from vaas.inference.pipeline import VAASPipeline

    checkpoint_dir = "checkpoints/DF2023_VAAS_DF2023_20251217_163102"
    save_path = "tests/_artifacts/vaas_vis_test.png"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    pipeline = VAASPipeline.from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        device="cpu",
        alpha=0.5,
    )

    img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        mode="RGB",
    )

    pipeline.visualize(
        image=img,
        save_path=save_path,
        mode="all",
    )

    assert os.path.exists(save_path)

    os.remove(save_path)
