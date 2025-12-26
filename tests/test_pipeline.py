def test_vaas_pipeline_smoke():
    import numpy as np
    from PIL import Image

    from vaas.inference.pipeline import VAASPipeline

    repo_id = "OBA-Research/vaas-v1-df2023"

    pipeline = VAASPipeline.from_pretrained(repo_id, device="cpu", alpha=0.5)

    img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8), mode="RGB"
    )

    result = pipeline(img)

    assert isinstance(result, dict)

    assert "S_F" in result
    assert "S_P" in result
    assert "S_H" in result
    assert "anomaly_map" in result

    assert isinstance(result["S_F"], float)
    assert isinstance(result["S_P"], float)
    assert isinstance(result["S_H"], float)

    anomaly_map = result["anomaly_map"]
    assert isinstance(anomaly_map, np.ndarray)
    assert anomaly_map.ndim == 2
    assert anomaly_map.shape[0] > 0
    assert anomaly_map.shape[1] > 0

    print("VAASPipeline smoke test passed")
    print(
        f"S_F: {result['S_F']}",
        f"S_P: {result['S_P']}",
        f"S_H: {result['S_H']}",
        f"anomaly_map shape: {result['anomaly_map'].shape}",
    )


if __name__ == "__main__":
    test_vaas_pipeline_smoke()
