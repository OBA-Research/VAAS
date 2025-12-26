import pytest

pytestmark = pytest.mark.smoke


def test_vaas_importable_without_torch():
    import vaas

    assert hasattr(vaas, "__version__")


def test_pipeline_class_importable():
    from vaas.inference.pipeline import VAASPipeline

    assert VAASPipeline is not None


def test_pipeline_fails_cleanly():
    from vaas.inference.pipeline import VAASPipeline

    with pytest.raises(SystemExit):
        VAASPipeline.from_pretrained("dummy-repo")


def test_public_api_surface():
    from vaas.inference.pipeline import VAASPipeline

    assert hasattr(VAASPipeline, "from_pretrained")
    assert hasattr(VAASPipeline, "from_checkpoint")
    assert callable(VAASPipeline)
