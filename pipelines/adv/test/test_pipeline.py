import xarray as xr
from pathlib import Path
from tsdat import PipelineConfig, assert_close


def test_adv_pipeline():
    config_path = Path("pipelines/adv/config/pipeline.yaml")
    config = PipelineConfig.from_yaml(config_path)
    pipeline = config.instantiate_pipeline()

    test_file = "pipelines/adv/test/data/input/vector_example.VEC"
    expected_file = "pipelines/adv/test/data/expected/cook_inlet.adv-velocity-125ms.a1.20210701.134002.nc"

    dataset = pipeline.run([test_file])
    expected: xr.Dataset = xr.open_dataset(expected_file)  # type: ignore
    assert_close(dataset, expected, check_attrs=False)
