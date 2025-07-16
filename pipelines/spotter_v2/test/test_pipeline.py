import xarray as xr
from pathlib import Path
from tsdat import PipelineConfig, assert_close


def test_spotter_v2_pipeline_v2():
    config_path = Path("pipelines/spotter_v2/config/pipeline.yaml")
    config = PipelineConfig.from_yaml(config_path)
    pipeline = config.instantiate_pipeline()

    test_file = "pipelines/spotter_v2/test/data/input/spot1945.zip"
    expected_file = "pipelines/spotter_v2/test/data/expected/pnnl.spotter-1945.a1.20210903.160801.nc"

    dataset = pipeline.run([test_file])
    expected: xr.Dataset = xr.open_dataset(expected_file)  # type: ignore
    assert_close(dataset, expected, check_attrs=False)


def test_spotter_v2_pipeline_v3():
    config_path = Path("pipelines/spotter_v2/config/pipeline.yaml")
    config = PipelineConfig.from_yaml(config_path)
    pipeline = config.instantiate_pipeline()

    test_file = "pipelines/spotter_v2/test/data/input/spot30857c.zip"
    expected_file = "pipelines/spotter_v2/test/data/expected/pnnl.spotter-30857C.a1.20250110.194406.nc"

    dataset = pipeline.run([test_file])
    expected: xr.Dataset = xr.open_dataset(expected_file)  # type: ignore
    assert_close(dataset, expected, check_attrs=False)


def test_spotter_v2_pipeline_extra_files():
    config_path = Path("pipelines/spotter_v2/config/pipeline.yaml")
    config = PipelineConfig.from_yaml(config_path)
    pipeline = config.instantiate_pipeline()

    test_file = "pipelines/spotter_v2/test/data/input/spot32632C.zip"
    expected_file = "pipelines/spotter_v2/test/data/expected/pnnl.spotter-32632C.a1.20250708.222403.nc"

    dataset = pipeline.run([test_file])
    expected: xr.Dataset = xr.open_dataset(expected_file)  # type: ignore
    assert_close(dataset, expected, check_attrs=False)
