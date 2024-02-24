import xarray as xr
from pathlib import Path
from tsdat import PipelineConfig, assert_close


def test_waves_pipeline():
    config_path = Path("pipelines/spotter/config/pipeline_flt.yaml")
    config = PipelineConfig.from_yaml(config_path)
    pipeline = config.instantiate_pipeline()

    test_file = "pipelines/spotter/test/data/input/0009_FLT.CSV"
    expected_file = "pipelines/spotter/test/data/expected/clallam.spotter-pos-400ms.a1.20210819.210649.nc"

    dataset = pipeline.run([test_file])
    expected: xr.Dataset = xr.open_dataset(expected_file)  # type: ignore
    assert_close(dataset, expected, check_attrs=False)


def test_gps_pipeline():
    config_path = Path("pipelines/spotter/config/pipeline_loc.yaml")
    config = PipelineConfig.from_yaml(config_path)
    pipeline = config.instantiate_pipeline()

    test_file = "pipelines/spotter/test/data/input/0009_LOC.CSV"
    expected_file = "pipelines/spotter/test/data/expected/clallam.spotter-gps-1min.a1.20210819.210649.nc"

    dataset = pipeline.run([test_file])
    expected: xr.Dataset = xr.open_dataset(expected_file)  # type: ignore
    assert_close(dataset, expected, check_attrs=False)
