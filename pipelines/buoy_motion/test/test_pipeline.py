import xarray as xr
from pathlib import Path
from tsdat import PipelineConfig, assert_close


def test_waves_pipeline():
    config_path = Path("pipelines/buoy_motion/config/pipeline_flt.yaml")
    config = PipelineConfig.from_yaml(config_path)
    pipeline = config.instantiate_pipeline()

    test_file = "pipelines/buoy_motion/test/data/input/0009_FLT.CSV"
    expected_file = "pipelines/buoy_motion/test/data/expected/clallam.spotter-pos-400ms.a1.20210819.210649.nc"

    dataset = pipeline.run([test_file])
    expected: xr.Dataset = xr.open_dataset(expected_file)  # type: ignore
    assert_close(dataset, expected, check_attrs=False)
