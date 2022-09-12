import xarray as xr
from pathlib import Path
from tsdat import PipelineConfig, assert_close


def test_current_pipeline():
    config_path = Path("pipelines/current/config/pipeline.yaml")
    config = PipelineConfig.from_yaml(config_path)
    pipeline = config.instantiate_pipeline()

    test_file = "pipelines/current/test/data/input/Sig1000_tidal.ad2cp"
    expected_file = "pipelines/current/test/data/expected/adcp.current-velocity-1s.a1.20200815.002000.nc"

    dataset = pipeline.run([test_file])
    expected: xr.Dataset = xr.open_dataset(expected_file)  # type: ignore
    assert_close(dataset, expected, check_attrs=False)
