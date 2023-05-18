import xarray as xr
from pathlib import Path
from tsdat import PipelineConfig, assert_close


def test_wave_stats_pipeline():
    config_path = Path("pipelines/wave_stats/config/pipeline.yaml")
    config = PipelineConfig.from_yaml(config_path)
    pipeline = config.instantiate_pipeline()

    test_file = (
        "pipelines/wave_stats/test/data/input/clallam.vap_pos.c0.20210824.180425.nc"
    )
    expected_file = "pipelines/wave_stats/test/data/expected/clallam.wave_stats.c1.20210824.180925.nc"

    dataset = pipeline.run([test_file])
    expected: xr.Dataset = xr.open_dataset(expected_file)  # type: ignore
    assert_close(dataset, expected, check_attrs=False)
