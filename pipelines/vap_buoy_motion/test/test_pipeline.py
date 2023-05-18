import xarray as xr
from pathlib import Path
from tsdat import PipelineConfig, assert_close


# def test_vap_waves_pipeline():
#     config_path = Path("pipelines/vap_waves/config/pipeline.yaml")
#     config = PipelineConfig.from_yaml(config_path)
#     pipeline = config.instantiate_pipeline()

#     test_file = [
#         # "pipelines/vap_waves/test/data/input/clallam.spotter-wave-400ms.a1.20210824.180425.displacement.png",
#         "20210824.200000",
#         "20210825.200000",
#     ]
#     expected_file = (
#         "pipelines/vap_waves/test/data/expected/abc.example.a1.20220424.000000.nc"
#     )

#     dataset = pipeline.run(test_file)
#     expected: xr.Dataset = xr.open_dataset(expected_file)  # type: ignore
#     assert_close(dataset, expected, check_attrs=False)
