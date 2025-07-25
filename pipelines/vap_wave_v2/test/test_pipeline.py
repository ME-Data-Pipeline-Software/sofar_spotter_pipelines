from pathlib import Path
import pytest
import xarray as xr
from tsdat import assert_close, PipelineConfig, TransformationPipeline


@pytest.mark.dependency(
    depends=["../../spotter_v2/test/test_spotter_v2_pipeline_extra_files.py"]
)
def test_vap_wave_v2_pipeline():
    # The transformation pipeline will likely depend on the output of an ingestion
    # pipeline. To account for this we first run the ingest to generate input data for
    # the vap, and then run the vap test. Please update the line below to point to the
    #  correct folder / test name
    from pipelines.spotter_v2.test.test_pipeline import (
        test_spotter_v2_pipeline_extra_files,
    )

    test_spotter_v2_pipeline_extra_files()

    config_path = Path("pipelines/vap_wave_v2/config/pipeline.yaml")
    config = PipelineConfig.from_yaml(config_path)
    pipeline: TransformationPipeline = config.instantiate_pipeline()  # type: ignore

    # Transformation pipelines require an input of [date.time, date.time] formatted as
    # YYYYMMDD.hhmmss. The start date is inclusive, the end date is exclusive.
    run_dates = ["20250709.150000", "20250710.030000"]
    dataset = pipeline.run(run_dates)

    # You will need to create this file after running the data through the pipeline
    # OR: Delete this and perform sanity checks on the input data instead of comparing
    # with an expected output file
    expected_file = "pipelines/vap_wave_v2/test/data/expected/pnnl.spotter-32632C.c1.20250709.173821.nc"
    expected: xr.Dataset = xr.open_dataset(expected_file)  # type: ignore
    assert_close(dataset, expected, check_attrs=False, atol=1e-5)
