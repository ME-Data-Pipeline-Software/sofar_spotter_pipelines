from pathlib import Path
import pytest
import xarray as xr
from tsdat import assert_close, PipelineConfig, TransformationPipeline


@pytest.mark.dependency(depends=["../../example_pipeline/test/test_pipeline.py"])
def test_vap_gps_pipeline():
    # The transformation pipeline will likely depend on the output of an ingestion
    # pipeline. To account for this we first run the ingest to generate input data for
    # the vap, and then run the vap test. Please update the line below to point to the
    #  correct folder / test name
    from pipelines.spotter.test.test_pipeline import test_gps_pipeline

    test_gps_pipeline()

    config_path = Path("pipelines/vap_gps/config/pipeline.yaml")
    config = PipelineConfig.from_yaml(config_path)
    pipeline: TransformationPipeline = config.instantiate_pipeline()  # type: ignore

    # Transformation pipelines require an input of [date.time, date.time] formatted as
    # YYYYMMDD.hhmmss. The start date is inclusive, the end date is exclusive.
    run_dates = ["20210903.000000", "20210904.000000"]
    dataset = pipeline.run(run_dates)

    # You will need to create this file after running the data through the pipeline
    # OR: Delete this and perform sanity checks on the input data instead of comparing
    # with an expected output file
    expected_file = (
        "pipelines/vap_gps/test/data/expected/clallam.gps.c0.20210903.000500.nc"
    )
    expected: xr.Dataset = xr.open_dataset(expected_file)  # type: ignore
    assert_close(dataset, expected, check_attrs=False)
