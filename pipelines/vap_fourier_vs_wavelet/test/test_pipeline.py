from pathlib import Path
import xarray as xr
from tsdat import assert_close, PipelineConfig, TransformationPipeline


def test_vap_fourier_vs_wavelet():
    # The transformation pipeline will likely depend on the output of an ingestion
    # pipeline. To account for this we first run the ingest to generate input data for
    # the vap, and then run the vap test. Please update the line below to point to the
    #  correct folder / test name
    from pipelines.spotter_v1.test.test_pipeline import test_waves_pipeline

    test_waves_pipeline()

    config_path = Path("pipelines/vap_fourier_vs_wavelet/config/pipeline.yaml")
    config = PipelineConfig.from_yaml(config_path)
    pipeline: TransformationPipeline = config.instantiate_pipeline()  # type: ignore

    # Transformation pipelines require an input of [date.time, date.time] formatted as
    # YYYYMMDD.hhmmss. The start date is inclusive, the end date is exclusive.
    run_dates = ["20210903.170000", "20210903.200000"]
    dataset = pipeline.run(run_dates)

    # You will need to create this file after running the data through the pipeline
    # OR: Delete this and perform sanity checks on the input data instead of comparing
    # with an expected output file
    expected_file = "pipelines/vap_fourier_vs_wavelet/test/data/expected/clallam.wavelets.c0.20210903.171459.nc"
    expected: xr.Dataset = xr.open_dataset(expected_file)  # type: ignore
    assert_close(dataset, expected, check_attrs=False, atol=1e-5)
