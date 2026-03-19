# VAP_Wave_Raw Transformation Pipeline

This VAP pipeline works in conjuction with the `spotter_v1` ingest pipeline to combine and plot
the buoy's wave data. The output of this pipeline feeds into the `vap_wave_stats_v1`
pipeline.

## Prerequisites

* Ensure that your development environment has been set up according to
[the instructions](../../README.md#development-environment-setup).

## Running your pipeline

1. Navigate to the repository root from the terminal (i.e., 2 levels up from this file)
2. Run `runner.py` and specify the transformation pipeline that should run:

        ```shell
        python runner.py vap pipelines/vap_wave_raw_v1/config/pipeline.yaml -b 20230324 -e 20230325
        ```


## Testing your pipeline
This template is set up with a pytest unit test to ensure your pipeline is working correctly.  It is intended that the
pytest unit tests will be run automatically before pipeline deployment to prevent against breaking code changes.  To
run your tests locally, run these commands from your anaconda environment shell:

```bash
cd $REPOSITORY_ROOT
pytest
```
