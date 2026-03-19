# VAP Waves V2 Transformation Pipeline

Data pipeline that reads in files that output by the `spotter_v2` ingest pipeline. It computes
wave parameters from the input data, copies over the metocean variables, and creates plots of
the final wave products

One will need to adjust attributes in the dataset.yaml file to save and plot the correct 
metadata. If there are multiple input files coming into this pipeline, you may need to adjust the 
"time_padding" parameter in retriever.yaml depending on difference in file timestamps.

## Prerequisites

* Ensure that your development environment has been set up according to
[the instructions](../../README.md#development-environment-setup).

## Running your pipeline

1. Navigate to the repository root from the terminal (i.e., 2 levels up from this file)
2. Run `runner.py` and specify the transformation pipeline that should run:

        ```shell
        python runner.py vap pipelines/vap_wave_v2/config/pipeline.yaml -b 20230324 -e 20230325
        ```


## Testing your pipeline
This template is set up with a pytest unit test to ensure your pipeline is working correctly.  It is intended that the
pytest unit tests will be run automatically before pipeline deployment to prevent against breaking code changes.  To
run your tests locally, run these commands from your anaconda environment shell:

```bash
cd $REPOSITORY_ROOT
pytest
```
