# Spotter V1 Ingestion Pipeline

Data pipeline for ingesting raw data from the original Sofar spotter wave buoy. Datasets from new
buoys should utilize the v2 or v3 pipelines. 
For this pipeline, the user must first 
extract the zip folder and then run this pipeline on the extracted folder. It reads in buoy 
motion, position, and sea surface temperature from the respective csv files and saves
them individually. This pipeline is meant to be used in conjunction with the `vap_wave_raw_v1`
and `vap_wave_stats_v1` pipelines.

```bash
cd $REPOSITORY_ROOT
conda activate spotter # <-- you only need to do this the first time you start a terminal shell
python runner.py ingest <path/to/your/extracted/folder>/*.csv
```

## Prerequisites

* Ensure that your development environment has been set up according to
[the instructions](../../README.md#development-environment-setup).

> **Windows Users** - Make sure to run your `conda` commands from an Anaconda prompt OR from a WSL shell with miniconda
> installed. If using WSL, see [this tutorial on WSL](https://tsdat.readthedocs.io/en/latest/tutorials/wsl.html) for
> how to set up a WSL environment and attach VS Code to it.

* Make sure to activate the spotter anaconda environment before running any commands:  `conda activate tsdat-pipelines`

## Running your pipeline
This section shows you how to run the ingest pipeline created by the template.  Note that `{ingest-name}` refers
to the pipeline name you typed into the template prompt, and `{location}` refers to the location you typed into
the template prompt.

1. Make sure to be at your $REPOSITORY_ROOT. (i.e., where you cloned the pipeline-template repository)

2. Run the runner.py with your test data input file as shown below:

```bash
cd $REPOSITORY_ROOT
conda activate spotter
python runner.py ingest pipelines/{ingest-name}/test/data/input/{location}_data.csv
```

## Testing your pipeline
This template is set up with a pytest unit test to ensure your pipeline is working correctly.  It is intended that the
pytest unit tests will be run automatically before pipeline deployment to prevent against breaking code changes.  To
run your tests locally, run these commands from your anaconda environment shell:

```bash
cd $REPOSITORY_ROOT
pytest
```
