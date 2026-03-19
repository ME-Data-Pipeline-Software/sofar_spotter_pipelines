# Spotter V3 Ingestion Pipeline

Data pipeline for reading in zip folders from Sofar Spotter3 wave buoys. These buoys are the ones that
also measure air temperature, humidity, and pressure in addition to sea surface temperature, buoy charge/power
stats, and wave measurements. 
It is set up to ingest a zip folder containing all of the files directly from a
Spotter3's SD card. It reads the buoy motion, GPS position, metocean variables, and 
battery/charge parameters from the version 3 Spotter buoy. It is not backwards compatible
with the version 1 or 2 buoys. It is meant to be used in conjunction with the `vap_wave_v3` pipeline, which
calculates the wave statistics.

This pipeline functions better if one clears the SD card before starting a new 
deployment; otherwise delete the old files before running this pipeline.

```bash
cd $REPOSITORY_ROOT
conda activate spotter # <-- you only need to do this the first time you start a terminal shell
python runner.py ingest <path/to/your/zip/folder>.zip
```

## Prerequisites

* Ensure that your development environment has been set up according to
[the instructions](../../README.md#development-environment-setup).

> **Windows Users** - Make sure to run your `conda` commands from an Anaconda prompt OR from a WSL shell with miniconda
> installed. If using WSL, see [this tutorial on WSL](https://tsdat.readthedocs.io/en/latest/tutorials/wsl.html) for
> how to set up a WSL environment and attach VS Code to it.

* Make sure to activate the spotter anaconda environment before running any commands:  `conda activate spotter`

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
