import os
import shutil
import math
import zipfile
import csv
from itertools import groupby
from pathlib import Path
import datetime
from time import sleep


def group_by_range(data, range_size):
    # Sort the data first, as groupby only groups consecutive identical keys
    data.sort()

    # Define a key function to determine the bin for each number
    # Using floor division (//) creates the groups (e.g., numbers 0-9 go into bin 0, 10-19 into bin 1, etc.)
    key_func = lambda x: math.floor(x / range_size) * range_size

    grouped_data = {}
    for key, group in groupby(data, key=key_func):
        # Convert the group iterator to a list and store with the range key
        grouped_data[f"{key}-{key + range_size - 1}"] = list(group)

    return grouped_data


def extract_timestamp_from_filename(filename):
    # Open the CSV file and read the first two rows to get the timestamp
    with open(f"{filename}", "r") as csvfile:
        reader = csv.reader(csvfile)
        # Read the first two rows
        header = next(reader)
        try:
            first_line = next(reader)
        except StopIteration:
            return None
        # Get timestamp
        timestamp = datetime.datetime.fromtimestamp(float(first_line[1]), datetime.UTC)
        return timestamp.strftime("%Y%m%d.%H%M%S")


def split_zip_folders(zip_file_path):
    """
    This function takes a zip file path as input, extracts the files, and groups them into new zip
    files based on their ID numbers and associated timestamps.

    The ID numbers are extracted from the filenames, and files are grouped into ranges of 10 (e.g., 0-9, 10-19, etc.).
    Each new zip file is named based on the original zip file name and the timestamp extracted
    from the first non-empty FLT file in each group. The original zip file is not modified, and
    the new zip files are created in the same directory as the original zip file.
    """
    output_folder_base = Path(zip_file_path).stem
    all_folders = [output_folder_base]

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        flt_filelist = [f for f in zip_ref.namelist() if "_FLT" in f]
        # Grab file number ID
        ids = [int(Path(f).stem.split("_")[0]) for f in flt_filelist]
        # Sort IDs into groups of 10 sequentially
        sorted_ids = group_by_range(ids, 10)

        for id_list in sorted_ids:
            # First, get timestamp from the first populated file in each ID group
            # Some FLT files might be empty, which makes life harder
            timestamp_str = None
            i = 0
            while timestamp_str is None:
                # Get a FLT file
                timestamp_file = next(
                    (
                        f
                        for f in flt_filelist
                        if Path(f).stem.startswith(str(sorted_ids[id_list][0]).zfill(4))
                    ),
                    None,
                )
                # Temporarily extract the timestamp file to read the timestamp and name the output zip file
                zip_ref.extract(timestamp_file, ".")
                # Read the timestamp if it has data
                timestamp_str = extract_timestamp_from_filename(timestamp_file)
                # If there is no data, remove the ID from the group
                if timestamp_str is None:
                    sorted_ids[id_list].pop(0)
                # If there are no IDs left in the group, break out
                if not sorted_ids[id_list]:
                    break
                i += 1
                # Delete the temporary file
                os.remove(f"{timestamp_file}")

            # If there was no data in the group, skip to the next group
            if not sorted_ids[id_list]:
                continue

            # Create new output folder for each group of IDs
            output_folder = f"{output_folder_base}_{id_list}"
            all_folders.append(output_folder)
            # If a timestamp was found, manually chop off and replace old timestamp
            output_folder_zip = f"{output_folder_base[:-18]}.{timestamp_str}.zip"

            # Extract files corresponding to the current group of IDs
            for id in sorted_ids[id_list]:
                # Fetch files corresponding to IDs (loops one ID at a time)
                members = [
                    f
                    for f in zip_ref.namelist()
                    if Path(f).stem.startswith(str(id).zfill(4))
                ]

                # Save them to a temporary output folder
                for member in members:
                    zip_ref.extract(member, output_folder)

                # Compress the extracted files into a new zip file
                with zipfile.ZipFile(
                    output_folder_zip, "a", compression=zipfile.ZIP_DEFLATED
                ) as new_zip:
                    for member in members:
                        new_zip.write(f"{output_folder}/{member}", Path(member).name)

    # Delete the temporary output folders after creating the zip files
    # This was causing errors if I tried deleting them immediately after creating each zip file
    sleep(0.25)
    for folder in all_folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)


if __name__ == "__main__":
    zip_file_path = "PWS_SPOTTER_30903c_08022024_05152025.zip"
    split_zip_folders(zip_file_path)
