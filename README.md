## Installation

To install the necessary libraries, first ensure you have Python installed in your environment. Then, create a virtual environment (optional) and install the dependencies using the provided `requirements.txt`.

1. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```


2. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```


## Usage

#### download_litte_r.py

This script is used to download data from a remote data service based on specific date and location parameters.

Usage:

   ```bash
    python download_litte_r.py -t '2024-07-19 12:00:00' -o ./output -lat_min -10 -lat_max 10 -lon_min -70 -lon_max -45
   ```

* -t : Date and time for downloading data in the format 'YYYY-MM-DD HH:MM:SS'.
* -o : Directory where the downloaded files will be stored.
* -lat_min : Minimum latitude of the area of interest.
* -lat_max : Maximum latitude of the area of interest.
* -lon_min : Minimum longitude of the area of interest.
* -lon_max : Maximum longitude of the area of interest.


####  littler2netCDF.py
This script converts the downloaded data into NetCDF spatial data

Usage

   ```bash
   python littler2netCDF.py -i './output' -o './output_nc' -var 'Temperature (K)'
   ```

* -i : Input directory where the downloaded files are located.
* -o : Output directory where the converted NetCDF files will be stored.
* -var : Name of the variable to be converted to NetCDF format.


Contribution
If you would like to contribute to the project, feel free to open issues and pull requests.

Author: Helvecio Neto - 2024
