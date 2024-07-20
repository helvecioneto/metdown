import sys
import requests
import pathlib
import argparse
import regex as re
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from io import BytesIO
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


########### HEADER ################
header_cols = ['Latitude', 'Longitude', 'ID', 'Name', 'Platform (FM‑Code)', 
                'Source', 'Elevation', 'Valid fields', 'Num. errors', 'Num. warnings','Sequence number','Num. duplicates',
                'Is sounding?','Is bogus?','Discard?',
                'Unix time','Julian day','Date',
                'SLP','SLP-QC',
                'Ref Pressure','Ref Pressure-QC',
                'Precip','Precip-QC',
                'Daily Max T','Daily Max T-QC',
                'Daily Min T','Daily Min T-QC',
                'Night Min T','Night Min T-QC',
                '3hr Pres Change','3hr Pres Change-QC',
                '24hr Pres Change','24hr Pres Change-QC',
                'Cloud cover','Cloud cover-QC',
                'Ceiling','Ceiling-QC',
                'Precipitable water','Precipitable water-QC']
header_sizes = [20,20,40,40,40,40,20,10,10,10,10,10,10,10,10,10,10,20,13,7,13,7,13,7,13,7,13,7,13,7,13,7,13,7,13,7,13,7,13,7]
########### DATA ################
data_cols = ['Pressure (Pa)','Pressure (Pa)-QC',
            'Height (m)','Height (m)-QC',
            'Temperature (K)','Temperature (K)-QC',
            'Dew point (K)','Dew point (K)-QC',
            'Wind speed (m/s)','Wind speed (m/s)-QC',
            'Wind direction (deg)','Wind direction (deg)-QC',
            'Wind U (m/s)','Wind U (m/s)-QC',
            'Wind V (m/s)','Wind V (m/s)-QC',
            'Relative humidity (%)','Relative humidity (%)-QC',
            'Thickness (m)','Thickness (m)-QC']
data_sizes = [13,7,13,7,13,7,13,7,13,7,13,7,13,7,13,7,13,7,13,7]

def download_file(url):
    response = requests.head(url)
    if response.status_code == 200:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024)
        file_in_memory = BytesIO()
        for chunk in response.iter_content(chunk_size=1024):
            file_in_memory.write(chunk)
            progress_bar.update(len(chunk))
        progress_bar.close()
        file_in_memory.seek(0)
        return file_in_memory
    else:
        print(f"File Not Exist: {url}")
    return None

def procRow(row, sizes):
    begin = 0
    proc_row = []
    for i in range(len(sizes)):
        value = row[begin:begin + sizes[i]].strip()
        proc_row.append(value)
        begin += sizes[i]
    return proc_row


if __name__ == '__main__':
    current_date  = pd.Timestamp.now()
    # Current date possible is only 00:00, 06:00, 12:00, 18:00
    current_date = current_date.replace(hour=(current_date.hour//6)*6, minute=0, second=0, microsecond=0)    
    current_date = current_date - pd.Timedelta(hours=24)
    # current date to epilog
    epilog = f"Exemple:\n\tpython download_litte_r.py -t '{current_date.strftime('%Y-%m-%d %H:%M:%S')}'" \
            " -o ./output -lat_min -10 -lat_max 10 -lon_min -70 -lon_max -45\n\n" \
            "Note: The allowed timestamp is only 00:00, 06:00, 12:00, 18:00\n" \
            "Developed by: Helecio-Neto (2024) - helecioblneto@gmail.com"

    parser = argparse.ArgumentParser(description='Download Little R Data from RDA UCAR',
                                    formatter_class=argparse.RawTextHelpFormatter,
                                    epilog=epilog)
    parser.add_argument('-t', '--timestamp', dest='timestamp', type=str,
                        help='Timestamp to download data', required=True)
    parser.add_argument('-o', '--output', dest='output', type=str,
                        help='Output Directory', default='./output')
    parser.add_argument('-lat_min', dest='lat_min', type=float,
                        help='Minimum Latitude', default=-90)
    parser.add_argument('-lat_max', dest='lat_max', type=float,
                        help='Maximum Latitude', default=90)
    parser.add_argument('-lon_min', dest='lon_min', type=float,
                        help='Minimum Longitude', default=-180)
    parser.add_argument('-lon_max', dest='lon_max', type=float,
                        help='Maximum Longitude ', default=180)
    args = parser.parse_args()
    
    ### CONFIGURATION #################
    timestamp = pd.to_datetime(args.timestamp)
    lat_min = args.lat_min
    lat_max = args.lat_max
    lon_min = args.lon_min
    lon_max = args.lon_max
    output = pathlib.Path(args.output)
    pathlib.Path(output).mkdir(parents=True, exist_ok=True)
    
    ##################################
    url = 'https://data.rda.ucar.edu/ds461.0/little_r/{year}/SURFACE_OBS:{year}{month}{day}{hour}'
    url = url.format(year=timestamp.strftime('%Y'),month=timestamp.strftime('%m'),day=timestamp.strftime('%d'),hour=timestamp.strftime('%H'))
    print(f"- Downloading From: {url}")
    file = download_file(url)
    if file is None:
        print("Error Downloading File" + url)
        exit(1)
    data = file.read()
    print("- Processing Data")
    split_data = re.split(r"\n", data.decode('utf-8', errors='ignore'))
    split_array = np.array(split_data)
    # Split Header and Data
    header = split_array[::4][:-1]
    data = split_array[1::4]
    # Add to data Frame
    header_df = pd.DataFrame([procRow(row, header_sizes) for row in header], columns=header_cols)
    data_df = pd.DataFrame([procRow(row, data_sizes) for row in data], columns=data_cols)
    df = pd.concat([header_df, data_df], axis=1)
    # Cast Latitude And Longitude To numeric
    print("- Applying Geopoints and Cropping Region [%.2f,%.2f,%.2f,%.2f]" % (lat_min, lat_max, lon_min, lon_max))
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    # Apply geopoints and crop the region
    geopoints = df[['Longitude','Latitude']].astype(float).apply(Point, axis=1)
    geopoints = gpd.GeoSeries(geopoints).cx[lon_min:lon_max, lat_min:lat_max]
    # Fit to points
    df = df.loc[geopoints.index]
    # Transform all data possible to numeric using pd.to_numeric
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass
    filename = pathlib.Path(url).name + '.csv'
    df.to_csv(output / filename, index=False)
    print(f"- Saving to {output / filename}")