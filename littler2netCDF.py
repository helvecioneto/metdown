import argparse
import glob
import pathlib
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import Point
from metpy.interpolate import interpolate_to_grid
import cartopy.crs as ccrs
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings("ignore")

def remap_to_base(ds, base_ds):
    return ds.interp_like(base_ds)

def fullgrid(matrix):
    x, y = np.meshgrid(np.arange(matrix.shape[1]), np.arange(matrix.shape[0]))
    points = np.array((x[~np.isnan(matrix)], y[~np.isnan(matrix)])).T
    values = matrix[~np.isnan(matrix)]
    filled_matrix = griddata(points, values, (x, y), method='cubic')
    mask = np.isnan(filled_matrix)
    filled_matrix[mask] = griddata(points, values, (x, y), method='nearest')[mask]
    return filled_matrix

def df2xarray(df, var, ovar, full_grid=True, g_filter=True, g_sigma=20, 
            nodata=99999.00000, res = 10000, radius = 100000,
            interp_type='cressman', minimum_neighbors=1,
            gamma=0.25, kappa_star=5.052,
            boundary_coords=None):
    print('Processing variable:', var)
    # select the columns of interest
    geo_data = df[['Date','Longitude','Latitude',var]]
    geo_data = geo_data[geo_data[var] != nodata]
    # convert to geoDataFrame
    geo_data = gpd.GeoDataFrame(geo_data, crs="EPSG:4326",
                                geometry=gpd.points_from_xy(geo_data.Longitude, geo_data.Latitude))
    # Apply Pseudo UTM projection
    geo_data = geo_data.to_crs("EPSG:3857")
    # Group the data by date where groups have more than 10 points
    geo_group = geo_data.groupby(['Date'])
    datasets = []
    for date, group in geo_group:
        if len(group) < 10: # skip if less than 10 points
            continue       
        coords = np.array([(point.x, point.y) for point in group.geometry])
        x, y = coords[:,0], coords[:,1]
        data = group[var].values
        gx, gy, img = interpolate_to_grid(x, y, data, interp_type=interp_type, 
                                        minimum_neighbors=minimum_neighbors,
                                        gamma=gamma, kappa_star=kappa_star, 
                                        search_radius=radius, hres=res, 
                                        boundary_coords=boundary_coords)
        if full_grid:
            img = fullgrid(img)
        if g_filter:
            img = gaussian_filter(img, sigma=g_sigma)
        min_x, max_x, min_y, max_y = gx.min(), gx.max(), gy.min(), gy.max()
        points = [Point(min_x, min_y), Point(max_x, max_y)]
        geo_pts = gpd.GeoSeries(points, crs="EPSG:3857")
        geo_pts = geo_pts.to_crs(ccrs.PlateCarree().proj4_init)
        # Mount lats and lons arrays
        lats = np.linspace(geo_pts.geometry.y.min(), geo_pts.geometry.y.max(), img.shape[0])
        lons = np.linspace(geo_pts.geometry.x.min(), geo_pts.geometry.x.max(), img.shape[1])
        # Create the xarray dataset
        time_ = pd.to_datetime(date, format='%Y%m%d%H%M%S')[0]
        print('- Processing Time:', time_)
        ds = xr.Dataset({ovar: (['latitude', 'longitude'], img)},
                        coords={'latitude': lats, 'longitude': lons, 'time': time_})
        ds.longitude.attrs['units'] = 'degrees_east'
        ds.latitude.attrs['units'] = 'degrees_north'
        # Add Method used in variable
        ds[ovar].attrs['method'] = interp_type
        ds[ovar].attrs['radius'] = str(radius / 1000) + ' km'
        ds[ovar].attrs['resolution'] = str(res / 1000) + ' km'
        ds[ovar].attrs['minimum_neighbors'] = minimum_neighbors
        ds[ovar].attrs['gamma'] = gamma
        ds[ovar].attrs['kappa_star'] = kappa_star
        ds[ovar].attrs['description'] = 'Interpolated data from surface observations using MetPy'
        ds.attrs['title'] = 'Little R Data interpolated'
        ds.attrs['author'] = 'Helvecio Neto (2024) - helecioblneto@gmail.com/github.com/helecioneto'  
        datasets.append(ds)
    if len(datasets) > 0:
        base_ds = datasets[0]
        remapped_datasets = [remap_to_base(ds, base_ds) for ds in datasets]
        dataset = xr.concat(remapped_datasets, dim='time')
        return dataset
    elif len(datasets) == 1:
        return datasets[0]
    else:
        return None
    
if __name__ == '__main__':
    epilog = f"Exemple:\n\tpython littler2netCDF.py -i './output'" \
                " -o './output_nc' -var 'Temperature (K)' -ovar 'T2'"
    parser = argparse.ArgumentParser(description='Convert Little R Data to NetCDF',
                                    formatter_class=argparse.RawTextHelpFormatter,
                                    epilog=epilog)
    parser.add_argument('-i', '--input', dest='input', type=str,
                        help='Input Directory', required=True)
    parser.add_argument('-o', '--output', dest='output', type=str,
                        help='Output Directory', default='./output_nc')
    parser.add_argument('-var', '--variable', dest='variable', type=str,
                        help='Variable to convert', default='Temperature')
    parser.add_argument('-ovar', '--outputvar', dest='ovar', type=str,
                        help='Output variable name', default=None)
    parser.add_argument('-nodata', '--nodata', dest='nodata', type=float,
                        help='No data value', default=99999.00000)
    parser.add_argument('-res', '--resolution', dest='res', type=int,
                        help='Resolution in meters', default=10000)
    parser.add_argument('-radius', '--radius', dest='radius', type=int,
                        help='Search radius in meters', default=100000)
    parser.add_argument('-interp', '--interp', dest='interp', type=str,
                        help='Interpolation method', default='cressman')
    parser.add_argument('-min_neigh', '--min_neigh', dest='min_neigh', type=int,
                        help='Minimum neighbors', default=1)
    parser.add_argument('-gamma', '--gamma', dest='gamma', type=float,
                        help='Gamma value', default=0.25)
    parser.add_argument('-kappa', '--kappa', dest='kappa', type=float,
                        help='Kappa star value', default=5.052)
    # add parameter full_grid
    parser.add_argument('-fg', '--full_grid', dest='full_grid', type=bool,
                        help='Full grid interpolation', default=True)
    parser.add_argument('-gf', '--gaussian_filter', dest='gaussian_filter', type=bool,
                        help='Apply gaussian filter', default=True)
    parser.add_argument('-gs', '--gaussian_sigma', dest='gaussian_sigma', type=int,
                        help='Gaussian sigma value', default=20)
    args = parser.parse_args()
    # Read all files in the input directory
    files = glob.glob(args.input + '/*.csv')
    # Create the output directory if not exists
    pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)
    # Check if oar is none get name of ar
    if args.ovar is None:
        args.ovar = args.variable
    for file in files:
        print('Processing file:', file)
        df = pd.read_csv(file)
        # Transform column Data to datetime
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d%H%M%S', errors='coerce')
        time_file = file.split('/')[-1].split('.')[0].split(':')[-1]
        timefile = pd.to_datetime(time_file, format='%Y%m%d%H', errors='coerce')
        df = df.loc[df['Date'] == timefile]
        file_name = file.split('/')[-1].replace('.csv', '.nc')
        ####### Apply other filters here ########
        # df = df.loc[df[filter_var]]
        dataset = df2xarray(df, args.variable, args.ovar, 
                            args.full_grid, args.gaussian_filter, 
                            args.gaussian_sigma,
                            args.nodata, args.res, args.radius,
                            args.interp, args.min_neigh, args.gamma, args.kappa)
        if dataset is not None:
            output_file = args.output + '/' + file_name
            # Save as netCDF format classic
            dataset.to_netcdf(output_file)
            print('File saved:', output_file)
        else:
            print('No data to save')
    print('Process finished')
    