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


def remap_to_base(ds, base_ds):
    return ds.interp_like(base_ds)

def df2xarray(df, var, nodata=99999.00000, res = 9000, radius = 50000,
            interp_type='cressman', minimum_neighbors=1, gamma=0.25, kappa_star=5.052,
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
        ds = xr.Dataset({var: (['latitude', 'longitude'], img)},
                        coords={'latitude': lats, 'longitude': lons, 'time': time_})
        # Add Method used in variable
        ds[var].attrs['method'] = interp_type
        ds[var].attrs['radius'] = str(radius / 1000) + ' km'
        ds[var].attrs['resolution'] = str(res / 1000) + ' km'
        ds[var].attrs['minimum_neighbors'] = minimum_neighbors
        ds[var].attrs['gamma'] = gamma
        ds[var].attrs['kappa_star'] = kappa_star
        ds[var].attrs['description'] = 'Interpolated data from surface observations using MetPy'
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
                " -o './output_nc' -var 'Temperature (K)'"
    parser = argparse.ArgumentParser(description='Convert Little R Data to NetCDF',
                                    formatter_class=argparse.RawTextHelpFormatter,
                                    epilog=epilog)
    parser.add_argument('-i', '--input', dest='input', type=str,
                        help='Input Directory', required=True)
    parser.add_argument('-o', '--output', dest='output', type=str,
                        help='Output Directory', default='./output_nc')
    parser.add_argument('-var', '--variable', dest='variable', type=str,
                        help='Variable to convert', default='Temperature')
    parser.add_argument('-nodata', '--nodata', dest='nodata', type=float,
                        help='No data value', default=99999.00000)
    parser.add_argument('-res', '--resolution', dest='res', type=int,
                        help='Resolution in meters', default=10000)
    parser.add_argument('-radius', '--radius', dest='radius', type=int,
                        help='Search radius in meters', default=300000)
    parser.add_argument('-interp', '--interp', dest='interp', type=str,
                        help='Interpolation method', default='cressman')
    parser.add_argument('-min_neigh', '--min_neigh', dest='min_neigh', type=int,
                        help='Minimum neighbors', default=1)
    parser.add_argument('-gamma', '--gamma', dest='gamma', type=float,
                        help='Gamma value', default=0.25)
    parser.add_argument('-kappa', '--kappa', dest='kappa', type=float,
                        help='Kappa star value', default=5.052)    
    args = parser.parse_args()
    # Read all files in the input directory
    files = glob.glob(args.input + '/*.csv')
    # Create the output directory if not exists
    pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)
    for file in files:
        print('Processing file:', file)
        df = pd.read_csv(file)
        dataset = df2xarray(df, args.variable, args.nodata, args.res, args.radius,
                            args.interp, args.min_neigh, args.gamma, args.kappa)
        if dataset is not None:
            output_file = args.output + '/' + args.variable + '.nc'
            dataset.to_netcdf(output_file)
            print('File saved:', output_file)
        else:
            print('No data to save')
    print('Process finished')
    