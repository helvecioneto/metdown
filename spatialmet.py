import os
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
import xarray as xr
from scipy.interpolate import griddata
from distutils.util import strtobool
import warnings
warnings.filterwarnings("ignore")

# Importar MetPy para interpolação meteorológica
try:
    from metpy.interpolate import interpolate_to_grid as metpy_interpolate
    METPY_AVAILABLE = True
except ImportError:
    METPY_AVAILABLE = False
    print("Aviso: MetPy não está instalado. Para usar interpolação meteorológica avançada, instale com: pip install metpy")

# Suprimir avisos e erros GDAL (controle de nível C)
os.environ['CPL_DEBUG'] = 'OFF'  # Desativar mensagens de debug
os.environ['GDAL_PAM_ENABLED'] = 'NO'  # Desativar PAM (Persistent Auxiliary Metadata)
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'  # Otimização de desempenho
os.environ['GDAL_ERROR_ON_LIBJPEG_WARNING'] = 'FALSE'  # Suprimir avisos libjpeg

# Configurar nível de erro do GDAL para "ERROR" (2) - ignora warnings
try:
    from osgeo import gdal
    gdal.UseExceptions()  # Fazer GDAL lançar exceções em vez de imprimir erros
    gdal.PushErrorHandler('CPLQuietErrorHandler')  # Manipulador silencioso de erros
except ImportError:
    pass  # Se GDAL não estiver disponível diretamente, continuamos


def fullgrid(grid):
    """Preenche uma grade com NaNs nas bordas"""
    # Cria uma máscara onde os valores são NaN
    mask = np.isnan(grid)
    
    # Se não houver NaNs, retorna a grade original
    if not np.any(mask):
        return grid
    
    # Obtém os índices dos valores não-NaN
    idx = np.where(~mask, np.arange(mask.shape[0])[:, None], 0)
    idy = np.where(~mask, np.arange(mask.shape[1]), 0)
    
    # Obtém os índices onde a grade tem valores NaN
    nan_idx = np.where(mask)
    
    # Interpola os valores NaN usando os valores não-NaN mais próximos
    grid[nan_idx] = grid[idx[nan_idx], idy[nan_idx]]
    
    return grid


def remap_to_base(ds, base_ds):
    """Remapeia um dataset para as mesmas coordenadas de um dataset base"""
    # Interpola para as coordenadas do dataset base
    return ds.interp(
        lat=base_ds.lat,
        lon=base_ds.lon,
        method='linear'
    )


class LittleRProcessor:
    """Class to download and process Little R data from UCAR"""
    
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

    ##### INTERESTING VARIABLES AND THRESHOLDS #####
    interesting_vars = {
        'Precip': (0, 1000),
        'Pressure (Pa)': (0, 100000),
        'Height (m)': (0, 5000),
        'Temperature (K)': (100, 400),
        'Dew point (K)': (0, 400),
        'Wind speed (m/s)': (0, 100),
        'Wind direction (deg)': (0, 360),
        'Wind U (m/s)': (-100, 100),
        'Wind V (m/s)': (-100, 100),
        'Relative humidity (%)': (0, 100),
        'Thickness (m)': (0, 10000)
    }
    
    def __init__(self, timestamp, output_dir="./output", 
                 lat_min=-90, lat_max=90, lon_min=-180, lon_max=180,
                 grid_resolution=None, interpolation_method='linear',
                 output_type='csv', metpy_interp_type='rbf',
                 metpy_search_radius=100000, metpy_min_neighbors=1,
                 metpy_gamma=0.25, kappa_star=5.052,
                 rbf_func='linear', rbf_smooth=0.1):  # Aumentado o valor padrão para 0.1
        """
        Initialize the LittleRProcessor
        
        Parameters:
        -----------
        timestamp : str
            Timestamp for the data (YYYY-MM-DD HH:MM:SS)
        output_dir : str
            Directory to save outputs
        lat_min, lat_max, lon_min, lon_max : float
            Geographical boundaries
        grid_resolution : tuple or None
            (lon_res, lat_res) in degrees for spatial grid interpolation
            If None, no grid interpolation is performed
        interpolation_method : str
            Method for spatial interpolation ('linear', 'cubic', 'nearest', 'metpy')
        output_type : str
            Type of output ('csv' para dados pontuais ou 'netcdf' para matriz espacial)
        metpy_interp_type : str
            Type of MetPy interpolation ('cressman', 'barnes', 'natural_neighbor', 'linear')
        metpy_search_radius : float
            Search radius in meters for MetPy interpolation
        metpy_min_neighbors : int
            Minimum number of neighbors for MetPy interpolation
        metpy_gamma : float
            Gamma parameter for Barnes interpolation
        kappa_star : float
            Kappa_star parameter for Barnes interpolation
        rbf_func : str
            Função RBF para interpolação (multiquadric, inverse, gaussian, linear, cubic, quintic, thin_plate)
        rbf_smooth : float
            Fator de suavização para interpolação RBF
        """
        self.timestamp = pd.to_datetime(timestamp)
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        
        self.grid_resolution = grid_resolution
        self.interpolation_method = interpolation_method
        self.output_type = output_type
        
        # Parâmetros para interpolação MetPy
        self.metpy_interp_type = metpy_interp_type
        self.metpy_search_radius = metpy_search_radius
        self.metpy_min_neighbors = metpy_min_neighbors
        self.metpy_gamma = metpy_gamma
        self.kappa_star = kappa_star
        
        # Parâmetros para RBF
        self.rbf_func = rbf_func
        self.rbf_smooth = rbf_smooth
        
        self.df = None
        self.grid_data = None

    def download_file(self, url):
        """Download file from URL with progress bar"""
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
            print(f"Arquivo não existe: {url}")
        return None

    def proc_row(self, row, sizes):
        """Process a row of fixed-width data according to size specifications"""
        begin = 0
        proc_row = []
        for i in range(len(sizes)):
            value = row[begin:begin + sizes[i]].strip() if begin < len(row) else ""
            proc_row.append(value)
            begin += sizes[i]
        return proc_row

    def process_data(self):
        """Download and process Little R data"""
        url = 'https://data.rda.ucar.edu/ds461.0/little_r/{year}/SURFACE_OBS:{year}{month}{day}{hour}'
        url = url.format(
            year=self.timestamp.strftime('%Y'),
            month=self.timestamp.strftime('%m'),
            day=self.timestamp.strftime('%d'),
            hour=self.timestamp.strftime('%H')
        )
        
        print(f"- Baixando de: {url}")
        file = self.download_file(url)
        if file is None:
            print("Erro ao baixar arquivo: " + url)
            return False
            
        data = file.read()
        print("- Processando dados")
        split_data = re.split(r"\n", data.decode('utf-8', errors='ignore'))
        split_array = np.array(split_data)
        
        # Split Header and Data
        header = split_array[::4][:-1]
        data = split_array[1::4]
        
        # Add to data Frame
        header_df = pd.DataFrame([self.proc_row(row, self.header_sizes) for row in header], columns=self.header_cols)
        data_df = pd.DataFrame([self.proc_row(row, self.data_sizes) for row in data], columns=self.data_cols)
        
        # Combine data frames
        self.df = pd.concat([header_df, data_df], axis=1)
        
        # Cast Latitude And Longitude To numeric
        print(f"- Aplicando pontos geográficos e recortando região [%.2f,%.2f,%.2f,%.2f]" % 
              (self.lat_min, self.lat_max, self.lon_min, self.lon_max))
        
        self.df['Latitude'] = pd.to_numeric(self.df['Latitude'], errors='coerce')
        self.df['Longitude'] = pd.to_numeric(self.df['Longitude'], errors='coerce')
        
        # Apply geopoints and crop the region
        geopoints = self.df[['Longitude','Latitude']].astype(float).apply(Point, axis=1)
        geopoints = gpd.GeoSeries(geopoints).cx[self.lon_min:self.lon_max, self.lat_min:self.lat_max]
        
        # Filter dataframe to points
        self.df = self.df.loc[geopoints.index]
        
        # Transform all data possible to numeric using pd.to_numeric
        for col in self.df.columns:
            try:
                self.df[col] = pd.to_numeric(self.df[col])
            except:
                pass
        
        return True

    def interpolate_to_grid(self):
        """Interpolate point data to a regular grid"""
        if self.df is None or self.grid_resolution is None:
            return False
            
        print(f"- Interpolando para grade com resolução: {self.grid_resolution}°")
        
        # Create grid coordinates
        lon_res, lat_res = self.grid_resolution
        lon_grid = np.arange(self.lon_min, self.lon_max + lon_res, lon_res)
        lat_grid = np.arange(self.lat_min, self.lat_max + lat_res, lat_res)
        
        # Create meshgrid
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        # Get coordinates and values
        points = self.df[['Longitude', 'Latitude']].values
        
        # Determine the interpolation method to use
        interpolation_method = self.interpolation_method
        if interpolation_method == 'metpy':
            interpolation_method = 'linear'  # Use 'linear' as fallback for 'metpy'

        # Initialize a dictionary to hold all gridded variables
        grid_data_dict = {
            'lon': lon_grid,
            'lat': lat_grid,
        }
        
        # Dictionary to store variable metadata (units)
        var_attrs = {}
        
        # Define valor único para nodata
        NODATA_VALUE = 99999
        
        # Filtrar para usar apenas as variáveis listadas em interesting_vars
        variables_to_interpolate = [col for col in self.df.columns 
                                   if col in self.interesting_vars.keys() and
                                   not col.endswith('-QC') and col not in ['Longitude', 'Latitude']]
        
        # print(f"- Interpolando {len(variables_to_interpolate)} variáveis de interesse")
        
        for var in variables_to_interpolate:
            if var in self.df.columns:
                # Extract variable name and unit
                var_clean = var
                unit = ""
                if '(' in var and ')' in var:
                    # Extract the variable name (without the unit) and the unit
                    parts = var.split('(')
                    var_clean = parts[0].strip()
                    unit = '(' + parts[1]  # Include the parentheses
                
                # Remove spaces and special characters to make NetCDF-safe names
                var_clean = var_clean.replace(' ', '_').replace('-', '_')
                
                try:
                    # Cópia dos dados para não alterar o dataframe original
                    var_values = self.df[var].copy()
                    
                    # Converter para numérico primeiro (caso esteja como string)
                    var_values = pd.to_numeric(var_values, errors='coerce')
                    
                    # Substituir valores fora dos thresholds definidos para NODATA_VALUE
                    min_threshold, max_threshold = self.interesting_vars[var]
                    var_values = var_values.mask((var_values < min_threshold) | (var_values > max_threshold), np.nan)
                    
                    # Substituir valores problemáticos por NaN
                    var_values = var_values.mask(var_values > 800000, np.nan)  # Para valores próximos a 888888
                    var_values = var_values.mask(var_values < -800000, np.nan) # Para valores próximos a -888888
                    var_values = var_values.mask(var_values > 90000, np.nan)   # Para valores próximos a 99999
                    var_values = var_values.mask(var_values < -90000, np.nan)  # Para valores próximos a -99999
                    
                    # Verificar mínimo e máximo após substituição
                    if not var_values.isna().all():
                        min_val = var_values.min() if not np.isnan(var_values.min()) else "Todos NaN"
                        max_val = var_values.max() if not np.isnan(var_values.max()) else "Todos NaN"
                        # print(f"- Interpolando variável: {var_clean} ({var}), min: {min_val}, max: {max_val}")
                        # print(f"  Thresholds aplicados: [{min_threshold}, {max_threshold}]")
                    
                    # Skip if all values are NaN or if we don't have enough valid data
                    if var_values.isna().all():
                        # print(f"- Ignorando variável {var_clean}: Todos os valores são NaN após filtro")
                        continue
                        
                    # Remove NaN values for interpolation
                    values = var_values.values
                    valid_indices = ~np.isnan(values)
                    if np.sum(valid_indices) > 3:  # Need at least 3 points for some interpolation methods
                        try:
                            grid_values = griddata(
                                points[valid_indices], 
                                values[valid_indices], 
                                (lon_mesh, lat_mesh), 
                                method=interpolation_method  # Use the local variable
                            )
                            # Preencher grade completa se necessário
                            grid_values = fullgrid(grid_values)

                            grid_data_dict[var_clean] = grid_values
                            # Store the original variable name, unit and nodata value
                            var_attrs[var_clean] = {
                                'long_name': var_clean,
                                'standard_name': var_clean.replace(' ', '_').replace('-', '_'),
                                'units': unit.strip('()') if unit else "",
                                '_FillValue': NODATA_VALUE,
                                'valid_range': f"{min_threshold}, {max_threshold}"
                            }
                        except Exception as e:
                            print(f"Erro na interpolação de {var}: {e}")
                    else:
                        # print(f"- Ignorando variável {var_clean}: Número insuficiente de pontos válidos ({np.sum(valid_indices)})")
                        pass
                except Exception as e:
                    print(f"Ignorando variável {var} (não numérica): {e}")
        
        if len(grid_data_dict) <= 2:  # Só temos lon e lat
            print("- Nenhuma variável numérica adequada para interpolação foi encontrada")
            return False
        
        # Convert to xarray Dataset
        self.grid_data = xr.Dataset(
            {var: (['lat', 'lon'], grid_data_dict[var]) 
             for var in grid_data_dict if var not in ['lon', 'lat']},
            coords={
                'lon': grid_data_dict['lon'],
                'lat': grid_data_dict['lat'],
                'time': self.timestamp
            }
        )
        
        # Add variable attributes (metadata) and set nodata values
        for var_name, attrs in var_attrs.items():
            if var_name in self.grid_data:
                # Substituir NaNs pelo valor NODATA
                self.grid_data[var_name] = self.grid_data[var_name].fillna(NODATA_VALUE)
                
                # Aplicar todos os atributos
                for attr_name, attr_value in attrs.items():
                    self.grid_data[var_name].attrs[attr_name] = attr_value
        
        # Adicionar atributos globais
        self.grid_data.attrs['title'] = f'UCAR Little-R Surface Observations - {self.timestamp.strftime("%Y-%m-%d %H:%M")}'
        self.grid_data.attrs['source'] = 'UCAR RDA ds461.0'
        self.grid_data.attrs['creation_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        self.grid_data.attrs['nodata_description'] = f'Valores {NODATA_VALUE} representam dados ausentes ou fora dos limites válidos'
        self.grid_data.attrs['variables_description'] = 'Aplicados thresholds conforme definido em interesting_vars'
        
        return True

    def interpolate_using_metpy(self, interp_type='rbf', minimum_neighbors=1,
                              search_radius=100000, hres=5000,
                              gamma=0.25, kappa_star=5.052):
        """
        Interpola dados pontuais para uma grade regular usando MetPy
        
        Parameters:
        -----------
        interp_type : str
            Tipo de interpolação a ser usada ('cressman', 'barnes', 'natural_neighbor', 'linear')
        minimum_neighbors : int
            Número mínimo de vizinhos a considerar na interpolação
        search_radius : float
            Raio de busca em metros para interpolação
        hres : float
            Resolução horizontal da grade em metros
        gamma : float
            Parâmetro gamma para interpolação de Barnes
        kappa_star : float
            Parâmetro kappa_star para interpolação de Barnes
        
        Returns:
        --------
        bool
            True se a interpolação foi bem-sucedida, False caso contrário
        """
        if not METPY_AVAILABLE:
            print("- Erro: MetPy não está instalado. Usando interpolação regular.")
            return self.interpolate_to_grid()
            
        if self.df is None:
            return False
            
        print(f"- Interpolando para grade usando MetPy ({interp_type})")
        
        # Define valor único para nodata
        NODATA_VALUE = 99999
        
        # Filtrar para usar apenas as variáveis listadas em interesting_vars
        variables_to_interpolate = [col for col in self.df.columns 
                                   if col in self.interesting_vars.keys() and
                                   not col.endswith('-QC') and col not in ['Longitude', 'Latitude']]
        
        print(f"- Interpolando {len(variables_to_interpolate)} variáveis de interesse")
        
        # Converter para GeoDataFrame e projetar para pseudo-UTM
        gdf = gpd.GeoDataFrame(
            self.df, 
            geometry=gpd.points_from_xy(self.df.Longitude, self.df.Latitude),
            crs="EPSG:4326"
        )
        gdf_utm = gdf.to_crs("EPSG:3857")  # Projeção Web Mercator
        
        # Extrair coordenadas x, y em UTM
        x = np.array([point.x for point in gdf_utm.geometry])
        y = np.array([point.y for point in gdf_utm.geometry])
        
        # Criar dicionário para armazenar dados da grade
        datasets = []
        
        # Interpolar cada variável
        for var in variables_to_interpolate:
            try:
                # Preparar dados para interpolação
                var_values = pd.to_numeric(self.df[var], errors='coerce')
                
                # Aplicar thresholds
                min_threshold, max_threshold = self.interesting_vars[var]
                var_values = var_values.mask((var_values < min_threshold) | (var_values > max_threshold), np.nan)
                
                # Tratar valores problemáticos
                var_values = var_values.mask(var_values > 800000, np.nan)
                var_values = var_values.mask(var_values < -800000, np.nan)
                var_values = var_values.mask(var_values > 90000, np.nan)
                var_values = var_values.mask(var_values < -90000, np.nan)
                
                # Se todos os valores são NaN, pular
                if var_values.isna().all():
                    # print(f"- Ignorando variável {var}: Todos os valores são NaN após filtro")
                    continue
                
                # Obter valores válidos
                values = var_values.values
                valid_indices = ~np.isnan(values)
                valid_x = x[valid_indices]
                valid_y = y[valid_indices]
                valid_values = values[valid_indices]
                
                # Verificar se temos pontos suficientes
                if len(valid_values) <= minimum_neighbors:
                    # print(f"- Ignorando variável {var}: Número insuficiente de pontos válidos ({len(valid_values)})")
                    continue
                    
                # Extrair nome e unidade da variável
                var_clean = var
                unit = ""
                if '(' in var and ')' in var:
                    parts = var.split('(')
                    var_clean = parts[0].strip()
                    unit = parts[1].strip(')')
                
                # Nome seguro para NetCDF
                var_name = var_clean.replace(' ', '_').replace('-', '_').upper()
                
                # print(f"- Interpolando variável: {var_name} (pontos válidos: {len(valid_values)})")
                
                # Realizar interpolação usando MetPy
                # print(f"- Usando interpolação MetPy: {interp_type}")
                try:
                    gx, gy, img = metpy_interpolate(
                        valid_x, valid_y, valid_values,
                        interp_type=interp_type,
                        minimum_neighbors=minimum_neighbors,
                        search_radius=search_radius,
                        hres=hres,
                        gamma=gamma,
                        kappa_star=kappa_star,
                        rbf_func=self.rbf_func,
                        rbf_smooth=self.rbf_smooth
                    )
                    print(f"- Interpolação MetPy concluída com sucesso para {var_name}")
                except np.linalg.LinAlgError as e:
                    if "singular" in str(e).lower() and interp_type == 'rbf':
                        print(f"- Erro de matriz singular na interpolação RBF para {var_name}")
                        print("  Tentando aumentar o parâmetro rbf_smooth para resolver o problema...")
                        
                        # Tentar aumentar o valor de rbf_smooth para resolver o problema
                        increased_smooth = max(0.1, self.rbf_smooth * 10)
                        
                        try:
                            print(f"  Tentando com rbf_smooth={increased_smooth}...")
                            gx, gy, img = metpy_interpolate(
                                valid_x, valid_y, valid_values,
                                interp_type=interp_type,
                                minimum_neighbors=minimum_neighbors,
                                search_radius=search_radius,
                                hres=hres,
                                gamma=gamma,
                                kappa_star=kappa_star,
                                rbf_func=self.rbf_func,
                                rbf_smooth=increased_smooth
                            )
                            print(f"- Interpolação MetPy concluída com sucesso após ajuste para {var_name}")
                        except Exception as inner_e:
                            # Se a tentativa com maior rbf_smooth falhar, tentar outro rbf_func
                            print(f"  Ainda com erro após aumentar rbf_smooth: {str(inner_e)}")
                            print("  Tentando com outra função RBF (multiquadric)...")
                            
                            try:
                                gx, gy, img = metpy_interpolate(
                                    valid_x, valid_y, valid_values,
                                    interp_type=interp_type,
                                    minimum_neighbors=minimum_neighbors,
                                    search_radius=search_radius,
                                    hres=hres,
                                    gamma=gamma,
                                    kappa_star=kappa_star,
                                    rbf_func='multiquadric',  # Tentar com outra função RBF
                                    rbf_smooth=increased_smooth
                                )
                                print(f"- Interpolação MetPy concluída com sucesso usando multiquadric para {var_name}")
                            except Exception as final_e:
                                print(f"- Todas as tentativas de interpolação RBF falharam para {var_name}")
                                print(f"  Erro final: {str(final_e)}")
                                raise
                    else:
                        # Se não for erro de matriz singular ou não for RBF, levantar o erro original
                        raise
                
                # Preencher grade completa se necessário
                img = fullgrid(img)

                # Calcular limites da grade em coordenadas geográficas
                min_x, max_x = gx.min(), gx.max()
                min_y, max_y = gy.min(), gy.max()
                
                # Converter de volta para coordenadas geográficas
                points = [Point(min_x, min_y), Point(max_x, max_y)]
                geo_points = gpd.GeoSeries(points, crs="EPSG:3857")
                geo_points = geo_points.to_crs("EPSG:4326")
                
                # Criar arrays de latitude e longitude
                lons = np.linspace(geo_points.geometry[0].x, geo_points.geometry[1].x, img.shape[1])
                lats = np.linspace(geo_points.geometry[0].y, geo_points.geometry[1].y, img.shape[0])
                
                # Criar Dataset do xarray
                ds = xr.Dataset(
                    {var_name: (['lat', 'lon'], img)},
                    coords={
                        'lon': lons,
                        'lat': lats,
                        'time': self.timestamp
                    }
                )
                
                # Adicionar metadados
                ds[var_name].attrs = {
                    'long_name': var_clean,
                    'units': unit,
                    '_FillValue': NODATA_VALUE,
                    'valid_range': f"{min_threshold}, {max_threshold}",
                    'interpolation_method': interp_type,
                    'search_radius_meters': search_radius,
                    'resolution_meters': hres
                }
                
                datasets.append(ds)
                
            except Exception as e:
                print(f"- Erro ao interpolar {var}: {e}")
        
        # Se não temos datasets, retornar falha
        if not datasets:
            print("- Nenhuma variável foi interpolada com sucesso")
            return False
        
        # Mesclar todos os datasets
        if len(datasets) > 1:
            # Remapear todos os datasets para as coordenadas do primeiro
            base_ds = datasets[0]
            remapped_datasets = [ds if ds is base_ds else remap_to_base(ds, base_ds) for ds in datasets]
            
            # Mesclar em um único dataset
            merged_ds = xr.merge(remapped_datasets)
        else:
            merged_ds = datasets[0]
        
        # Adicionar atributos globais
        merged_ds.attrs = {
            'title': f'UCAR Little-R Surface Observations - {self.timestamp.strftime("%Y-%m-%d %H:%M")}',
            'source': 'UCAR RDA ds461.0',
            'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'nodata_description': f'Valores {NODATA_VALUE} representam dados ausentes ou fora dos limites válidos',
            'variables_description': 'Aplicados thresholds conforme definido em interesting_vars',
            'interpolation_method': interp_type,
            'author': 'Helvecio Neto (2024) - helecioblneto@gmail.com/github.com/helecioneto'
        }
        
        # Substituir NaNs pelo valor NODATA
        for var in merged_ds.data_vars:
            merged_ds[var] = merged_ds[var].fillna(NODATA_VALUE)
        
        # Armazenar o resultado
        self.grid_data = merged_ds
        
        return True

    def save_data(self):
        """Save data in the specified format"""
        # Create filename base
        filename_base = f"SURFACE_OBS:{self.timestamp.strftime('%Y%m%d%H')}"
        
        if self.output_type == 'csv':
            # Save the point data as CSV
            point_csv_path = self.output_dir / f"{filename_base}_points.csv"
            self.df.to_csv(point_csv_path, index=False)
            print(f"- Dados pontuais salvos em CSV: {point_csv_path}")
        
        elif self.output_type in ['geojson', 'gpkg', 'shapefile']:
            # Converter para GeoDataFrame
            try:
                # Criar uma coluna geometry com os pontos
                geometry = gpd.points_from_xy(self.df['Longitude'], self.df['Latitude'])
                gdf = gpd.GeoDataFrame(self.df, geometry=geometry, crs="EPSG:4326")
                
                # Salvar no formato desejado
                if self.output_type == 'geojson':
                    output_path = self.output_dir / f"{filename_base}_points.geojson"
                    gdf.to_file(output_path, driver='GeoJSON')
                    print(f"- Dados pontuais salvos em GeoJSON: {output_path}")
                    
                elif self.output_type == 'gpkg':
                    try:
                        output_path = self.output_dir / f"{filename_base}_points.gpkg"
                        gdf.to_file(output_path, driver='GPKG', layer=f'surface_obs_{self.timestamp.strftime("%Y%m%d%H")}')
                        print(f"- Dados pontuais salvos em GeoPackage: {output_path}")
                    except Exception as gpkg_error:
                        if "undefined symbol: sqlite3_total_changes64" in str(gpkg_error):
                            print(f"- Erro de compatibilidade SQLite/GDAL ao salvar GPKG: {gpkg_error}")
                            print("- Tentando salvar em GeoJSON como alternativa...")
                            geojson_path = self.output_dir / f"{filename_base}_points.geojson"
                            gdf.to_file(geojson_path, driver='GeoJSON')
                            print(f"- Dados pontuais salvos em GeoJSON: {geojson_path}")
                        else:
                            raise
                        
                elif self.output_type == 'shapefile':
                    output_path = self.output_dir / f"{filename_base}_points.shp"
                    gdf.to_file(output_path, driver='ESRI Shapefile')
                    print(f"- Dados pontuais salvos em Shapefile: {output_path}")
                
            except Exception as e:
                print(f"- Erro ao salvar em formato espacial: {e}")
                # Fallback to CSV
                point_csv_path = self.output_dir / f"{filename_base}_points.csv"
                self.df.to_csv(point_csv_path, index=False)
                print(f"- Salvando dados pontuais em CSV como alternativa: {point_csv_path}")
        
        elif self.output_type == 'netcdf':
            # Interpolate to grid if not already done
            if self.grid_data is None and self.grid_resolution is not None:
                self.interpolate_to_grid()
            
            if self.grid_data is not None:
                # Save as NetCDF
                netcdf_path = self.output_dir / f"{filename_base}_grid.nc"
                self.grid_data.to_netcdf(netcdf_path)
                print(f"- Dados em grade salvos em NetCDF: {netcdf_path}")
            else:
                print("- Erro: Não foi possível criar dados em grade. Verifique a resolução da grade.")
                # Fallback to CSV
                point_csv_path = self.output_dir / f"{filename_base}_points.csv"
                self.df.to_csv(point_csv_path, index=False)
                print(f"- Salvando dados pontuais em CSV como alternativa: {point_csv_path}")

    def run(self):
        """Run the full processing pipeline"""
        if self.process_data():
            print(f"- Dados processados com sucesso: {len(self.df)} registros")
            
            # Interpolate to grid if NetCDF output is requested
            if self.output_type == 'netcdf' and self.grid_resolution is not None:
                if self.interpolation_method == 'metpy':
                    # Converter grid_resolution de graus para metros (aproximado no equador)
                    lon_res, lat_res = self.grid_resolution
                    # 1 grau ≈ 111 km no equador
                    hres = min(lon_res, lat_res) * 111000
                    
                    # Se o usuário escolheu 'metpy' como método, usamos o metpy_interp_type
                    if not self.interpolate_using_metpy(
                        hres=hres,
                        interp_type=self.metpy_interp_type,
                        search_radius=self.metpy_search_radius,
                        minimum_neighbors=self.metpy_min_neighbors,
                        gamma=self.metpy_gamma,
                        kappa_star=self.kappa_star
                    ):
                        print("- ERRO: A interpolação MetPy falhou. O processamento será interrompido.")
                        print("  Verifique os parâmetros e tente novamente.")
                        return False
                else:
                    if not self.interpolate_to_grid():
                        print("- Falha na interpolação para grade")
                        return False
            
            # Save data in the requested format
            self.save_data()
            
            return True
        return False


def get_default_timestamp():
    """Get the default timestamp (most recent 6-hourly timestamp from 24 hours ago)"""
    current_date = pd.Timestamp.now()
    current_date = current_date.replace(hour=(current_date.hour//6)*6, minute=0, second=0, microsecond=0)    
    return current_date - pd.Timedelta(hours=24)


if __name__ == '__main__':
    default_time = get_default_timestamp()
    
    epilog = f"Exemplo:\n\tpython spatialmet.py -t '{default_time.strftime('%Y-%m-%d %H:%M:%S')}'" \
            " -o ./output -lat_min -10 -lat_max 10 -lon_min -70 -lon_max -45 -grid 0.5 0.5 -file_type netcdf -i metpy\n\n" \
            "Nota: Os horários permitidos são apenas 00:00, 06:00, 12:00, 18:00\n" \
            "Desenvolvido por: Helecio-Neto (2024) - helecioblneto@gmail.com"

    parser = argparse.ArgumentParser(description='Download e Processamento de Dados Little R da RDA UCAR',
                                    formatter_class=argparse.RawTextHelpFormatter,
                                    epilog=epilog)
    
    parser.add_argument('-t', '--timestamp', dest='timestamp', type=str,
                        default=default_time.strftime("%Y-%m-%d %H:%M:%S"),
                        help=f'Horário para download dos dados (padrão: {default_time.strftime("%Y-%m-%d %H:%M:%S")})')
    
    parser.add_argument('-o', '--output', dest='output', type=str,
                        help='Diretório de saída (padrão: ./output)', default='./output')
    
    parser.add_argument('-lat_min', dest='lat_min', type=float,
                        help='Latitude mínima (padrão: -90)', default=-90)
    
    parser.add_argument('-lat_max', dest='lat_max', type=float,
                        help='Latitude máxima (padrão: 90)', default=90)
    
    parser.add_argument('-lon_min', dest='lon_min', type=float,
                        help='Longitude mínima (padrão: -180)', default=-180)
    
    parser.add_argument('-lon_max', dest='lon_max', type=float,
                        help='Longitude máxima (padrão: 180)', default=180)
    
    parser.add_argument('-grid', '--grid_resolution', dest='grid_resolution', type=float, nargs=2,
                        help='Resolução da grade [lon_res lat_res] em graus (padrão: sem grade)', default=None)
    
    parser.add_argument('-file_type', dest='output_type', type=str,
                    choices=['csv', 'netcdf', 'geojson', 'gpkg', 'shapefile'], default='csv',
                    help='Tipo de saída: csv/geojson/gpkg/shapefile (pontos originais) ou netcdf (grade espacial)')
    
    parser.add_argument('-i', '--interpolation', dest='interpolation', type=str,
                        choices=['linear', 'cubic', 'nearest', 'metpy'], default='linear',
                        help='Método de interpolação (padrão: linear)')
                        
    parser.add_argument('--metpy-interp', dest='metpy_interp_type', type=str,
                        choices=['rbf', 'cressman', 'barnes', 'natural_neighbor', 'linear'], default='rbf',
                        help='Tipo de interpolação MetPy (padrão: rbf)')
    
    parser.add_argument('--metpy-radius', dest='metpy_search_radius', type=float, default=100000,
                        help='Raio de busca em metros para interpolação MetPy (padrão: 100000)')
    
    parser.add_argument('--metpy-neighbors', dest='metpy_min_neighbors', type=int, default=1,
                        help='Número mínimo de vizinhos para interpolação MetPy (padrão: 1)')
    
    parser.add_argument('--metpy-gamma', dest='metpy_gamma', type=float, default=0.25,
                        help='Parâmetro gamma para interpolação Barnes (padrão: 0.25)')
    
    parser.add_argument('--metpy-kappa-star', dest='metpy_kappa_star', type=float, default=5.052,
                        help='Parâmetro kappa_star para interpolação Barnes (padrão: 5.052)')
        
    parser.add_argument('--gaussian', dest='apply_gaussian', type=lambda x: bool(strtobool(x)),
                         default=True, help='Aplicar filtro gaussiano aos dados interpolados')
    
    parser.add_argument('--gaussian-sigma', dest='gaussian_sigma', type=float, default=1.5,
                        help='Sigma para filtro gaussiano (padrão: 1.5)')
    
    parser.add_argument('--rbf-func', dest='rbf_func', type=str,
                        choices=['multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate'], 
                        default='linear',
                        help='Função RBF para interpolação (padrão: linear)')

    parser.add_argument('--rbf-smooth', dest='rbf_smooth', type=float, default=0.1,
                        help='Fator de suavização para interpolação RBF (padrão: 0.1)')
    
    args = parser.parse_args()
    
    # Create and run processor
    processor = LittleRProcessor(
        timestamp=args.timestamp,
        output_dir=args.output,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        grid_resolution=args.grid_resolution,
        interpolation_method=args.interpolation,
        output_type=args.output_type,
        metpy_interp_type=args.metpy_interp_type,
        metpy_search_radius=args.metpy_search_radius,
        metpy_min_neighbors=args.metpy_min_neighbors,
        metpy_gamma=args.metpy_gamma,
        kappa_star=args.metpy_kappa_star,  # Adicionado este parâmetro
        rbf_func=args.rbf_func,
        rbf_smooth=args.rbf_smooth
    )
    
    processor.run()