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
import warnings
warnings.filterwarnings("ignore")

# Configurar ambiente
os.environ.update({
    'CPL_DEBUG': 'OFF',
    'GDAL_PAM_ENABLED': 'NO',
    'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
    'GDAL_ERROR_ON_LIBJPEG_WARNING': 'FALSE'
})

# Importar MetPy para interpolação meteorológica
try:
    from metpy.interpolate import interpolate_to_grid as metpy_interpolate
    METPY_AVAILABLE = True
except ImportError:
    METPY_AVAILABLE = False
    print("Aviso: MetPy não está instalado. Para usar interpolação meteorológica avançada, instale com: pip install metpy")

# Configurar GDAL
try:
    from osgeo import gdal
    gdal.UseExceptions()
    gdal.PushErrorHandler('CPLQuietErrorHandler')
except ImportError:
    pass


# Funções utilitárias
def fullgrid(grid):
    """Preenche uma grade com NaNs nas bordas"""
    mask = np.isnan(grid)
    if not np.any(mask):
        return grid
    
    idx = np.where(~mask, np.arange(mask.shape[0])[:, None], 0)
    idy = np.where(~mask, np.arange(mask.shape[1]), 0)
    nan_idx = np.where(mask)
    grid[nan_idx] = grid[idx[nan_idx], idy[nan_idx]]
    
    return grid


def parse_variable_name(var):
    """Extrai nome e unidade de uma variável"""
    var_clean = var
    unit = ""
    if '(' in var and ')' in var:
        parts = var.split('(')
        var_clean = parts[0].strip()
        unit = parts[1].strip(')')
    
    var_name = var_clean.replace(' ', '_').replace('-', '_').upper()
    return var_clean, unit, var_name


def create_grid_coordinates(lon_min, lon_max, lat_min, lat_max, resolution):
    """Cria coordenadas de grade"""
    lon_res, lat_res = resolution
    lons = np.linspace(lon_min, lon_max, int((lon_max-lon_min)/lon_res)+1)
    lats = np.linspace(lat_min, lat_max, int((lat_max-lat_min)/lat_res)+1)
    return lons, lats


def get_utm_boundaries(lon_min, lon_max, lat_min, lat_max):
    """Obtém limites UTM a partir de coordenadas geográficas"""
    bbox_points = [
        Point(lon_min, lat_min),
        Point(lon_max, lat_max)
    ]
    bbox_gdf = gpd.GeoDataFrame(geometry=bbox_points, crs="EPSG:4326")
    bbox_utm = bbox_gdf.to_crs("EPSG:3857")
    
    min_x = bbox_utm.geometry[0].x
    min_y = bbox_utm.geometry[0].y
    max_x = bbox_utm.geometry[1].x
    max_y = bbox_utm.geometry[1].y
    
    return min_x, max_x, min_y, max_y


def points_to_utm(df):
    """Converte pontos para UTM"""
    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df.Longitude, df.Latitude),
        crs="EPSG:4326"
    )
    return gdf.to_crs("EPSG:3857")


class LittleRProcessor:
    """Processador para dados Little R da UCAR."""
    
    # Constantes
    NODATA_VALUE = 99999
    
    # Definições de colunas e tamanhos
    header_cols = ['Latitude', 'Longitude', 'ID', 'Name', 'Platform (FM‑Code)', 
                  'Source', 'Elevation', 'Valid fields', 'Num. errors', 'Num. warnings',
                  'Sequence number','Num. duplicates', 'Is sounding?','Is bogus?','Discard?',
                  'Unix time','Julian day','Date',
                  'SLP','SLP-QC', 'Ref Pressure','Ref Pressure-QC',
                  'Precip','Precip-QC', 'Daily Max T','Daily Max T-QC',
                  'Daily Min T','Daily Min T-QC', 'Night Min T','Night Min T-QC',
                  '3hr Pres Change','3hr Pres Change-QC', '24hr Pres Change','24hr Pres Change-QC',
                  'Cloud cover','Cloud cover-QC', 'Ceiling','Ceiling-QC',
                  'Precipitable water','Precipitable water-QC']
    
    header_sizes = [20,20,40,40,40,40,20,10,10,10,10,10,10,10,10,10,10,20,13,7,13,7,13,7,13,7,13,7,13,7,13,7,13,7,13,7,13,7,13,7]
    
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

    # Limiares para variáveis de interesse
    interesting_vars = {
        'Precip': (0, 1000),
        'Pressure (Pa)': (0, 100000),
        'Height (m)': (0, 5000),
        'Temperature (K)': (270, 300),
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
                 rbf_func='linear', rbf_smooth=0.1):
        """Inicializa o processador."""
        
        # Parâmetros básicos
        self.timestamp = pd.to_datetime(timestamp)
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Limites geográficos
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        
        # Parâmetros de processamento e saída
        self.grid_resolution = grid_resolution
        self.interpolation_method = interpolation_method
        self.output_type = output_type
        
        # Parâmetros de interpolação
        self.metpy_interp_type = metpy_interp_type
        self.metpy_search_radius = metpy_search_radius
        self.metpy_min_neighbors = metpy_min_neighbors
        self.metpy_gamma = metpy_gamma
        self.kappa_star = kappa_star
        self.rbf_func = rbf_func
        self.rbf_smooth = rbf_smooth
        
        # Estado
        self.df = None
        self.grid_data = None
        self.unique_timestamps = None
    
    def download_file(self, url):
        """Baixa arquivo da URL com barra de progresso."""
        response = requests.head(url)
        if response.status_code != 200:
            print(f"Arquivo não existe: {url}")
            return None
            
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        url_filename = url.split('/')[-1]  # Extract filename from URL
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024,
                   desc=f"- Baixando de: \t{url}\t")
        file_in_memory = BytesIO()
        
        for chunk in response.iter_content(chunk_size=1024):
            file_in_memory.write(chunk)
            progress_bar.update(len(chunk))
            
        progress_bar.close()
        file_in_memory.seek(0)
        return file_in_memory
    
    def process_data(self):
        """Baixa e processa dados Little R."""
        # Construir URL
        url_template = 'https://data.rda.ucar.edu/ds461.0/little_r/{year}/SURFACE_OBS:{year}{month}{day}{hour}'
        url = url_template.format(
            year=self.timestamp.strftime('%Y'),
            month=self.timestamp.strftime('%m'),
            day=self.timestamp.strftime('%d'),
            hour=self.timestamp.strftime('%H')
        )
        
        # print(f"- Baixando de: {url}")
        file = self.download_file(url)
        if file is None:
            return False
            
        # Processar dados
        data = file.read()
        print("- Convertendo dados para dataframe...")
        
        # Dividir e processar linhas
        split_data = re.split(r"\n", data.decode('utf-8', errors='ignore'))
        split_array = np.array(split_data)
        
        # Extrair cabeçalho e dados
        header = split_array[::4][:-1]
        data = split_array[1::4]
        
        # Processar linhas em dataframes
        header_df = pd.DataFrame([self._proc_row(row, self.header_sizes) for row in header], 
                                 columns=self.header_cols)
        data_df = pd.DataFrame([self._proc_row(row, self.data_sizes) for row in data], 
                               columns=self.data_cols)
        
        # Combinar dataframes
        self.df = pd.concat([header_df, data_df], axis=1)
        
        # Processar coordenadas e aplicar filtro geográfico
        print(f"- Convertendo para GeoDataframe e recortando região [%.2f,%.2f,%.2f,%.2f]..." % 
              (self.lat_min, self.lat_max, self.lon_min, self.lon_max))
        
        self._apply_geo_filter()
        self._convert_numeric_columns()
        self._create_datetime_column()
        
        return True
    
    def _proc_row(self, row, sizes):
        """Processa uma linha de largura fixa de acordo com as especificações de tamanho."""
        begin = 0
        proc_row = []
        
        for size in sizes:
            value = row[begin:begin + size].strip() if begin < len(row) else ""
            proc_row.append(value)
            begin += size
            
        return proc_row
    
    def _apply_geo_filter(self):
        """Aplica filtro geográfico aos dados."""
        self.df['Latitude'] = pd.to_numeric(self.df['Latitude'], errors='coerce')
        self.df['Longitude'] = pd.to_numeric(self.df['Longitude'], errors='coerce')
        
        # Filtrar por região geográfica
        geopoints = self.df[['Longitude','Latitude']].astype(float).apply(Point, axis=1)
        geopoints = gpd.GeoSeries(geopoints).cx[self.lon_min:self.lon_max, self.lat_min:self.lat_max]
        self.df = self.df.loc[geopoints.index]
    
    def _convert_numeric_columns(self):
        """Converte todas as colunas possíveis para formato numérico."""
        for col in self.df.columns:
            try:
                self.df[col] = pd.to_numeric(self.df[col])
            except:
                pass
    
    def _create_datetime_column(self):
        """Cria coluna datetime a partir da coluna Date."""
        if 'Date' in self.df.columns:
            try:
                self.df['datetime'] = pd.to_datetime(
                    self.df['Date'].astype(str), 
                    format='%Y%m%d%H%M%S', 
                    errors='coerce'
                )
                # print(f"- Coluna 'datetime' criada ({self.df['datetime'].nunique()} timestamps únicos)")
            except Exception as e:
                print(f"- Erro ao converter coluna Date para datetime: {e}")
    
    def aggregate_hourly_data(self):
        """Agrupa dados por hora e calcula médias horárias."""
        if 'datetime' not in self.df.columns:
            print("- Aviso: Coluna datetime não existe, não será feita agregação temporal")
            return False
        
        # print("- Realizando agregação horária dos dados...")
        
        # Truncar para hora
        self.df['hour'] = self.df['datetime'].dt.floor('H')
        
        # Obter variáveis numéricas para agregação
        numeric_vars = [col for col in self.df.columns 
                        if col in self.interesting_vars.keys() and
                        not col.endswith('-QC') and col not in ['Longitude', 'Latitude']]
        
        # Configurar agregação
        grouped = self.df.groupby(['hour', 'Longitude', 'Latitude'])
        agg_dict = {var: 'mean' for var in numeric_vars}
        agg_dict['ID'] = 'first'  # Manter ID da estação como identificador
        
        # Realizar agregação
        df_hourly = grouped.agg(agg_dict).reset_index()
        
        # Adicionar contagem de observações
        count_per_group = grouped.size().reset_index(name='num_obs')
        df_hourly = df_hourly.merge(count_per_group, on=['hour', 'Longitude', 'Latitude'])
        
        # print(f"- Dados agregados: {len(self.df)} observações → {len(df_hourly)} médias horárias")
        # print(f"- Timestamps únicos após agregação: {df_hourly['hour'].nunique()}")
        
        # Atualizar estado
        self.df = df_hourly
        self.unique_timestamps = sorted(self.df['hour'].unique())
        
        return True
    
    def clean_data(self):
        """Limpa os dados, removendo outliers e valores problemáticos."""
        # print("- Limpando dados e removendo outliers")
        
        for var in self.interesting_vars.keys():
            if var not in self.df.columns:
                continue
                
            # Obter limites e aplicar filtros
            min_threshold, max_threshold = self.interesting_vars[var]
            
            # Aplicar máscaras para valores inválidos
            masks = [
                (self.df[var] < min_threshold) | (self.df[var] > max_threshold),
                self.df[var] > 800000,    # Valores próximos a 888888
                self.df[var] < -800000,   # Valores próximos a -888888
                self.df[var] > 90000,     # Valores próximos a 99999
                self.df[var] < -90000     # Valores próximos a -99999
            ]
            
            # Aplicar todas as máscaras de uma vez
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = combined_mask | mask
                
            self.df[var] = self.df[var].mask(combined_mask, np.nan)
            
            # Mostrar estatísticas
            self._show_variable_stats(var)
    
    def _show_variable_stats(self, var):
        """Mostra estatísticas de uma variável após limpeza."""
        valid_count = self.df[var].notna().sum()
        
        if valid_count > 0:
            min_val = self.df[var].min() if not np.isnan(self.df[var].min()) else "Todos NaN"
            max_val = self.df[var].max() if not np.isnan(self.df[var].max()) else "Todos NaN"
            # print(f"  • {var}: {valid_count} valores válidos (min: {min_val}, max: {max_val})")
        else:
            # print(f"  • {var}: Nenhum valor válido após filtragem")
            pass
    
    def interpolate_to_grid(self):
        """Interpola dados pontuais para uma grade regular com dimensão temporal."""
        if self.df is None or self.grid_resolution is None:
            return False
            
        # print(f"- Interpolando para grade com resolução: {self.grid_resolution}°")
        
        # Verificar se temos dimensão temporal
        has_time_dimension = 'hour' in self.df.columns and len(self.df['hour'].unique()) > 1
        
        # Criar coordenadas da grade
        lon_grid, lat_grid = create_grid_coordinates(
            self.lon_min, self.lon_max, 
            self.lat_min, self.lat_max,
            self.grid_resolution
        )
        
        # Ajustar método de interpolação
        interpolation_method = 'linear' if self.interpolation_method == 'metpy' else self.interpolation_method
        
        # Variáveis a serem interpoladas
        variables = self._get_variables_to_interpolate()
        
        # Interpolar de acordo com a dimensão temporal
        if not has_time_dimension:
            return self._interpolate_single_time(
                self.df, lon_grid, lat_grid, variables, interpolation_method
            )
        
        # Interpolar com dimensão temporal
        return self._interpolate_with_time_dimension(
            lon_grid, lat_grid, variables, interpolation_method
        )
    
    def _get_variables_to_interpolate(self):
        """Retorna lista de variáveis para interpolação."""
        return [col for col in self.df.columns 
                if col in self.interesting_vars.keys() and
                not col.endswith('-QC') and col not in ['Longitude', 'Latitude', 'hour', 'datetime']]
    
    def _interpolate_with_time_dimension(self, lon_grid, lat_grid, variables, interpolation_method):
        """Interpola dados com dimensão temporal."""
        print(f"- Interpolando dados com dimensão temporal ({len(self.unique_timestamps)} timestamps)")
        
        # Lista para armazenar datasets para cada timestamp
        time_datasets = []
        
        # Interpolar para cada timestamp
        for timestamp in self.unique_timestamps:
            # print(f"  • Interpolando para {timestamp.strftime('%Y-%m-%d %H:%M')}")
            
            # Filtrar dados para este timestamp
            df_time = self.df[self.df['hour'] == timestamp]
            
            # Interpolar para este timestamp
            success, ds_time = self._interpolate_single_time(
                df_time, lon_grid, lat_grid, variables, interpolation_method, timestamp
            )
            
            if success:
                time_datasets.append(ds_time)
        
        # Verificar se temos datasets
        if not time_datasets:
            print("- Nenhum timestamp pôde ser interpolado com sucesso")
            return False
        
        # Combinar datasets e adicionar metadados
        self.grid_data = xr.concat(time_datasets, dim='time')
        self._add_global_attributes(True)
        
        return True
    
    def _interpolate_single_time(self, df_time, lon_grid, lat_grid, variables, 
                                interpolation_method, timestamp=None):
        """Interpola dados para um único timestamp."""
        # Criar meshgrid
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        # Obter coordenadas e valores
        points = df_time[['Longitude', 'Latitude']].values
        
        # Dicionários para armazenar dados e metadados
        grid_data = {'lon': lon_grid, 'lat': lat_grid}
        var_attrs = {}
        
        # Interpolar cada variável
        for var in variables:
            if var not in df_time.columns:
                continue
                
            # Processar nome e unidade
            var_clean, unit, _ = parse_variable_name(var)
            var_name = var_clean.replace(' ', '_').replace('-', '_').upper()
            
            # Obter limites
            min_threshold, max_threshold = self.interesting_vars.get(var, (0, 0))
            
            try:
                # Verificar valores válidos
                var_values = df_time[var]
                if var_values.isna().all():
                    continue
                    
                # Filtrar valores NaN
                values = var_values.values
                valid_indices = ~np.isnan(values)
                
                if np.sum(valid_indices) <= 3:
                    continue
                    
                # Realizar interpolação
                grid_values = griddata(
                    points[valid_indices], 
                    values[valid_indices], 
                    (lon_mesh, lat_mesh), 
                    method=interpolation_method
                )
                
                # Preencher NaNs nas bordas
                grid_values = fullgrid(grid_values)
                
                # Aplicar limites às variáveis
                grid_values = self.apply_variable_limits(grid_values, var_name)
                
                # Armazenar dados e metadados
                grid_data[var_name] = grid_values
                var_attrs[var_name] = {
                    'long_name': var_clean,
                    'standard_name': var_name,
                    'units': unit.strip('()') if unit else "",
                    '_FillValue': self.NODATA_VALUE,
                    'valid_range': f"{min_threshold}, {max_threshold}"
                }
            except Exception as e:
                print(f"Erro na interpolação de {var}: {e}")
        
        # Verificar se temos variáveis
        if len(grid_data) <= 2:
            print("  • Nenhuma variável numérica adequada para interpolação foi encontrada")
            return False, None
        
        # Criar dataset
        dataset = self._create_dataset(grid_data, var_attrs, timestamp)
        return True, dataset
    
    def _create_dataset(self, grid_data, var_attrs, timestamp=None):
        """Cria um dataset xarray a partir dos dados interpolados."""
        # Usar timestamp fornecido ou padrão
        time_value = timestamp if timestamp is not None else self.timestamp
        
        # Aplicar limites de valores para cada variável antes de criar o dataset
        for var in list(grid_data.keys()):
            if var not in ['lon', 'lat']:
                # Aplicar limites conforme definidos em interesting_vars
                grid_data[var] = self.apply_variable_limits(grid_data[var], var)
        
        # Criar dataset
        dataset = xr.Dataset(
            {var: (['lat', 'lon'], grid_data[var]) 
             for var in grid_data if var not in ['lon', 'lat']},
            coords={
                'lon': grid_data['lon'],
                'lat': grid_data['lat'],
                'time': [time_value]
            }
        )
        
        # Adicionar atributos às variáveis
        for var_name, attrs in var_attrs.items():
            if var_name in dataset:
                dataset[var_name] = dataset[var_name].fillna(self.NODATA_VALUE)
                dataset[var_name].attrs.update(attrs)
        
        return dataset
    
    def _add_global_attributes(self, has_time_dimension=False):
        """Adiciona atributos globais ao dataset."""
        self.grid_data.attrs.update({
            'title': 'UCAR Little-R Surface Observations' + 
                     (' - Multiple Timestamps' if has_time_dimension else ''),
            'source': 'UCAR RDA ds461.0',
            'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'nodata_value': str(self.NODATA_VALUE),
            'author': 'Helvecio Neto (2024) - helecioblneto@gmail.com'
        })
        
        if has_time_dimension:
            self.grid_data.attrs['temporal_aggregation'] = 'Dados agregados por médias horárias'
    
    def apply_variable_limits(self, data_array, var_name):
        """Aplica limites mínimos e máximos aos dados interpolados."""
        for orig_var, limits in self.interesting_vars.items():
            _, _, std_var = parse_variable_name(orig_var)
            if var_name == std_var:
                min_val, max_val = limits
                # Aplicar limites usando numpy clip
                data_array = np.clip(data_array, min_val, max_val)
                # print(f"  • Aplicando limites à variável {var_name}: [{min_val}, {max_val}]")
                break
        return data_array
    
    def interpolate_using_metpy(self, interp_type='rbf', minimum_neighbors=1,
                             search_radius=100000, hres=5000,
                             gamma=0.25, kappa_star=5.052):
        """Interpola dados pontuais para grade regular usando MetPy com dimensão temporal."""
        if not METPY_AVAILABLE:
            print("- Erro: MetPy não está instalado. Usando interpolação regular.")
            return self.interpolate_to_grid()
            
        if self.df is None:
            return False
            
        # print(f"- Interpolando para grade usando MetPy ({interp_type})")
        
        # Verificar dimensão temporal
        has_time_dimension = (hasattr(self, 'unique_timestamps') and 
                              self.unique_timestamps is not None and 
                              len(self.unique_timestamps) > 1)
        
        # Processar de acordo com dimensão temporal
        if not has_time_dimension:
            success, dataset = self._interpolate_metpy_single_time(
                self.df, interp_type, minimum_neighbors,
                search_radius, hres, gamma, kappa_star
            )
            if success:
                self.grid_data = dataset
                self._add_global_attributes()
                self.grid_data.attrs['interpolation_method'] = interp_type
                return True
            return False
            
        # Processar múltiplos timestamps
        return self._interpolate_metpy_with_time_dimension(
            interp_type, minimum_neighbors, search_radius, hres, gamma, kappa_star
        )
    
    def _interpolate_metpy_with_time_dimension(self, interp_type, minimum_neighbors, 
                                             search_radius, hres, gamma, kappa_star):
        """Interpola com MetPy para múltiplos timestamps."""
        print(f"- Interpolando dados com MetPy ({interp_type}) com ({len(self.unique_timestamps)} timestamps)...")
        
        time_datasets = []
        
        for timestamp in self.unique_timestamps:
            # print(f"  • Interpolando para {timestamp.strftime('%Y-%m-%d %H:%M')}")
            
            # Filtrar dados para este timestamp
            df_time = self.df[self.df['hour'] == timestamp]
            
            # Interpolar para este timestamp
            success, ds_time = self._interpolate_metpy_single_time(
                df_time, interp_type, minimum_neighbors,
                search_radius, hres, gamma, kappa_star,
                timestamp=timestamp
            )
            
            if success:
                time_datasets.append(ds_time)
        
        # Verificar se temos datasets
        if not time_datasets:
            print("- Nenhum timestamp pôde ser interpolado com sucesso")
            return False
        
        # Combinar datasets e adicionar metadados
        self.grid_data = xr.concat(time_datasets, dim='time')
        self._add_global_attributes(True)
        self.grid_data.attrs['interpolation_method'] = interp_type
        
        return True
    
    def _interpolate_metpy_single_time(self, df_time, interp_type, minimum_neighbors,
                                     search_radius, hres, gamma, kappa_star, timestamp=None):
        """Interpola dados para um único timestamp usando MetPy."""
        if not METPY_AVAILABLE:
            return False, None
        
        # Obter variáveis para interpolação
        variables = self._get_variables_to_interpolate()
        # print(f"  • Interpolando {len(variables)} variáveis de interesse")
        
        # Obter limites UTM
        min_x, max_x, min_y, max_y = get_utm_boundaries(
            self.lon_min, self.lon_max, self.lat_min, self.lat_max
        )
        
        # Calcular dimensões da grade
        x_size = max(int(np.ceil((max_x - min_x) / hres)), 10)
        y_size = max(int(np.ceil((max_y - min_y) / hres)), 10)
        
        # Converter pontos para UTM
        gdf_utm = points_to_utm(df_time)
        x = np.array([point.x for point in gdf_utm.geometry])
        y = np.array([point.y for point in gdf_utm.geometry])
        
        # Lista para armazenar datasets
        datasets = []
        
        # Interpolar cada variável
        for var in variables:
            try:
                # Processar nome e unidade
                var_clean, unit, var_name = parse_variable_name(var)
                
                # Obter valores válidos
                var_values = df_time[var]
                if var_values.isna().all():
                    continue
                    
                values = var_values.values
                valid_indices = ~np.isnan(values)
                valid_x = x[valid_indices]
                valid_y = y[valid_indices]
                valid_values = values[valid_indices]
                
                if len(valid_values) < minimum_neighbors:
                    continue
                
                # Obter limites
                min_threshold, max_threshold = self.interesting_vars.get(var, (0, 0))
                
                # Realizar interpolação
                ds = self._try_metpy_interpolation(
                    var_name, var_clean, unit, valid_x, valid_y, valid_values,
                    interp_type, hres, minimum_neighbors, search_radius, gamma, kappa_star,
                    min_threshold, max_threshold, df_time, valid_indices
                )
                
                if ds is not None:
                    datasets.append(ds)
                    
            except Exception as e:
                print(f"  • Erro ao interpolar {var}: {e}")
        
        # Verificar se temos datasets
        if not datasets:
            print("  • Nenhuma variável pôde ser interpolada com sucesso")
            return False, None
        
        # Mesclar datasets e adicionar dimensão temporal
        merged_ds = xr.merge(datasets)
        time_value = timestamp if timestamp is not None else self.timestamp
        merged_ds = merged_ds.expand_dims(time=[time_value])
        
        return True, merged_ds
    
    def _try_metpy_interpolation(self, var_name, var_clean, unit, valid_x, valid_y, valid_values,
                               interp_type, hres, minimum_neighbors, search_radius, gamma, kappa_star,
                               min_threshold, max_threshold, df_time, valid_indices):
        """Tenta interpolar usando MetPy, com fallback para scipy."""
        try:
            # Usar MetPy para interpolação
            gx, gy, img = metpy_interpolate(
                valid_x, valid_y, valid_values,
                interp_type=interp_type,
                hres=hres,
                minimum_neighbors=minimum_neighbors,
                search_radius=search_radius,
                gamma=gamma,
                kappa_star=kappa_star,
                rbf_func=self.rbf_func,
                rbf_smooth=self.rbf_smooth
            )
            
            # Reamostrar para coordenadas geográficas
            return self._resample_to_geographic(
                gx, gy, img, var_name, var_clean, unit, min_threshold, max_threshold, interp_type
            )
            
        except Exception as e:
            print(f"  • Erro na interpolação MetPy para {var_clean}: {e}")
            print(f"  • Tentando método alternativo para {var_clean}")
            
            # Tentar método alternativo
            return self._try_alternative_interpolation(
                df_time, valid_indices, valid_values, var_name, var_clean, unit,
                min_threshold, max_threshold, interp_type
            )
    
    def _resample_to_geographic(self, gx, gy, img, var_name, var_clean, unit, 
                              min_threshold, max_threshold, interp_type):
        """Reamostra dados de UTM para coordenadas geográficas."""
        # Criar grade geográfica
        lon_grid, lat_grid = create_grid_coordinates(
            self.lon_min, self.lon_max, self.lat_min, self.lat_max, self.grid_resolution
        )
        
        # Converter coordenadas UTM para geográficas
        gx_gdf = gpd.GeoDataFrame(
            geometry=[Point(x, 0) for x in gx[0]], crs="EPSG:3857"
        ).to_crs("EPSG:4326")
        gy_gdf = gpd.GeoDataFrame(
            geometry=[Point(0, y) for y in gy[:, 0]], crs="EPSG:3857"
        ).to_crs("EPSG:4326")
        
        x_lon = np.array([p.x for p in gx_gdf.geometry])
        y_lat = np.array([p.y for p in gy_gdf.geometry])
        
        # Criar dataset temporário
        temp_ds = xr.Dataset(
            {var_name: (['y', 'x'], img)},
            coords={'x': x_lon, 'y': y_lat}
        )
        
        # Reamostrar para coordenadas desejadas
        resampled = temp_ds.interp(x=lon_grid, y=lat_grid)
        img_final = resampled[var_name].values
        
        # Preencher grade
        img_final = fullgrid(img_final)
        
        # Aplicar limites definidos em interesting_vars
        img_final = self.apply_variable_limits(img_final, var_name)
        
        # Criar dataset final
        ds = xr.Dataset(
            {var_name: (['lat', 'lon'], img_final)},
            coords={'lon': lon_grid, 'lat': lat_grid}
        )
        
        # Adicionar metadados
        ds[var_name].attrs = {
            'long_name': var_clean,
            'units': unit,
            '_FillValue': self.NODATA_VALUE,
            'valid_range': f"{min_threshold}, {max_threshold}",
            'interpolation_method': interp_type
        }
        
        return ds
    
    def _try_alternative_interpolation(self, df_time, valid_indices, valid_values, 
                                      var_name, var_clean, unit, min_threshold, 
                                      max_threshold, interp_type):
        """Tenta interpolação alternativa usando scipy griddata."""
        try:
            # Criar grade com dimensões desejadas
            lon_grid, lat_grid = create_grid_coordinates(
                self.lon_min, self.lon_max, self.lat_min, self.lat_max, self.grid_resolution
            )
            lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
            
            # Preparar pontos e valores
            points = np.column_stack((
                df_time.Longitude.values[valid_indices], 
                df_time.Latitude.values[valid_indices]
            ))
            
            # Interpolar
            img_final = griddata(
                points,
                valid_values,
                (lon_mesh, lat_mesh),
                method='linear'
            )
            
            # Preencher grade
            img_final = fullgrid(img_final)
            
            # Aplicar limites definidos em interesting_vars
            img_final = self.apply_variable_limits(img_final, var_name)
            
            # Criar dataset
            ds = xr.Dataset(
                {var_name: (['lat', 'lon'], img_final)},
                coords={'lon': lon_grid, 'lat': lat_grid}
            )
            
            # Adicionar metadados
            ds[var_name].attrs = {
                'long_name': var_clean,
                'units': unit,
                '_FillValue': self.NODATA_VALUE,
                'valid_range': f"{min_threshold}, {max_threshold}",
                'interpolation_method': 'linear (fallback)'
            }
            
            return ds
            
        except Exception as e2:
            print(f"  • Método alternativo também falhou: {e2}")
            return None
    
    def save_data(self):
        """Salva dados no formato especificado."""
        # Determinar nome do arquivo
        filename_base = self._generate_filename_base()
        
        # Salvar no formato apropriado
        if self.output_type == 'csv':
            self._save_as_csv(filename_base)
        elif self.output_type in ['geojson', 'gpkg', 'shapefile']:
            self._save_as_spatial_format(filename_base)
        elif self.output_type == 'netcdf':
            self._save_as_netcdf(filename_base)
    
    def _generate_filename_base(self):
        """Gera base do nome do arquivo de saída."""
        has_time_dimension = (hasattr(self, 'unique_timestamps') and 
                            self.unique_timestamps is not None and 
                            len(self.unique_timestamps) > 1)
        
        if has_time_dimension and self.output_type == 'netcdf':
            try:
                start_date = min(self.unique_timestamps).strftime('%Y%m%d%H')
                end_date = max(self.unique_timestamps).strftime('%Y%m%d%H')
                return f"SURFACE_OBS:{start_date}_to_{end_date}"
            except (ValueError, AttributeError):
                pass
                
        return f"SURFACE_OBS:{self.timestamp.strftime('%Y%m%d%H')}"
    
    def _save_as_csv(self, filename_base):
        """Salva dados em formato CSV."""
        point_csv_path = self.output_dir / f"{filename_base}_points.csv"
        self.df.to_csv(point_csv_path, index=False)
        print(f"- Dados pontuais salvos em CSV: {point_csv_path}")
    
    def _save_as_spatial_format(self, filename_base):
        """Salva dados em formato espacial (GeoJSON, GeoPackage, Shapefile)."""
        try:
            # Criar GeoDataFrame
            geometry = gpd.points_from_xy(self.df['Longitude'], self.df['Latitude'])
            gdf = gpd.GeoDataFrame(self.df, geometry=geometry, crs="EPSG:4326")
            
            # Salvar no formato apropriado
            if self.output_type == 'geojson':
                self._save_as_geojson(gdf, filename_base)
            elif self.output_type == 'gpkg':
                self._save_as_geopackage(gdf, filename_base)
            elif self.output_type == 'shapefile':
                self._save_as_shapefile(gdf, filename_base)
                
        except Exception as e:
            print(f"- Erro ao salvar em formato espacial: {e}")
            # Fallback para CSV
            self._save_as_csv(filename_base)
    
    def _save_as_geojson(self, gdf, filename_base):
        """Salva dados em formato GeoJSON."""
        output_path = self.output_dir / f"{filename_base}_points.geojson"
        gdf.to_file(output_path, driver='GeoJSON')
        print(f"- Dados pontuais salvos em GeoJSON: {output_path}")
    
    def _save_as_geopackage(self, gdf, filename_base):
        """Salva dados em formato GeoPackage."""
        try:
            output_path = self.output_dir / f"{filename_base}_points.gpkg"
            layer_name = f'surface_obs_{self.timestamp.strftime("%Y%m%d%H")}'
            gdf.to_file(output_path, driver='GPKG', layer=layer_name)
            print(f"- Dados pontuais salvos em GeoPackage: {output_path}")
        except Exception as gpkg_error:
            if "undefined symbol: sqlite3_total_changes64" in str(gpkg_error):
                print(f"- Erro de compatibilidade SQLite/GDAL ao salvar GPKG: {gpkg_error}")
                print("- Tentando salvar em GeoJSON como alternativa...")
                self._save_as_geojson(gdf, filename_base)
            else:
                raise
    
    def _save_as_shapefile(self, gdf, filename_base):
        """Salva dados em formato Shapefile."""
        output_path = self.output_dir / f"{filename_base}_points.shp"
        gdf.to_file(output_path, driver='ESRI Shapefile')
        print(f"- Dados pontuais salvos em Shapefile: {output_path}")
    
    def _save_as_netcdf(self, filename_base):
        """Salva dados em formato NetCDF."""
        # Interpolar para grade se necessário
        if self.grid_data is None and self.grid_resolution is not None:
            self.interpolate_to_grid()
        
        if self.grid_data is not None:
            # Salvar como NetCDF
            netcdf_path = self.output_dir / f"{filename_base}_grid.nc"
            self.grid_data.to_netcdf(netcdf_path)
            print(f"- Dados em grade salvos em NetCDF: {netcdf_path}")
        else:
            print("- Erro: Não foi possível criar dados em grade.")
            # Fallback para CSV
            self._save_as_csv(filename_base)
    
    def run(self):
        """Executa todo o pipeline de processamento."""
        if not self.process_data():
            return False
            
        print(f"- Dados processados com sucesso: {len(self.df)} registros")
        
        # Processar para NetCDF se necessário
        if self.output_type == 'netcdf' and self.grid_resolution is not None:
            self.aggregate_hourly_data()
            self.clean_data()
            
            if self.interpolation_method == 'metpy':
                # Converter resolução para metros (aproximado no equador)
                lon_res, lat_res = self.grid_resolution
                hres = min(lon_res, lat_res) * 111000
                
                if not self.interpolate_using_metpy(
                    hres=hres,
                    interp_type=self.metpy_interp_type,
                    search_radius=self.metpy_search_radius,
                    minimum_neighbors=self.metpy_min_neighbors,
                    gamma=self.metpy_gamma,
                    kappa_star=self.kappa_star
                ):
                    print("- ERRO: A interpolação MetPy falhou.")
                    return False
            else:
                if not self.interpolate_to_grid():
                    print("- Falha na interpolação para grade")
                    return False
        
        # Salvar dados
        self.save_data()
        return True


def get_default_timestamp():
    """Obtém timestamp padrão (mais recente marca de 6 horas de 24 horas atrás)."""
    current_date = pd.Timestamp.now()
    current_date = current_date.replace(
        hour=(current_date.hour//6)*6, 
        minute=0, second=0, microsecond=0
    )    
    return current_date - pd.Timedelta(hours=24)


if __name__ == '__main__':
    default_time = get_default_timestamp()
    
    epilog = (
        f"Exemplo:\n\tpython spatialmet.py -t '{default_time.strftime('%Y-%m-%d %H:%M:%S')}'" 
        " -o ./output -lat_min -10 -lat_max 10 -lon_min -70 -lon_max -45 -grid 0.5 0.5 -file_type netcdf -i metpy\n\n"
        "Nota: Os horários permitidos são apenas 00:00, 06:00, 12:00, 18:00\n"
        "Desenvolvido por: Helecio-Neto (2024) - helecioblneto@gmail.com"
    )

    parser = argparse.ArgumentParser(
        description='Download e Processamento de Dados Little R da RDA UCAR',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=epilog
    )
    
    # Argumentos de processamento básico
    parser.add_argument('-t', '--timestamp', dest='timestamp', type=str,
                      default=default_time.strftime("%Y-%m-%d %H:%M:%S"),
                      help=f'Horário para download dos dados (padrão: {default_time.strftime("%Y-%m-%d %H:%M:%S")})')
    
    parser.add_argument('-o', '--output', dest='output', type=str,
                      help='Diretório de saída (padrão: ./output)', default='./output')
    
    # Argumentos de limite geográfico
    parser.add_argument('-lat_min', dest='lat_min', type=float,
                      help='Latitude mínima (padrão: -90)', default=-90)
    
    parser.add_argument('-lat_max', dest='lat_max', type=float,
                      help='Latitude máxima (padrão: 90)', default=90)
    
    parser.add_argument('-lon_min', dest='lon_min', type=float,
                      help='Longitude mínima (padrão: -180)', default=-180)
    
    parser.add_argument('-lon_max', dest='lon_max', type=float,
                      help='Longitude máxima (padrão: 180)', default=180)
    
    # Argumentos de processamento e saída
    parser.add_argument('-grid', '--grid_resolution', dest='grid_resolution', type=float, nargs=2,
                      help='Resolução da grade [lon_res lat_res] em graus (padrão: 1.0 1.0)',
                      default=(1.0, 1.0))
    
    parser.add_argument('-file_type', dest='output_type', type=str,
                      choices=['csv', 'netcdf', 'geojson', 'gpkg', 'shapefile'], default='csv',
                      help='Tipo de saída: csv/geojson/gpkg/shapefile (pontos originais) ou netcdf (grade espacial)')
    
    parser.add_argument('-i', '--interpolation', dest='interpolation', type=str,
                      choices=['linear', 'cubic', 'nearest', 'metpy'], default='linear',
                      help='Método de interpolação (padrão: linear)')
    
    # Argumentos específicos do MetPy
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
    
    parser.add_argument('--rbf-func', dest='rbf_func', type=str,
                      choices=['multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate'], 
                      default='linear',
                      help='Função RBF para interpolação (padrão: linear)')

    parser.add_argument('--rbf-smooth', dest='rbf_smooth', type=float, default=0.1,
                      help='Fator de suavização para interpolação RBF (padrão: 0.1)')
        
    args = parser.parse_args()
    
    # Criar e executar processador
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
        kappa_star=args.metpy_kappa_star,
        rbf_func=args.rbf_func,
        rbf_smooth=args.rbf_smooth
    )
    
    processor.run()