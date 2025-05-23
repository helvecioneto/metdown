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

# Constantes
NODATA_VALUE = 99999

# Definições de colunas e tamanhos
HEADER_COLS = ['Latitude', 'Longitude', 'ID', 'Name', 'Platform (FM‑Code)', 
              'Source', 'Elevation', 'Valid fields', 'Num. errors', 'Num. warnings',
              'Sequence number','Num. duplicates', 'Is sounding?','Is bogus?','Discard?',
              'Unix time','Julian day','Date',
              'SLP','SLP-QC', 'Ref Pressure','Ref Pressure-QC',
              'Precip','Precip-QC', 'Daily Max T','Daily Max T-QC',
              'Daily Min T','Daily Min T-QC', 'Night Min T','Night Min T-QC',
              '3hr Pres Change','3hr Pres Change-QC', '24hr Pres Change','24hr Pres Change-QC',
              'Cloud cover','Cloud cover-QC', 'Ceiling','Ceiling-QC',
              'Precipitable water','Precipitable water-QC']

HEADER_SIZES = [20,20,40,40,40,40,20,10,10,10,10,10,10,10,10,10,10,20,13,7,13,7,13,7,13,7,13,7,13,7,13,7,13,7,13,7,13,7,13,7]

DATA_COLS = ['Pressure (Pa)','Pressure (Pa)-QC',
            'Height (m)','Height (m)-QC',
            'Temperature (K)','Temperature (K)-QC',
            'Dew point (K)','Dew point (K)-QC',
            'Wind speed (m/s)','Wind speed (m/s)-QC',
            'Wind direction (deg)','Wind direction (deg)-QC',
            'Wind U (m/s)','Wind U (m/s)-QC',
            'Wind V (m/s)','Wind V (m/s)-QC',
            'Relative humidity (%)','Relative humidity (%)-QC',
            'Thickness (m)','Thickness (m)-QC']

DATA_SIZES = [13,7,13,7,13,7,13,7,13,7,13,7,13,7,13,7,13,7,13,7]

# Limiares para variáveis de interesse
INTERESTING_VARS = {
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

def download_file(url):
    """Baixa arquivo da URL com barra de progresso."""
    response = requests.head(url)
    if response.status_code != 200:
        print(f"Arquivo não existe: {url}")
        return None
        
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024,
               desc=f"- Baixando de: \t{url}\t")
    file_in_memory = BytesIO()
    
    for chunk in response.iter_content(chunk_size=1024):
        file_in_memory.write(chunk)
        progress_bar.update(len(chunk))
        
    progress_bar.close()
    file_in_memory.seek(0)
    return file_in_memory

def proc_row(row, sizes):
    """Processa uma linha de largura fixa de acordo com as especificações de tamanho."""
    begin = 0
    proc_row = []
    
    for size in sizes:
        value = row[begin:begin + size].strip() if begin < len(row) else ""
        proc_row.append(value)
        begin += size
        
    return proc_row

# def convert_numeric_columns(df):
#     """Converte todas as colunas possíveis para formato numérico, tratando valores especiais."""
#     # Valores que representam dados ausentes/inválidos em formatos meteorológicos
#     missing_values = [
#         '-888888.00000', '888888.00000', '-999999', '999999', 
#         '-99999', '99999', '-888888', '888888'
#     ]
    
#     for col in df.columns:
#         try:
#             # Substituir valores conhecidos por NaN antes da conversão
#             if df[col].dtype == 'object':  # Apenas para colunas de texto
#                 for missing_val in missing_values:
#                     df[col] = df[col].replace(missing_val, np.nan)
        
#             # Converter para numérico
#             df[col] = pd.to_numeric(df[col], errors='coerce')
#         except:
#             pass
    
#     return df

def create_datetime_columns(df):
    """Cria colunas datetime a partir da coluna Date."""
    if 'Date' in df.columns:
        try:
            # Converter para datetime com formato completo
            df['datetime'] = pd.to_datetime(
                df['Date'].astype(str), 
                format='%Y%m%d%H%M%S', 
                errors='coerce'
            )
            
            # Adicionar coluna para hora truncada (arredondada para a hora exata)
            df['datetime_hour'] = df['datetime'].dt.floor('H')
            
            print(f"- Coluna 'datetime' criada ({df['datetime'].nunique()} timestamps únicos)")
            print(f"- Coluna 'datetime_hour' criada ({df['datetime_hour'].nunique()} horas únicas)")
        except Exception as e:
            print(f"- Erro ao converter coluna Date para datetime: {e}")
    
    return df

def apply_geo_filter(df, lon_min, lon_max, lat_min, lat_max):
    """Aplica filtro geográfico aos dados."""
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    
    # Filtrar por região geográfica
    geopoints = df[['Longitude','Latitude']].astype(float).apply(Point, axis=1)
    geopoints = gpd.GeoSeries(geopoints).cx[lon_min:lon_max, lat_min:lat_max]
    return df.loc[geopoints.index]

def clean_data(df):
    """Limpa os dados, removendo outliers e valores problemáticos."""
    print("- Limpando dados e removendo outliers")
    
    for var in INTERESTING_VARS.keys():
        if var not in df.columns:
            continue

        # Convert to numeric
        df[var] = pd.to_numeric(df[var], errors='coerce')
            
        # Obter limites e aplicar filtros
        min_threshold, max_threshold = INTERESTING_VARS[var]
        
        # Aplicar máscaras para valores inválidos
        masks = [
            (df[var] < min_threshold) | (df[var] > max_threshold),
            df[var] > 800000,    # Valores próximos a 888888
            df[var] < -800000,   # Valores próximos a -888888
            df[var] > 90000,     # Valores próximos a 99999
            df[var] < -90000     # Valores próximos a -99999
        ]
        
        # Aplicar todas as máscaras de uma vez
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask = combined_mask | mask
            
        df[var] = df[var].mask(combined_mask, np.nan)
        
        # # Mostrar estatísticas
        # valid_count = df[var].notna().sum()
        # if valid_count > 0:
        #     min_val = df[var].min() if not np.isnan(df[var].min()) else "Todos NaN"
        #     max_val = df[var].max() if not np.isnan(df[var].max()) else "Todos NaN"
            # print(f"  • {var}: {valid_count} valores válidos, min: {min_val}, max: {max_val}")
    
    return df

def aggregate_hourly_data(df):
    """Agrupa dados por hora e calcula médias horárias."""
    if 'datetime' not in df.columns:
        print("- Aviso: Coluna datetime não existe, não será feita agregação temporal")
        return df, None
    
    # Truncar para hora
    df['hour'] = df['datetime'].dt.floor('H')
    
    # Obter variáveis numéricas para agregação
    numeric_vars = [col for col in df.columns 
                    if col in INTERESTING_VARS.keys() and
                    not col.endswith('-QC') and col not in ['Longitude', 'Latitude']]
    
    # Configurar agregação
    grouped = df.groupby(['hour', 'Longitude', 'Latitude'])
    agg_dict = {var: 'mean' for var in numeric_vars}
    agg_dict['ID'] = 'first'  # Manter ID da estação como identificador
    
    # Realizar agregação
    df_hourly = grouped.agg(agg_dict).reset_index()
    
    # Adicionar contagem de observações
    count_per_group = grouped.size().reset_index(name='num_obs')
    df_hourly = df_hourly.merge(count_per_group, on=['hour', 'Longitude', 'Latitude'])
    
    # Timestamps únicos
    unique_timestamps = sorted(df_hourly['hour'].unique())
    
    return df_hourly, unique_timestamps

def process_data(timestamp, lon_min, lon_max, lat_min, lat_max):
    """Baixa e processa dados Little R."""
    # Construir URL
    url_template = 'https://data.rda.ucar.edu/ds461.0/little_r/{year}/SURFACE_OBS:{year}{month}{day}{hour}'
    url = url_template.format(
        year=timestamp.strftime('%Y'),
        month=timestamp.strftime('%m'),
        day=timestamp.strftime('%d'),
        hour=timestamp.strftime('%H')
    )
    
    file = download_file(url)
    if file is None:
        return None
        
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
    header_df = pd.DataFrame([proc_row(row, HEADER_SIZES) for row in header], 
                             columns=HEADER_COLS)
    data_df = pd.DataFrame([proc_row(row, DATA_SIZES) for row in data], 
                           columns=DATA_COLS)
    
    # Combinar dataframes
    df = pd.concat([header_df, data_df], axis=1)
    
    # Processar coordenadas e aplicar filtro geográfico
    print(f"- Convertendo para GeoDataframe e recortando região [%.2f,%.2f,%.2f,%.2f]..." % 
          (lat_min, lat_max, lon_min, lon_max))
    
    df = apply_geo_filter(df, lon_min, lon_max, lat_min, lat_max)
    # df = convert_numeric_columns(df)
    df = create_datetime_columns(df)
    
    return df

def apply_variable_limits(data_array, var_name):
    """Aplica limites mínimos e máximos aos dados interpolados."""
    for orig_var, limits in INTERESTING_VARS.items():
        _, _, std_var = parse_variable_name(orig_var)
        if var_name == std_var:
            min_val, max_val = limits
            # Aplicar limites usando numpy clip
            data_array = np.clip(data_array, min_val, max_val)
            break
    return data_array

def resample_to_geographic(gx, gy, img, var_name, var_clean, unit, 
                           min_threshold, max_threshold, interp_type,
                           lon_min, lon_max, lat_min, lat_max, grid_resolution):
    """Reamostra dados de UTM para coordenadas geográficas."""
    # Criar grade geográfica
    lon_grid, lat_grid = create_grid_coordinates(
        lon_min, lon_max, lat_min, lat_max, grid_resolution
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
    img_final = apply_variable_limits(img_final, var_name)
    
    # Criar dataset final
    ds = xr.Dataset(
        {var_name: (['lat', 'lon'], img_final)},
        coords={'lon': lon_grid, 'lat': lat_grid}
    )
    
    # Adicionar metadados
    ds[var_name].attrs = {
        'long_name': var_clean,
        'units': unit,
        '_FillValue': NODATA_VALUE,
        'valid_range': f"{min_threshold}, {max_threshold}",
        'interpolation_method': interp_type
    }
    
    return ds

def try_alternative_interpolation(df_time, valid_indices, valid_values, 
                                 var_name, var_clean, unit, min_threshold, 
                                 max_threshold, interp_type,
                                 lon_min, lon_max, lat_min, lat_max, grid_resolution):
    """Tenta interpolação alternativa usando scipy griddata."""
    try:
        # Criar grade com dimensões desejadas
        lon_grid, lat_grid = create_grid_coordinates(
            lon_min, lon_max, lat_min, lat_max, grid_resolution
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
        img_final = apply_variable_limits(img_final, var_name)
        
        # Criar dataset
        ds = xr.Dataset(
            {var_name: (['lat', 'lon'], img_final)},
            coords={'lon': lon_grid, 'lat': lat_grid}
        )
        
        # Adicionar metadados
        ds[var_name].attrs = {
            'long_name': var_clean,
            'units': unit,
            '_FillValue': NODATA_VALUE,
            'valid_range': f"{min_threshold}, {max_threshold}",
            'interpolation_method': 'linear (fallback)'
        }
        
        return ds
        
    except Exception as e2:
        print(f"  • Método alternativo também falhou: {e2}")
        return None

def try_metpy_interpolation(var_name, var_clean, unit, valid_x, valid_y, valid_values,
                          interp_type, hres, minimum_neighbors, search_radius, gamma, kappa_star,
                          min_threshold, max_threshold, df_time, valid_indices,
                          lon_min, lon_max, lat_min, lat_max, grid_resolution,
                          rbf_func, rbf_smooth):
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
            rbf_func=rbf_func,
            rbf_smooth=rbf_smooth
        )
        
        # Reamostrar para coordenadas geográficas
        return resample_to_geographic(
            gx, gy, img, var_name, var_clean, unit, 
            min_threshold, max_threshold, interp_type,
            lon_min, lon_max, lat_min, lat_max, grid_resolution
        )
        
    except Exception as e:
        print(f"  • Erro na interpolação MetPy para {var_clean}: {e}")
        print(f"  • Tentando método alternativo para {var_clean}")
        
        # Tentar método alternativo
        return try_alternative_interpolation(
            df_time, valid_indices, valid_values, var_name, var_clean, unit,
            min_threshold, max_threshold, interp_type,
            lon_min, lon_max, lat_min, lat_max, grid_resolution
        )

def interpolate_metpy_single_time(df_time, interp_type, minimum_neighbors,
                               search_radius, hres, gamma, kappa_star, timestamp,
                               lon_min, lon_max, lat_min, lat_max, grid_resolution,
                               rbf_func, rbf_smooth):
    """Interpola dados para um único timestamp usando MetPy."""
    if not METPY_AVAILABLE:
        return False, None
    
    # Obter variáveis para interpolação
    variables = [col for col in df_time.columns 
                 if col in INTERESTING_VARS.keys() and
                 not col.endswith('-QC') and col not in ['Longitude', 'Latitude', 'hour', 'datetime']]
    
    # Obter limites UTM
    min_x, max_x, min_y, max_y = get_utm_boundaries(
        lon_min, lon_max, lat_min, lat_max
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
            min_threshold, max_threshold = INTERESTING_VARS.get(var, (0, 0))
            
            # Realizar interpolação
            ds = try_metpy_interpolation(
                var_name, var_clean, unit, valid_x, valid_y, valid_values,
                interp_type, hres, minimum_neighbors, search_radius, gamma, kappa_star,
                min_threshold, max_threshold, df_time, valid_indices,
                lon_min, lon_max, lat_min, lat_max, grid_resolution,
                rbf_func, rbf_smooth
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
    merged_ds = merged_ds.expand_dims(time=[timestamp])
    
    return True, merged_ds

def interpolate_metpy_with_time_dimension(df, unique_timestamps, interp_type, minimum_neighbors, 
                                       search_radius, hres, gamma, kappa_star,
                                       lon_min, lon_max, lat_min, lat_max, grid_resolution,
                                       rbf_func, rbf_smooth):
    """Interpola com MetPy para múltiplos timestamps."""
    print(f"- Interpolando dados com MetPy ({interp_type}) com ({len(unique_timestamps)} timestamps)...")
    
    time_datasets = []
    
    for timestamp in unique_timestamps:
        # Filtrar dados para este timestamp
        df_time = df[df['hour'] == timestamp]
        
        # Interpolar para este timestamp
        success, ds_time = interpolate_metpy_single_time(
            df_time, interp_type, minimum_neighbors,
            search_radius, hres, gamma, kappa_star, timestamp,
            lon_min, lon_max, lat_min, lat_max, grid_resolution,
            rbf_func, rbf_smooth
        )
        
        if success:
            time_datasets.append(ds_time)
    
    # Verificar se temos datasets
    if not time_datasets:
        print("- Nenhum timestamp pôde ser interpolado com sucesso")
        return None
    
    # Combinar datasets e adicionar metadados
    grid_data = xr.concat(time_datasets, dim='time')
    
    # Adicionar atributos globais
    grid_data.attrs.update({
        'title': 'UCAR Little-R Surface Observations - Multiple Timestamps',
        'source': 'UCAR RDA ds461.0',
        'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'nodata_value': str(NODATA_VALUE),
        'temporal_aggregation': 'Dados agregados por médias horárias',
        'interpolation_method': interp_type,
        'author': 'Helvecio Neto (2024) - helecioblneto@gmail.com'
    })
    
    return grid_data

def interpolate_using_metpy(df, unique_timestamps, interp_type='rbf', minimum_neighbors=1,
                          search_radius=100000, hres=5000, gamma=0.25, kappa_star=5.052,
                          lon_min=-180, lon_max=180, lat_min=-90, lat_max=90, 
                          grid_resolution=(1.0, 1.0), rbf_func='linear', rbf_smooth=0.1):
    """Interpola dados pontuais para grade regular usando MetPy com dimensão temporal."""
    if not METPY_AVAILABLE:
        print("- Erro: MetPy não está instalado. Usando interpolação regular.")
        return interpolate_to_grid(df, unique_timestamps, 'linear', 
                                 lon_min, lon_max, lat_min, lat_max, grid_resolution)
        
    if df is None:
        return None
        
    # Verificar dimensão temporal
    has_time_dimension = unique_timestamps is not None and len(unique_timestamps) > 1
    
    # Processar de acordo com dimensão temporal
    if not has_time_dimension:
        success, dataset = interpolate_metpy_single_time(
            df, interp_type, minimum_neighbors,
            search_radius, hres, gamma, kappa_star, 
            pd.Timestamp.now(),  # timestamp fictício se não houver temporal
            lon_min, lon_max, lat_min, lat_max, grid_resolution,
            rbf_func, rbf_smooth
        )
        
        if success:
            # Adicionar atributos globais
            dataset.attrs.update({
                'title': 'UCAR Little-R Surface Observations',
                'source': 'UCAR RDA ds461.0',
                'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'nodata_value': str(NODATA_VALUE),
                'interpolation_method': interp_type,
                'author': 'Helvecio Neto (2024) - helecioblneto@gmail.com'
            })
            
            return dataset
        return None
    
    # Processar múltiplos timestamps
    return interpolate_metpy_with_time_dimension(
        df, unique_timestamps, interp_type, minimum_neighbors, 
        search_radius, hres, gamma, kappa_star,
        lon_min, lon_max, lat_min, lat_max, grid_resolution,
        rbf_func, rbf_smooth
    )

def interpolate_to_grid(df, unique_timestamps, interpolation_method='linear',
                       lon_min=-180, lon_max=180, lat_min=-90, lat_max=90, 
                       grid_resolution=(1.0, 1.0)):
    """Interpola dados pontuais para uma grade regular."""
    if df is None or grid_resolution is None:
        return None
    
    # Verificar se temos dimensão temporal
    has_time_dimension = unique_timestamps is not None and len(unique_timestamps) > 1
    
    # Criar coordenadas da grade
    lon_grid, lat_grid = create_grid_coordinates(
        lon_min, lon_max, lat_min, lat_max, grid_resolution
    )
    
    # Funções internas para reutilização
    def interpolate_single_time(df_time, timestamp=None):
        """Interpola dados para um único timestamp."""
        # Criar meshgrid
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        # Obter coordenadas e valores
        points = df_time[['Longitude', 'Latitude']].values
        
        # Dicionários para armazenar dados e metadados
        grid_data = {'lon': lon_grid, 'lat': lat_grid}
        var_attrs = {}
        
        # Variáveis a interpolar
        variables = [col for col in df_time.columns 
                     if col in INTERESTING_VARS.keys() and
                     not col.endswith('-QC') and col not in ['Longitude', 'Latitude', 'hour', 'datetime']]
        
        # Interpolar cada variável
        for var in variables:
            if var not in df_time.columns:
                continue
                
            # Processar nome e unidade
            var_clean, unit, _ = parse_variable_name(var)
            var_name = var_clean.replace(' ', '_').replace('-', '_').upper()
            
            # Obter limites
            min_threshold, max_threshold = INTERESTING_VARS.get(var, (0, 0))
            
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
                grid_values = apply_variable_limits(grid_values, var_name)
                
                # Armazenar dados e metadados
                grid_data[var_name] = grid_values
                var_attrs[var_name] = {
                    'long_name': var_clean,
                    'standard_name': var_name,
                    'units': unit.strip('()') if unit else "",
                    '_FillValue': NODATA_VALUE,
                    'valid_range': f"{min_threshold}, {max_threshold}"
                }
            except Exception as e:
                print(f"Erro na interpolação de {var}: {e}")
        
        # Verificar se temos variáveis
        if len(grid_data) <= 2:
            print("  • Nenhuma variável numérica adequada para interpolação foi encontrada")
            return False, None
        
        # Criar dataset
        time_value = timestamp if timestamp is not None else pd.Timestamp.now()
        
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
                dataset[var_name] = dataset[var_name].fillna(NODATA_VALUE)
                dataset[var_name].attrs.update(attrs)
        
        return True, dataset
    
    # Interpolar de acordo com a dimensão temporal
    if not has_time_dimension:
        success, dataset = interpolate_single_time(df)
        if success:
            # Adicionar atributos globais
            dataset.attrs.update({
                'title': 'UCAR Little-R Surface Observations',
                'source': 'UCAR RDA ds461.0',
                'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'nodata_value': str(NODATA_VALUE),
                'interpolation_method': interpolation_method,
                'author': 'Helvecio Neto (2024) - helecioblneto@gmail.com'
            })
            return dataset
        return None
    
    # Interpolar com dimensão temporal
    print(f"- Interpolando dados com dimensão temporal ({len(unique_timestamps)} timestamps)")
    
    # Lista para armazenar datasets para cada timestamp
    time_datasets = []
    
    # Interpolar para cada timestamp
    for timestamp in unique_timestamps:
        # Filtrar dados para este timestamp
        df_time = df[df['hour'] == timestamp]
        
        # Interpolar para este timestamp
        success, ds_time = interpolate_single_time(df_time, timestamp)
        
        if success:
            time_datasets.append(ds_time)
    
    # Verificar se temos datasets
    if not time_datasets:
        print("- Nenhum timestamp pôde ser interpolado com sucesso")
        return None
    
    # Combinar datasets
    grid_data = xr.concat(time_datasets, dim='time')
    
    # Adicionar atributos globais
    grid_data.attrs.update({
        'title': 'UCAR Little-R Surface Observations - Multiple Timestamps',
        'source': 'UCAR RDA ds461.0',
        'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'nodata_value': str(NODATA_VALUE),
        'temporal_aggregation': 'Dados agregados por médias horárias',
        'interpolation_method': interpolation_method,
        'author': 'Helvecio Neto (2024) - helecioblneto@gmail.com'
    })
    
    return grid_data

def generate_filename_base(timestamp, unique_timestamps, output_type):
    """Gera base do nome do arquivo de saída."""
    has_time_dimension = unique_timestamps is not None and len(unique_timestamps) > 1
    
    if has_time_dimension and output_type == 'netcdf':
        try:
            start_date = min(unique_timestamps).strftime('%Y%m%d%H')
            end_date = max(unique_timestamps).strftime('%Y%m%d%H')
            return f"SURFACE_OBS:{start_date}_to_{end_date}"
        except (ValueError, AttributeError):
            pass
        
    return f"SURFACE_OBS:{timestamp.strftime('%Y%m%d%H')}"

def save_data(hourly_groups, grid_data, timestamp, unique_timestamps, output_dir, output_type):
    """Salva dados no formato especificado com uma abordagem simplificada."""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verificar se temos dados para salvar
    if not hourly_groups:
        print("- ERRO: Não há dados para salvar")
        return
    
    print(f"- Salvando dados em formato: {output_type}")
    
    # Gerar nome base para arquivo
    filename_base = generate_filename_base(timestamp, unique_timestamps, output_type)
    
    # Caso especial para NetCDF (salvando por horário)
    if output_type == 'netcdf' and grid_data is not None:
        # Verificar se temos dimensão temporal no dataset
        if 'time' in grid_data.dims and len(grid_data.time) > 1:
            print(f"- Salvando {len(grid_data.time)} arquivos NetCDF (um por horário)")
            
            # Para cada timestamp no dataset
            for time_value in grid_data.time.values:
                # Extrair hora como string formatada
                time_str = pd.Timestamp(time_value).strftime('%Y%m%d%H')
                file_base = f"SURFACE_OBS_{time_str}0000"
                
                # Selecionar apenas este timestamp
                single_time_data = grid_data.sel(time=[time_value])
                
                # Criar caminho do arquivo
                netcdf_path = output_dir / f"{file_base}.nc"
                
                # Salvar
                single_time_data.to_netcdf(netcdf_path)
                print(f"- Dados para {pd.Timestamp(time_value).strftime('%Y-%m-%d %H:00')} salvos em NetCDF: {netcdf_path}")
        else:
            # Dataset sem dimensão temporal ou com apenas um timestamp
            netcdf_path = output_dir / f"{filename_base}.nc"
            grid_data.to_netcdf(netcdf_path)
            print(f"- Dados em grade salvos em NetCDF: {netcdf_path}")
        
        return
    elif output_type == 'netcdf':
        print("- Erro: Não há dados em grade para salvar como NetCDF, usando CSV como fallback")
        output_type = 'csv'
    
    # Processar cada grupo horário (já recebidos prontos)
    for hour, group in hourly_groups:
        # Definir nome de arquivo baseado na hora ou usar nome base
        if hour is not None:
            file_datetime = hour.strftime('%Y%m%d%H')
            file_base = f"SURFACE_OBS_{file_datetime}0000"
            time_str = hour.strftime('%Y-%m-%d %H:00')
        else:
            file_base = filename_base
            time_str = "todos os horários"
        
        # Remover colunas datetime e datetime_hour antes de salvar
        save_group = group.copy()
        columns_to_drop = [col for col in ['datetime', 'datetime_hour'] if col in save_group.columns]
        if columns_to_drop:
            save_group = save_group.drop(columns=columns_to_drop)
        
        # Salvar de acordo com o formato
        if output_type == 'csv':
            file_path = output_dir / f"{file_base}.csv"
            save_group.to_csv(file_path, index=False)
            print(f"- Dados para {time_str} salvos em CSV: {file_path}")
            
        else:
            # Criar GeoDataFrame para formatos espaciais (se já não for)
            if not isinstance(save_group, gpd.GeoDataFrame):
                geometry = gpd.points_from_xy(save_group['Longitude'], save_group['Latitude'])
                gdf = gpd.GeoDataFrame(save_group, geometry=geometry, crs="EPSG:4326")
            else:
                gdf = save_group
            
            # Salvar no formato apropriado
            try:
                if output_type == 'geojson':
                    file_path = output_dir / f"{file_base}.geojson"
                    gdf.to_file(file_path, driver='GeoJSON')
                    print(f"- Dados para {time_str} salvos em GeoJSON: {file_path}")
                    
                elif output_type == 'gpkg':
                    file_path = output_dir / f"{file_base}.gpkg"
                    layer_name = file_base.replace('SURFACE_OBS:', '').replace('SURFACE_OBS_', '')
                    gdf.to_file(file_path, driver='GPKG', layer=layer_name)
                    print(f"- Dados para {time_str} salvos em GeoPackage: {file_path}")
                    
                elif output_type == 'shapefile':
                    file_path = output_dir / f"{file_base}.shp"
                    gdf.to_file(file_path, driver='ESRI Shapefile')
                    print(f"- Dados para {time_str} salvos em Shapefile: {file_path}")
                    
            except Exception as e:
                print(f"- Erro ao salvar {output_type} para {time_str}: {e}")
                # Fallback para CSV em caso de erro
                file_path = output_dir / f"{file_base}.csv"
                save_group.to_csv(file_path, index=False)
                print(f"- Dados para {time_str} salvos em CSV (fallback): {file_path}")

def process_and_save(timestamp, output_dir="./output", 
                    lat_min=-90, lat_max=90, lon_min=-180, lon_max=180,
                    grid_resolution=None, interpolation_method='linear',
                    output_type='csv', metpy_interp_type='rbf',
                    metpy_search_radius=100000, metpy_min_neighbors=1,
                    metpy_gamma=0.25, kappa_star=5.052,
                    rbf_func='linear', rbf_smooth=0.1):
    """Função principal que executa todo o pipeline de processamento."""
    # Converter timestamp para datetime se for string
    timestamp = pd.to_datetime(timestamp)
    
    # Criar diretório de saída
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Baixar e processar dados
    df = process_data(timestamp, lon_min, lon_max, lat_min, lat_max)
    if df is None or len(df) == 0:
        print("- Erro: Não foi possível obter dados para processamento.")
        return False
    
    print(f"- Dados processados com sucesso: {len(df)} registros")
    
    
    
    # Garantir que a coluna datetime_hour existe para todos os formatos
    if 'datetime' in df.columns and 'datetime_hour' not in df.columns:
        df['datetime_hour'] = df['datetime'].dt.floor('H')
        print(f"- Coluna datetime_hour criada ({df['datetime_hour'].nunique()} horas únicas)")
    
    # Agrupar por hora - ÚNICO AGRUPAMENTO PARA TODOS OS FORMATOS
    hourly_groups = None
    has_hour_groups = 'datetime_hour' in df.columns and df['datetime_hour'].nunique() > 0
    
    if has_hour_groups:
        print(f"- Agrupando por hora ({df['datetime_hour'].nunique()} grupos)")
        hourly_groups = [(hour, group) for hour, group in df.groupby('datetime_hour')]
    else:
        # Sem agrupamento - tratar como um único grupo
        print("- Sem dados temporais para agrupar, usando arquivo único")
        hourly_groups = [(None, df)]

    # Variável para armazenar dados interpolados
    grid_data = None
    
    # Para NetCDF, interpolar cada grupo individualmente
    if output_type == 'netcdf' and grid_resolution is not None:
    
        print("- Realizando interpolação para cada grupo de hora")
        
        
        # Para cada grupo temporal, criar um arquivo NetCDF separado
        for hour, group in hourly_groups:
            # Limpar dados do grupo
            group = clean_data(group)

            if hour is None:
                # Se não temos hora definida, usar o timestamp do arquivo
                time_value = timestamp
                file_datetime = timestamp.strftime('%Y%m%d%H')
            else:
                time_value = hour
                file_datetime = hour.strftime('%Y%m%d%H')
                
            # Adicionar coluna 'hour' para compatibilidade com funções existentes
            group = group.copy()
            group['hour'] = time_value
                
            file_base = f"SURFACE_OBS_{file_datetime}0000"
            time_str = time_value.strftime('%Y-%m-%d %H:00')
            
            print(f"- Processando interpolação para {time_str} ({len(group)} registros)")
            
            # Lista para armazenar datasets para cada variável
            datasets = []
            
            # Interpolar este grupo específico
            if interpolation_method == 'metpy':
                # Converter resolução para metros (aproximado no equador)
                lon_res, lat_res = grid_resolution
                hres = min(lon_res, lat_res) * 111000
                
                # Criar dataset por hora usando MetPy
                success, hour_grid = interpolate_metpy_single_time(
                    group, 
                    metpy_interp_type, 
                    metpy_min_neighbors,
                    metpy_search_radius, 
                    hres, 
                    metpy_gamma, 
                    kappa_star, 
                    time_value,
                    lon_min, lon_max, lat_min, lat_max, 
                    grid_resolution,
                    rbf_func, 
                    rbf_smooth
                )
            else:
                # Criar dataset por hora usando interpolação padrão
                success, hour_grid = interpolate_single_time(
                    group, 
                    interpolation_method, 
                    lon_min, lon_max, lat_min, lat_max, 
                    grid_resolution,
                    timestamp=time_value
                )
            
            if not success or hour_grid is None:
                print(f"- Erro na interpolação de {time_str}")
                continue
                
            # Adicionar atributos globais
            hour_grid.attrs.update({
                'title': f'UCAR Little-R Surface Observations - {time_str}',
                'source': 'UCAR RDA ds461.0',
                'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'nodata_value': str(NODATA_VALUE),
                'interpolation_method': interpolation_method if interpolation_method != 'metpy' else metpy_interp_type,
                'author': 'Helvecio Neto (2024) - helecioblneto@gmail.com'
            })
            
            # Salvar arquivo NetCDF para esta hora
            netcdf_path = output_dir / f"{file_base}.nc"
            hour_grid.to_netcdf(netcdf_path)
            print(f"- Dados interpolados para {time_str} salvos em NetCDF: {netcdf_path}")
            
            # Acumular em grid_data para compatibilidade
            if grid_data is None:
                grid_data = hour_grid
            else:
                grid_data = xr.concat([grid_data, hour_grid], dim='time')
    
    # Salvar dados pontuais nos outros formatos
    save_data(hourly_groups, grid_data, timestamp, None, output_dir, output_type)
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
                      choices=['linear', 'cubic', 'nearest', 'metpy'], default='metpy',
                      help='Método de interpolação (padrão: linear)')
    
    # Argumentos específicos do MetPy
    parser.add_argument('--metpy-interp', dest='metpy_interp_type', type=str,
                      choices=['rbf', 'cressman', 'barnes', 'natural_neighbor', 'linear'], default='cressman',
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
    
    # Executar processamento
    process_and_save(
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