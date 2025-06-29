# metdown - Biblioteca para Download e Processamento de Dados Meteorológicos

## Visão Geral

´metdown´ é uma biblioteca Python para download, processamento e visualização de dados meteorológicos de superfície do repositório UCAR (University Corporation for Atmospheric Research) no formato Little-R. 

**Fonte dos Dados**: https://rda.ucar.edu/datasets/d461000/

A biblioteca oferece múltiplas funcionalidades:

- Download automático de observações meteorológicas de superfície do dataset UCAR RDA ds461.0
- Filtragem geográfica por coordenadas
- Interpolação espacial com múltiplos métodos (linear, cúbica, nearest, métodos especializados via MetPy)
- Exportação para diversos formatos geoespaciais (NetCDF, GeoJSON, GeoPackage, Shapefile)
- Manipulação de dados temporais com agregação horária
- Controle de qualidade e limpeza de dados

## Fonte de Dados

Esta biblioteca utiliza o dataset **NCAR/UCAR Research Data Archive (RDA) ds461.0** 
Os dados são obtidos diretamente do servidor da UCAR em:

- **URL Base**: https://rda.ucar.edu/datasets/d461000/
- **Formato**: Little-R (formato usado em modelos meteorológicos)
- **Cobertura Temporal**: Dados históricos e em tempo quase real
- **Cobertura Espacial**: Global
- **Resolução Temporal**: Horários sinóticos (00:00, 06:00, 12:00, 18:00 UTC)
- **Tipo de Dados**: Observações de superfície e altitude de estações meteorológicas

### Citação dos Dados Fonte

```
NCAR/UCAR Research Data Archive. Integrated Global Radiosonde Archive (IGRA) and Other Upper-Air Data.
Dataset ID: ds461.0
URL: https://rda.ucar.edu/datasets/d461000/
```


## Instalação

### Requisitos

- Python 3.8 ou superior
- Bibliotecas dependentes (instaladas automaticamente)

### Passos para Instalação

1. **Clone o repositório**

   ```bash
   git clone https://github.com/helvecioneto/metdown/.git
   cd metdown
   ```

2. **Crie um ambiente virtual (recomendado)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```

3. **Instale as dependências**

   ```bash
   pip install -r requirements.txt
   ```


## Uso Básico

### 1. Exemplo Padrão (CSV)
```bash
# Download básico com formato padrão (CSV)
python -m metdown -t "2023-01-01 00:00:00" -o ./output
```

### 2. Exemplo com Dados Vetoriais (Shapefile)
```bash
# Download para região do Brasil com saída em Shapefile
python -m metdown -t "2023-01-01 00:00:00" -o ./output \
    -lat_min -45 -lat_max 13 -lon_min -82 -lon_max -30 \
    -file_type shapefile
```

### 3. Exemplo com Dados Raster (NetCDF com Interpolação MetPy usando método Cressman com raio de 300 km)
```bash
# Download para região do Brasil com interpolação MetPy (Cressman)
python -m metdown -t "2023-01-01 00:00:00" -o ./output \
    -lat_min -45 -lat_max 13 -lon_min -82 -lon_max -30 \
    -grid 0.5 0.5 -file_type netcdf \
    -i metpy --metpy-interp cressman --metpy-radius 300000
```

### 4. Processamento em Lote
```bash
# Processamento para os quatro horários sinóticos
for hour in 00 06 12 18; do
    python -m metdown -t "2023-01-01 $hour:00:00" -o ./output -file_type netcdf -grid 0.5 0.5
done
```

## Parâmetros Disponíveis

### Parâmetros Obrigatórios

| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `-t, --timestamp` | Horário para download (formato: YYYY-MM-DD HH:MM:SS) | 24h atrás da hora atual |
| `-o, --output` | Diretório de saída | ./output |

### Parâmetros de Região Geográfica

| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `-lat_min` | Latitude mínima | -90 |
| `-lat_max` | Latitude máxima | 90 |
| `-lon_min` | Longitude mínima | -180 |
| `-lon_max` | Longitude máxima | 180 |

### Parâmetros de Processamento e Saída

| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `-grid, --grid_resolution` | Resolução da grade em graus [lon_res lat_res] | 1 1 (graus) |
| `-file_type` | Tipo de arquivo de saída (csv, netcdf, geojson, gpkg, shapefile) | csv |
| `-i, --interpolation` | Método de interpolação (linear, cubic, nearest, metpy) | linear |

### Parâmetros Avançados de Interpolação MetPy

| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `--metpy-interp` | Tipo de interpolação MetPy (rbf, cressman, barnes, natural_neighbor, linear) | rbf |
| `--metpy-radius` | Raio de busca em metros | 100000 |
| `--metpy-neighbors` | Número mínimo de vizinhos | 1 |
| `--metpy-gamma` | Parâmetro gamma para Barnes | 0.25 |
| `--metpy-kappa-star` | Parâmetro kappa_star para Barnes | 5.052 |
| `--rbf-func` | Função RBF para interpolação (multiquadric, inverse, gaussian, linear, cubic, quintic, thin_plate) | linear |
| `--rbf-smooth` | Fator de suavização para RBF | 0.1 |

## Formatos de Saída Suportados

A biblioteca suporta os seguintes formatos de saída (configurados com `-file_type`):

1. **csv** - Arquivo CSV com dados pontuais
2. **netcdf** - Arquivo NetCDF com dados interpolados em grade (requer `-grid`)
3. **geojson** - Arquivo GeoJSON com pontos georreferenciados
4. **gpkg** - Arquivo GeoPackage com pontos georreferenciados
5. **shapefile** - Arquivo Shapefile ESRI com pontos georreferenciados

## Exemplos Avançados

```bash
# Interpolação MetPy com RBF (alta resolução)
python -m metdown -t "2023-07-15 12:00:00" -o ./output -lat_min -10 -lat_max 10 -lon_min -70 -lon_max -45 -grid 0.1 0.1 -i metpy --metpy-interp rbf --metpy-radius 50000

# Exportar como shapefile
python -m metdown -t "2023-01-01 00:00:00" -o ./output -lat_min -45 -lat_max 13 -lon_min -82 -lon_max -30 -file_type shapefile
```

## Processamento em Lote

**Bash:**
```bash
for hour in 00 06 12 18; do
    python -m metdown -t "2023-01-01 $hour:00:00" -o ./output -grid 0.5 0.5 -file_type netcdf
done
```

**PowerShell:**
```powershell
@("00", "06", "12", "18") | ForEach-Object {
    python -m metdown -t "2023-01-01 $_:00:00" -o ./output -grid 0.5 0.5 -file_type netcdf
}
```

## Variáveis Meteorológicas Disponíveis

A biblioteca processa as seguintes variáveis:

| Variável | Unidade | Limites Padrão |
|----------|---------|----------------|
| Pressure | Pa | 0-100000 |
| Height | m | 0-5000 |
| Temperature | K | 270-300 |
| Dew point | K | 0-400 |
| Wind speed | m/s | 0-100 |
| Wind direction | deg | 0-360 |
| Wind U | m/s | -100-100 |
| Wind V | m/s | -100-100 |
| Relative humidity | % | 0-100 |
| Thickness | m | 0-10000 |
| Precip | mm | 0-1000 |

## Estrutura dos Arquivos de Saída

### Arquivos NetCDF

Os arquivos NetCDF gerados incluem:
- Dimensões: latitude, longitude, tempo (se dados agregados)
- Variáveis: todas as variáveis meteorológicas disponíveis
- Atributos: unidades, descrição, limites válidos, método de interpolação
- Metadados: fonte dos dados, data de criação, autor

Exemplo de estrutura:
```
Dimensions:
  - lat: XX
  - lon: YY
  - time: Z (opcional)
Variables:
  - TEMPERATURE (K)
  - PRESSURE (Pa)
  - ...
```

### Formatos Vetoriais

Os formatos GeoJSON, GeoPackage e Shapefile incluem:
- Geometria do tipo ponto para cada observação
- Tabela de atributos com todas as variáveis meteorológicas
- Coordenadas em WGS84 (EPSG:4326)

## Interpolação Espacial

A biblioteca oferece quatro métodos principais de interpolação:

1. **linear** - Interpolação linear (triangulação Delaunay)
2. **cubic** - Interpolação cúbica para superfícies mais suaves
3. **nearest** - Vizinho mais próximo para manter valores originais
4. **metpy** - Métodos meteorológicos específicos:
   - **rbf** - Funções de base radial (várias opções)
   - **cressman** - Método de Cressman para análise meteorológica
   - **barnes** - Método de Barnes para análise sinótica
   - **natural_neighbor** - Interpolação por vizinhos naturais
   
Para dados meteorológicos, recomenda-se o uso de `-i metpy` combinado com o parâmetro `--metpy-interp` apropriado.

## Solução de Problemas

### Erro ao baixar dados

```
Arquivo não existe: https://data.rda.ucar.edu/ds461.0/little_r/...
```

**Solução**: Apenas horários sinóticos (00:00, 06:00, 12:00, 18:00 UTC) estão disponíveis. Verifique o horário informado.

### Erro relacionado a SQLite

```
Erro de compatibilidade SQLite/GDAL ao salvar GPKG: undefined symbol: sqlite3_total_changes64
```

**Solução**: Há uma incompatibilidade entre as versões de SQLite e GDAL. A biblioteca tenta automaticamente salvar em GeoJSON como alternativa.

### MetPy não disponível

```
Aviso: MetPy não está instalado. Para usar interpolação meteorológica avançada, instale com: pip install metpy
```

**Solução**: Instale MetPy conforme indicado: `pip install metpy`

### Nenhuma variável pôde ser interpolada

```
Nenhuma variável pôde ser interpolada com sucesso
```

**Solução**: Verifique se existem dados na região e horário selecionados. Tente expandir os limites geográficos ou escolher outro horário.

## Limitações Conhecidas

- **Horários disponíveis**: Apenas 00:00, 06:00, 12:00 e 18:00 UTC
- **Memória**: Áreas muito grandes ou resoluções muito altas podem causar problemas de memória
- **Interpolação**: Regiões com poucos pontos podem gerar artefatos na interpolação
- **Internet**: Necessária conexão estável para download dos dados

## Integração com Outras Ferramentas

Os dados processados pela biblioteca podem ser facilmente utilizados em:

- **Python**: Análises com xarray, pandas, matplotlib
- **GIS**: Importação em QGIS, ArcGIS, gvSIG
- **Meteorologia**: Modelos numéricos, validação de previsões
- **Visualização**: Dashboards com PyViz, Matplotlib, Plotly

## Citação e Atribuição

### Para o Software
```
Neto, H. (2024). metdown: Processamento de Dados Little-R da UCAR.
Email: helecioblneto@gmail.com
```

### Para os Dados
```
NCAR/UCAR Research Data Archive. Integrated Global Radiosonde Archive (IGRA) and Other Upper-Air Data.
Dataset ID: ds461.0. Disponível em: https://rda.ucar.edu/datasets/d461000/
```

### Citação Completa Recomendada
```
Os dados meteorológicos utilizados neste trabalho foram obtidos do NCAR/UCAR Research Data Archive 
(RDA) dataset ds461.0 (https://rda.ucar.edu/datasets/d461000/) e processados utilizando a biblioteca 
metdown (Neto, H., 2024).
```

## Licença

Este código é disponibilizado para uso acadêmico e de pesquisa.

## Contato

Para questões, sugestões ou contribuições:
Helvecio Neto - helecioblneto@gmail.com