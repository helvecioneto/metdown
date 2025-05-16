# SPATIALMET - Biblioteca para Download e Processamento de Dados Meteorológicos UCAR

## Visão Geral

SpatialMet é uma biblioteca Python para download, processamento e visualização de dados meteorológicos de superfície do repositório UCAR (University Corporation for Atmospheric Research) no formato Little-R. 

A biblioteca oferece múltiplas funcionalidades:

- Download automático de observações meteorológicas de superfície
- Filtragem geográfica por coordenadas
- Interpolação espacial com múltiplos métodos (linear, cúbica, nearest, métodos especializados via MetPy)
- Exportação para diversos formatos geoespaciais (NetCDF, GeoJSON, GeoPackage, Shapefile)
- Manipulação de dados temporais com agregação horária
- Controle de qualidade e limpeza de dados

## Instalação

### Requisitos

- Python 3.8 ou superior
- Bibliotecas dependentes (instaladas automaticamente)

### Passos para Instalação

1. **Clone o repositório**

   ```bash
   git clone https://github.com/helvecioneto/ucar-processing/.git
   cd ucar-processing
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

A biblioteca pode ser utilizada através da linha de comando após a instalação:

**Exemplo básico: Download de dados para o dia 1 de janeiro de 2023, com resolução de 1 grau (aproximadamente 100 km)**

```bash
python -m spatialmet -t "2023-01-01 00:00:00" -o ./output -grid 1 1 -file_type netcdf
```

**Exemplo com região específica: Recorte geográfico do Brasil e resolução de 0.5 graus (aproximadamente 50 km)**

```bash
python -m spatialmet -t "2023-01-01 00:00:00" -o ./output -lat_min -45 -lat_max 13 -lon_min -82 -lon_max -30 -grid 0.5 0.5 -file_type netcdf
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

### Interpolação com MetPy e alta resolução (0.1 graus, aproximadamente 10 km) com método RBF usando função multiquadric para interpolação e raio de 50 km

```bash
python -m spatialmet -t "2023-07-15 12:00:00" -o ./output -lat_min -10 -lat_max 10 -lon_min -70 -lon_max -45 -grid 0.1 0.1 -file_type netcdf -i metpy --metpy-interp rbf --metpy-radius 50000 --rbf-func multiquadric
```

### Download de dados para região específica do Brasil

```bash
python -m spatialmet -t "2023-01-01 00:00:00" -o ./output -lat_min -45 -lat_max 13 -lon_min -82 -lon_max -30 -file_type gpkg
```

### Criação de arquivo shapefile para uso em SIG

```bash
python -m spatialmet -t "2023-01-01 00:00:00" -o ./output -lat_min -45 -lat_max 13 -lon_min -82 -lon_max -30 -file_type shapefile
```

## Processamento em Lote

Para processar vários horários sequencialmente:

**Bash:**
```bash
#!/bin/bash
for hour in 00 06 12 18; do
    python -m spatialmet -t "2023-01-01 $hour:00:00" -o ./output -grid 0.5 0.5 -file_type netcdf
done
```

**PowerShell:**
```powershell
@("00", "06", "12", "18") | ForEach-Object {
    python -m spatialmet -t "2023-01-01 $_:00:00" -o ./output -grid 0.5 0.5 -file_type netcdf
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

```
Neto, H. (2024). SpatialMet: Processamento de Dados Little-R da UCAR.
Email: helecioblneto@gmail.com
```

## Licença

Este código é disponibilizado para uso acadêmico e de pesquisa.

## Contato

Para questões, sugestões ou contribuições:
Helvecio Neto - helecioblneto@gmail.com

helvecioneto: Escreva isso no arquivo README.md

GitHub Copilot: # SPATIALMET - Biblioteca para Download e Processamento de Dados Meteorológicos UCAR

## Visão Geral

SpatialMet é uma biblioteca Python para download, processamento e visualização de dados meteorológicos de superfície do repositório UCAR (University Corporation for Atmospheric Research) no formato Little-R. 

A biblioteca oferece múltiplas funcionalidades:

- Download automático de observações meteorológicas de superfície
- Filtragem geográfica por coordenadas
- Interpolação espacial com múltiplos métodos (linear, cúbica, nearest, métodos especializados via MetPy)
- Exportação para diversos formatos geoespaciais (NetCDF, GeoJSON, GeoPackage, Shapefile)
- Manipulação de dados temporais com agregação horária
- Controle de qualidade e limpeza de dados

## Instalação

### Requisitos

- Python 3.6 ou superior
- Bibliotecas dependentes (instaladas automaticamente)

### Passos para Instalação

1. **Clone o repositório**

   ```bash
   git clone https://github.com/helecioblneto/ucar-processing.git
   cd ucar-processing
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

4. **Instalação do MetPy (opcional, mas recomendado para interpolação avançada)**

   ```bash
   pip install metpy
   ```

## Uso Básico

A biblioteca pode ser utilizada através da linha de comando após a instalação:

**Exemplo básico: Dados globais com resolução de 1 grau (aproximadamente 100 km) em formato NetCDF**

```bash
python -m spatialmet -t "2023-01-01 00:00:00" -o ./output -grid 1 1 -file_type netcdf
```

**Exemplo com região específica: Recorte geográfico do Brasil e resolução de 0.5 graus (aproximadamente 50 km)**

```bash
python -m spatialmet -t "2023-01-01 00:00:00" -o ./output -lat_min -10 -lat_max 10 -lon_min -70 -lon_max -45 -grid 0.5 0.5 -file_type netcdf
```

## Parâmetros Disponíveis

### Parâmetros Obrigatórios

| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `-t, --timestamp` | Horário para download (formato: YYYY-MM-DD HH:MM:SS) | 24h atrás, horário múltiplo de 6h |
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
| `-grid, --grid_resolution` | Resolução da grade em graus [lon_res lat_res] | Sem grade |
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

### Interpolação com MetPy com resolução de 1 grau (aproximadamente 100 km)

```bash
python -m spatialmet -t "2023-07-15 12:00:00" -o ./output -lat_min -30 -lat_max 0 -lon_min -60 -lon_max -40 -grid 1 1 -file_type netcdf -i metpy --metpy-interp rbf --metpy-radius 50000 --rbf-func multiquadric
```

### Download de dados para região específica do Brasil com resolução de 0.5 graus (aproximadamente 50 km)

```bash
python -m spatialmet -t "2023-01-01 00:00:00" -o ./output -lat_min -23.5 -lat_max -20 -lon_min -47 -lon_max -43 -grid 0.5 0.5 -file_type gpkg
```

### Criação de arquivo shapefile para uso em SIG (formatos disponíveis: geojson, gpkg, shapefile)

```bash
python -m spatialmet -t "2023-01-01 00:00:00" -o ./output -lat_min -30 -lat_max 5 -lon_min -75 -lon_max -30 -file_type gpkg
```

## Processamento em Lote

Para processar vários horários sequencialmente:

**Bash:**
```bash
#!/bin/bash
for hour in 00 06 12 18; do
    python -m spatialmet -t "2023-01-01 $hour:00:00" -o ./output -grid 1 1 -file_type netcdf
done
```

**PowerShell:**
```powershell
@("00", "06", "12", "18") | ForEach-Object {
    python -m spatialmet -t "2023-01-01 $_:00:00" -o ./output -grid 1 1 -file_type netcdf
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

```
Neto, H. (2024). SpatialMet: Processamento de Dados Little-R da UCAR.
Email: helecioblneto@gmail.com
```

## Licença

Este código é disponibilizado para uso acadêmico e de pesquisa.

## Contato

Para questões, sugestões ou contribuições:
Helvecio Neto - helecioblneto@gmail.com
