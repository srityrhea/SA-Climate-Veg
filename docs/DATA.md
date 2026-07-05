# Data requirements

## External datasets

| Dataset | Variables/products | Role |
|---|---|---|
| ERA5-Land | Daily precipitation, maximum temperature, minimum temperature | Historical gridded reference, hydroclimatic features, and ETCCDI-style indices |
| CMIP6 historical | Precipitation, maximum temperature, minimum temperature | Historical model evaluation |
| ScenarioMIP | SSP2-4.5 and SSP5-8.5 daily/annual climate variables | Future projections |
| SRTM | Elevation | Static hydroclimatic zoning predictor |
| MOD13A2 | NDVI and EVI | Vegetation trends and climate linkages |
| MOD17A2H | GPP | Productivity-index trends and climate linkages |
| MOD17A3HGF | NPP | Productivity-index trends and climate linkages |
| Administrative boundaries | South Asian country boundaries and dissolved study domain | Mapping and masking |

## Expected local structure

```text
data/
|-- era5_land/
|-- cmip6/
|   |-- historical/
|   |-- ssp245/
|   `-- ssp585/
|-- modis/
|   |-- ndvi_evi/
|   |-- gpp/
|   `-- npp/
|-- srtm/
`-- boundaries/
```

The historical baseline is 1985-2014. The zoning product uses a 0.10-degree land grid
and contains 45,447 valid land pixels in the processed study domain.

## Data not included

Large gridded files are intentionally excluded from version control. Do not commit raw
or intermediate NetCDF, GeoTIFF, GRIB, HDF, or shapefile datasets directly to GitHub.
Use an external archive with a DOI for reproducibility, or provide download and
preprocessing instructions for source products.

## Reference-data limitation

ERA5-Land is a reanalysis product rather than an independent observational truth.
Precipitation-sensitive conclusions should be interpreted with this limitation and,
where possible, tested against gauge or satellite products such as IMD, APHRODITE,
CHIRPS, GPM, or IMERG.

