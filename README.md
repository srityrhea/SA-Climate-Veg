# Zone-aware CMIP6 climate extremes and vegetation linkages across South Asia

This repository contains the analysis workflow supporting a study of hydroclimatic
regionalization, CMIP6 evaluation, projected climate extremes, and vegetation-climate
linkages across South Asia.

The workflow delineates seven hydroclimatic zones, evaluates historical CMIP6
simulations against ERA5-Land, constructs a variable-specific screened and
zone-bias-corrected projection ensemble, assesses selected ETCCDI-style indices, and
examines observed and potential future vegetation responses using MODIS products.

## Scientific scope

The repository covers:

1. Consensus hydroclimatic zoning using K-Means, Ward hierarchical clustering, and
   Gaussian mixture modelling.
2. Independent zoning sensitivity analysis using a self-organizing map.
3. Historical CMIP6 evaluation for precipitation, maximum temperature, and minimum
   temperature.
4. CMIP6 daily-data export and quality control.
5. Annual ETCCDI-style index calculation and index-level bias correction.
6. Variable-specific model screening and zone-level projection construction under
   SSP2-4.5 and SSP5-8.5.
7. A historical machine-learning reconstruction benchmark.
8. Physical-consistency diagnostics.
9. Sen slope, Mann-Kendall, pre-whitened Mann-Kendall, Pettitt change-point, and
   false-discovery-rate analyses.
10. Observed MODIS vegetation trends and vegetation-climate linkages.
11. Confidence-qualified potential future vegetation-response diagnostics.

## Repository structure

```text
.
|-- notebooks/       Numbered analysis notebooks
|-- scripts/         Standalone analyses and publication-figure scripts
|-- docs/            Workflow, data, and reproducibility documentation
|-- data/            Local input-data location; large files are not tracked
|-- outputs/         Generated results; large outputs are not tracked
|-- figures/         Selected publication figures, when released
|-- manuscript/      Manuscript availability and citation notes
|-- environment.yml  Conda environment specification
|-- config.example.yml
|-- CITATION.cff
`-- .gitignore
```

## Analysis order

The notebooks are numbered in their recommended execution order. Some later analyses
are implemented as standalone scripts because they were designed for complete
non-interactive reruns.

See [docs/WORKFLOW.md](docs/WORKFLOW.md) for the full dependency sequence and
[docs/DATA.md](docs/DATA.md) for required datasets.

## Environment

The project was developed in a Conda environment named `geo`.

```bash
conda env create -f environment.yml
conda activate south-asia-geoai
```

Run notebooks from the repository root:

```bash
jupyter lab
```

Standalone scripts can be run with:

```bash
python scripts/run_som_sensitivity.py
```

## Data availability

Raw ERA5-Land, CMIP6, MODIS, SRTM, NetCDF, GeoTIFF, and shapefile inputs are not
committed because of their size and, where applicable, provider-specific access
conditions. The repository documents the expected folder structure and source
products so users can reconstruct the analysis inputs.

## Important interpretation notes

- ERA5-Land is treated as a gridded reference product, not absolute observational
  truth.
- Temporal correlation and KGE are retained as historical diagnostics, but
  year-matched interannual phasing is not treated as a decisive performance criterion
  for free-running GCM simulations.
- Final model weights were close to uniform. The main projection contribution is
  variable-specific model screening, hydroclimatic stratification, and zone-level bias
  correction rather than weighting-induced divergence from an equal-weight mean.
- Extreme indices were corrected at the annual-index level; daily quantile mapping was
  not applied to the final index products.
- Observed vegetation-climate linkages use a short overlap period, mainly 2000/2001
  through 2014. Future vegetation results should therefore be interpreted as
  confidence-qualified potential responses rather than deterministic forecasts.

## Reproducibility status

The repository contains the analysis code used for the manuscript. Exact numerical
reproduction requires the external datasets and intermediate files described in the
documentation. Before public release, add persistent links or DOIs for archived input
inventories and derived tabular results where permitted.

## Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff).

