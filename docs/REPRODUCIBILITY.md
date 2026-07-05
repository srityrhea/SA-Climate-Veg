# Reproducibility notes

## Portable paths

The GitHub copies of the scripts and notebooks use paths relative to the repository
root. They do not depend on the original `F:\GeoAI\Xee` location.

## Numerical reproduction

Exact reproduction depends on:

- identical source-product versions;
- the same CMIP6 models, ensemble members, calendars, and scenario files;
- consistent spatial masks and regridding choices;
- the processed 0.10-degree hydroclimatic-zone raster;
- the same MODIS quality-control and annual compositing procedures.

## Scientific cautions

1. Historical temporal correlation is diagnostic for free-running GCM simulations and
   should not be interpreted as reproduction of observed interannual phasing.
2. Near-uniform final weights mean the screened weighted and equal-weight means are
   numerically similar.
3. Annual extreme-index scaling is coarser than bias correction applied to the full
   daily distribution.
4. Vegetation-climate correlation does not establish causality.
5. The short 2000/2001-2014 overlap limits future vegetation-response inference.

## Recommended public archive

Before manuscript publication, archive the following small derived products:

- model and input-file inventories;
- hydroclimatic-zone summary table;
- historical model-evaluation metrics;
- annual zone-level projection changes;
- trend and change-point summary tables;
- vegetation linkage and validation summary tables.

These tables are sufficient to verify the headline manuscript values without
redistributing restricted or very large source datasets.

