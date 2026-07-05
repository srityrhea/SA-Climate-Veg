# Analysis workflow

## Recommended sequence

| Step | File | Purpose |
|---|---|---|
| 1 | `notebooks/01_hydroclimatic_zone_validation.ipynb` | Validate the seven-zone consensus partition. |
| 2 | `scripts/run_som_sensitivity.py` | Test zoning robustness using a self-organizing map. |
| 3 | `notebooks/02_cmip6_historical_evaluation.ipynb` | Evaluate historical CMIP6 behaviour by variable and zone. |
| 4 | `notebooks/03_cmip6_daily_export.ipynb` | Export selected daily CMIP6 variables and scenarios. |
| 5 | `notebooks/04_daily_data_quality_control.ipynb` | Check completeness, units, ranges, and calendar consistency. |
| 6 | `notebooks/05_etccdi_index_bias_correction.ipynb` | Calculate and correct annual ETCCDI-style indices. |
| 7 | `notebooks/08_screened_projection_ensemble.ipynb` | Construct annual screened, zone-bias-corrected projections. |
| 8 | `notebooks/09_ml_reconstruction_benchmark.ipynb` | Compare historical reconstruction approaches. |
| 9 | `notebooks/10_physical_consistency_diagnostics.ipynb` | Screen annual ensemble outputs for basic physical consistency. |
| 10 | `notebooks/06_etccdi_spatial_trend_analysis.ipynb` | Produce spatial extreme-index changes and trends. |
| 11 | `scripts/run_trend_change_point_analysis.py` | Apply Sen slope, PW-MK, Pettitt, and FDR diagnostics. |
| 12 | `scripts/run_scenario_emergence.py` | Quantify scenario separation and approximate emergence. |
| 13 | `notebooks/07_observed_vegetation_climate_linkage.ipynb` | Link observed MODIS vegetation with ERA5-derived extremes. |
| 14 | `scripts/run_vegetation_climate_refinement.py` | Refine vegetation trends, linkages, and spatial outputs. |
| 15 | `scripts/run_future_vegetation_response.py` | Estimate confidence-qualified potential future responses. |
| 16 | `scripts/restyle_publication_figures.py` | Create consistent publication-ready figures. |

## Supporting figure scripts

- `scripts/create_study_area_map.py`
- `scripts/create_workflow_figure.py`
- `scripts/create_qgis_preview_panels.py`

## Naming history

Some internal output directories retain development identifiers such as `o3a`, `o5`,
`o6`, `o7`, and `o8`. These are implementation labels only and should not appear in
manuscript prose or figure captions.

## Execution notes

- Run notebooks from the repository root so relative paths resolve correctly.
- The scripts derive the project root from their own location.
- Several stages depend on intermediate files created by earlier stages.
- Do not interpret the machine-learning reconstruction benchmark as the future
  projection engine.

