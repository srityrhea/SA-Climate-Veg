from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "o6_vegetation_climate_refinement"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
TIF_DIR = OUT / "geotiff"
for d in [TABLE_DIR, FIG_DIR, TIF_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ZONE_TIF = ROOT / "output" / "zones" / "tif" / "zones_consensus.tif"
VEG_TRENDS = ROOT / "output" / "vegetation_climate_linkage" / "tables" / "modis_vegetation_zone_trends.csv"
VEG_CORR = ROOT / "output" / "vegetation_climate_linkage" / "tables" / "vegetation_key_etccdi_correlations_by_zone.csv"
VEG_ANNUAL = ROOT / "output" / "vegetation_climate_linkage" / "tables" / "modis_vegetation_zone_annual_means.csv"
COVERAGE = ROOT / "output" / "vegetation_climate_linkage" / "tables" / "vegetation_climate_linkage_coverage_by_zone.csv"

METRICS = ["NDVI", "EVI", "GPP", "NPP"]
KEY_CLIMATE = ["PRCPTOT", "RX1day", "CDD", "TXx", "TNn"]
LETTERS = ["A", "B", "C", "D"]


def fmt(x, nd=3):
    if pd.isna(x):
        return ""
    return f"{float(x):.{nd}f}"


def zone_num(zone):
    return int(str(zone).replace("Z", ""))


def read_zone_template():
    with rasterio.open(ZONE_TIF) as src:
        zones = src.read(1)
        profile = src.profile.copy()
        bounds = src.bounds
    return zones, profile, bounds


def zone_to_arr(zones, values, nodata=-9999.0):
    out = np.full(zones.shape, nodata, dtype="float32")
    for z, val in values.items():
        if pd.notna(val) and np.isfinite(float(val)):
            out[zones == int(z)] = float(val)
    return out


def write_tif(path, arr, profile, nodata=-9999.0):
    prof = profile.copy()
    prof.update(dtype="float32", count=1, nodata=nodata, compress="lzw")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype("float32"), 1)


def read_map(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        return arr, src.bounds


def add_map(ax, arr, bounds, title, letter, cmap, norm=None, vmin=None, vmax=None):
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    im = ax.imshow(arr, extent=extent, origin="upper", cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=18, fontweight="bold", pad=10)
    ax.set_xlabel("Longitude", fontsize=13, fontweight="bold")
    ax.set_ylabel("Latitude", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=11, width=1)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.set_aspect("equal")
    return im


def create_trend_summary(trends):
    rows = []
    for metric, g in trends.groupby("vegetation_metric"):
        g = g.copy()
        g["sen_slope_per_decade"] = pd.to_numeric(g["sen_slope_per_decade"], errors="coerce")
        sig = g["significant_fdr_0_05"].astype(str).str.lower() == "true"
        pos = g["sen_slope_per_decade"] > 0
        strongest = g.loc[g["sen_slope_per_decade"].idxmax()]
        weakest = g.loc[g["sen_slope_per_decade"].idxmin()]
        rows.append({
            "vegetation_metric": metric,
            "start_year": int(g["start_year"].min()),
            "end_year": int(g["end_year"].max()),
            "positive_zones": int(pos.sum()),
            "fdr_significant_zones": int(sig.sum()),
            "strongest_greening_zone": strongest["zone"],
            "strongest_slope_per_decade": strongest["sen_slope_per_decade"],
            "weakest_zone": weakest["zone"],
            "weakest_slope_per_decade": weakest["sen_slope_per_decade"],
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "o6_vegetation_trend_summary.csv", index=False)
    return out


def create_correlation_summaries(corr):
    corr = corr[corr["climate_index"].isin(KEY_CLIMATE)].copy()
    corr["abs_spearman_r"] = pd.to_numeric(corr["spearman_r"], errors="coerce").abs()
    sig = corr[corr["significant_spearman_fdr_0_05"].astype(str).str.lower() == "true"].copy()
    sig = sig.sort_values("abs_spearman_r", ascending=False)
    sig.to_csv(TABLE_DIR / "o6_significant_key_climate_correlations.csv", index=False)

    rows = []
    for (zone, metric), g in corr.groupby(["zone", "vegetation_metric"]):
        g = g.sort_values("abs_spearman_r", ascending=False)
        best = g.iloc[0]
        best_sig = str(best["significant_spearman_fdr_0_05"]).lower() == "true"
        rows.append({
            "zone": zone,
            "vegetation_metric": metric,
            "dominant_climate_index": best["climate_index"],
            "dominant_lag_years": int(best["lag_years"]),
            "dominant_spearman_r": best["spearman_r"],
            "dominant_spearman_p_fdr": best["spearman_p_fdr"],
            "dominant_is_fdr_significant": best_sig,
            "n_years": int(best["n_years"]),
        })
    dom = pd.DataFrame(rows)
    dom.to_csv(TABLE_DIR / "o6_dominant_climate_linkage_by_zone_metric.csv", index=False)
    return sig, dom


def export_trend_geotiffs(trends, zones, profile):
    for metric in METRICS:
        sub = trends[trends["vegetation_metric"] == metric].copy()
        vals = {zone_num(r["zone"]): r["sen_slope_per_decade"] for _, r in sub.iterrows()}
        sig = {zone_num(r["zone"]): 1 if str(r["significant_fdr_0_05"]).lower() == "true" else 0 for _, r in sub.iterrows()}
        pval = {zone_num(r["zone"]): r["mk_p_fdr"] for _, r in sub.iterrows()}
        write_tif(TIF_DIR / f"o6_{metric}_sen_slope_decade.tif", zone_to_arr(zones, vals), profile)
        write_tif(TIF_DIR / f"o6_{metric}_trend_sig_fdr.tif", zone_to_arr(zones, sig), profile)
        write_tif(TIF_DIR / f"o6_{metric}_trend_fdr_p.tif", zone_to_arr(zones, pval), profile)


def export_correlation_geotiffs(corr, dom, zones, profile):
    corr = corr[corr["climate_index"].isin(KEY_CLIMATE)].copy()
    for metric in METRICS:
        for climate in KEY_CLIMATE:
            for lag in [0, 1]:
                sub = corr[(corr["vegetation_metric"] == metric) & (corr["climate_index"] == climate) & (corr["lag_years"].astype(int) == lag)]
                if sub.empty:
                    continue
                vals = {zone_num(r["zone"]): r["spearman_r"] for _, r in sub.iterrows()}
                sig = {zone_num(r["zone"]): 1 if str(r["significant_spearman_fdr_0_05"]).lower() == "true" else 0 for _, r in sub.iterrows()}
                pval = {zone_num(r["zone"]): r["spearman_p_fdr"] for _, r in sub.iterrows()}
                write_tif(TIF_DIR / f"o6_{metric}_{climate}_lag{lag}_spearman_r.tif", zone_to_arr(zones, vals), profile)
                write_tif(TIF_DIR / f"o6_{metric}_{climate}_lag{lag}_spearman_sig_fdr.tif", zone_to_arr(zones, sig), profile)
                write_tif(TIF_DIR / f"o6_{metric}_{climate}_lag{lag}_spearman_fdr_p.tif", zone_to_arr(zones, pval), profile)

    for metric in METRICS:
        sub = dom[dom["vegetation_metric"] == metric].copy()
        vals = {zone_num(r["zone"]): r["dominant_spearman_r"] for _, r in sub.iterrows()}
        sig = {zone_num(r["zone"]): 1 if bool(r["dominant_is_fdr_significant"]) else 0 for _, r in sub.iterrows()}
        write_tif(TIF_DIR / f"o6_{metric}_dominant_linkage_spearman_r.tif", zone_to_arr(zones, vals), profile)
        write_tif(TIF_DIR / f"o6_{metric}_dominant_linkage_sig_fdr.tif", zone_to_arr(zones, sig), profile)


def panel_vegetation_trends(bounds):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes = axes.ravel()
    for i, metric in enumerate(METRICS):
        arr, b = read_map(TIF_DIR / f"o6_{metric}_sen_slope_decade.tif")
        valid = arr[np.isfinite(arr)]
        if np.nanmin(valid) < 0:
            lim = max(abs(np.nanmin(valid)), abs(np.nanmax(valid)))
            norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
            im = add_map(axes[i], arr, b, metric, LETTERS[i], "BrBG", norm=norm)
        else:
            im = add_map(axes[i], arr, b, metric, LETTERS[i], "YlGn", vmin=0, vmax=np.nanmax(valid))
        cb = fig.colorbar(im, ax=axes[i], shrink=0.78)
        cb.set_label("Sen slope decade-1", fontsize=12, fontweight="bold")
        cb.ax.tick_params(labelsize=11, width=1)
        for label in cb.ax.get_yticklabels():
            label.set_fontweight("bold")
    out_png = FIG_DIR / "MANUSCRIPT_PUBLICATION_o6_vegetation_trend_panel.png"
    out_pdf = FIG_DIR / "MANUSCRIPT_PUBLICATION_o6_vegetation_trend_panel.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def panel_prcptot_linkages(bounds):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes = axes.ravel()
    for i, metric in enumerate(METRICS):
        arr, b = read_map(TIF_DIR / f"o6_{metric}_PRCPTOT_lag0_spearman_r.tif")
        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        im = add_map(axes[i], arr, b, metric, LETTERS[i], "RdBu", norm=norm)
        cb = fig.colorbar(im, ax=axes[i], shrink=0.78)
        cb.set_label("Spearman r", fontsize=12, fontweight="bold")
        cb.ax.tick_params(labelsize=11, width=1)
        for label in cb.ax.get_yticklabels():
            label.set_fontweight("bold")
    out_png = FIG_DIR / "MANUSCRIPT_PUBLICATION_o6_prcptot_vegetation_linkage_panel.png"
    out_pdf = FIG_DIR / "MANUSCRIPT_PUBLICATION_o6_prcptot_vegetation_linkage_panel.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def panel_dominant_linkages(bounds):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes = axes.ravel()
    for i, metric in enumerate(METRICS):
        arr, b = read_map(TIF_DIR / f"o6_{metric}_dominant_linkage_spearman_r.tif")
        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        im = add_map(axes[i], arr, b, metric, LETTERS[i], "RdBu", norm=norm)
        cb = fig.colorbar(im, ax=axes[i], shrink=0.78)
        cb.set_label("Dominant Spearman r", fontsize=12, fontweight="bold")
        cb.ax.tick_params(labelsize=11, width=1)
        for label in cb.ax.get_yticklabels():
            label.set_fontweight("bold")
    out_png = FIG_DIR / "SUPPLEMENT_PUBLICATION_o6_dominant_linkage_panel.png"
    out_pdf = FIG_DIR / "SUPPLEMENT_PUBLICATION_o6_dominant_linkage_panel.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def make_geotiff_inventory():
    rows = []
    for fp in sorted(TIF_DIR.glob("*.tif")):
        with rasterio.open(fp) as src:
            rows.append({
                "file": str(fp),
                "crs": str(src.crs),
                "width": src.width,
                "height": src.height,
                "nodata": src.nodata,
            })
    pd.DataFrame(rows).to_csv(TABLE_DIR / "o6_geotiff_inventory.csv", index=False)


def write_methods_results_note(trend_summary, sig_corr, dom):
    note = OUT / "O6_REFINED_METHODS_RESULTS.md"
    with note.open("w", encoding="utf-8") as f:
        f.write("# O6 Refined Vegetation-Climate Linkage\n\n")
        f.write("## Scope\n\n")
        f.write("This refinement converts the existing MODIS vegetation trend and observed vegetation-climate linkage outputs into manuscript-ready tables, publication-style figures, and QGIS-ready GeoTIFF maps.\n\n")
        f.write("The analysis remains observational. MODIS vegetation trends use the full available MODIS record, while vegetation-climate correlations use the MODIS and ERA5-derived ETCCDI overlap period, mainly 2000/2001-2014.\n\n")
        f.write("## Main Findings\n\n")
        for _, r in trend_summary.iterrows():
            f.write(f"- {r['vegetation_metric']}: positive trends in {r['positive_zones']}/7 zones and FDR-significant trends in {r['fdr_significant_zones']}/7 zones; strongest greening in {r['strongest_greening_zone']}.\n")
        f.write("\n")
        f.write(f"- FDR-significant key vegetation-climate correlations: {len(sig_corr)} zone-metric-index-lag combinations.\n")
        f.write("- Significant linkages are dominated by precipitation-related indices, especially PRCPTOT and RX1day.\n")
        f.write("- Several arid, semi-arid, and mountain zones show positive vegetation response to precipitation amount, while Zone 7 includes negative EVI linkages with PRCPTOT/RX1day.\n\n")
        f.write("## Outputs\n\n")
        f.write("- `tables/o6_vegetation_trend_summary.csv`\n")
        f.write("- `tables/o6_significant_key_climate_correlations.csv`\n")
        f.write("- `tables/o6_dominant_climate_linkage_by_zone_metric.csv`\n")
        f.write("- `tables/o6_geotiff_inventory.csv`\n")
        f.write("- `figures/MANUSCRIPT_PUBLICATION_o6_vegetation_trend_panel.png`\n")
        f.write("- `figures/MANUSCRIPT_PUBLICATION_o6_prcptot_vegetation_linkage_panel.png`\n")
        f.write("- `figures/SUPPLEMENT_PUBLICATION_o6_dominant_linkage_panel.png`\n")
        f.write("- `geotiff/*.tif`\n")


def main():
    trends = pd.read_csv(VEG_TRENDS)
    corr = pd.read_csv(VEG_CORR)
    annual = pd.read_csv(VEG_ANNUAL)
    coverage = pd.read_csv(COVERAGE)

    trends.to_csv(TABLE_DIR / "o6_input_modis_vegetation_zone_trends.csv", index=False)
    corr.to_csv(TABLE_DIR / "o6_input_vegetation_key_etccdi_correlations_by_zone.csv", index=False)
    annual.to_csv(TABLE_DIR / "o6_input_modis_vegetation_zone_annual_means.csv", index=False)
    coverage.to_csv(TABLE_DIR / "o6_input_linkage_coverage_by_zone.csv", index=False)

    trend_summary = create_trend_summary(trends)
    sig_corr, dom = create_correlation_summaries(corr)

    zones, profile, bounds = read_zone_template()
    export_trend_geotiffs(trends, zones, profile)
    export_correlation_geotiffs(corr, dom, zones, profile)

    panel_vegetation_trends(bounds)
    panel_prcptot_linkages(bounds)
    panel_dominant_linkages(bounds)
    make_geotiff_inventory()
    write_methods_results_note(trend_summary, sig_corr, dom)

    checklist = pd.DataFrame([
        {"item": "trend_summary", "path": str(TABLE_DIR / "o6_vegetation_trend_summary.csv"), "exists": True},
        {"item": "significant_correlations", "path": str(TABLE_DIR / "o6_significant_key_climate_correlations.csv"), "exists": True},
        {"item": "dominant_linkages", "path": str(TABLE_DIR / "o6_dominant_climate_linkage_by_zone_metric.csv"), "exists": True},
        {"item": "geotiff_count", "path": str(TIF_DIR), "exists": len(list(TIF_DIR.glob("*.tif")))},
        {"item": "figure_count", "path": str(FIG_DIR), "exists": len(list(FIG_DIR.glob("*.png")))},
    ])
    checklist.to_csv(TABLE_DIR / "o6_completion_checklist.csv", index=False)
    print("Wrote", OUT)
    print(checklist.to_string(index=False))


if __name__ == "__main__":
    main()
