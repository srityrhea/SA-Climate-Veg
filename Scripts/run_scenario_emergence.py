from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.transform import from_origin


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "o7_scenario_emergence"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
TIF_DIR = OUT / "geotiff"
MAP_DIR = OUT / "maps"

for directory in [TABLE_DIR, FIG_DIR, TIF_DIR, MAP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

KEY_INDICES = ["PRCPTOT", "RX1day", "CDD", "TXx", "TNn"]
PERIODS = {
    "near_future_2021_2060": (2021, 2060),
    "far_future_2061_2100": (2061, 2100),
}
UNITS = {
    "PRCPTOT": "mm",
    "RX1day": "mm",
    "CDD": "days",
    "TXx": "degC",
    "TNn": "degC",
}


def first_data_var(ds):
    return list(ds.data_vars)[0]


def standardise_xy(da):
    rename = {}
    for cand in ["latitude", "y"]:
        if cand in da.coords or cand in da.dims:
            rename[cand] = "lat"
    for cand in ["longitude", "x"]:
        if cand in da.coords or cand in da.dims:
            rename[cand] = "lon"
    if rename:
        da = da.rename(rename)
    if "lat" in da.coords:
        da = da.sortby("lat")
    if "lon" in da.coords:
        da = da.sortby("lon")
    return da


def write_geotiff(path, da, nodata=-9999.0):
    da = standardise_xy(da.squeeze())
    arr = da.values.astype("float32")
    lat = da["lat"].values
    lon = da["lon"].values
    if lat[0] < lat[-1]:
        arr = arr[::-1, :]
        lat = lat[::-1]
    xres = float(abs(lon[1] - lon[0]))
    yres = float(abs(lat[1] - lat[0]))
    transform = from_origin(float(lon.min()) - xres / 2, float(lat.max()) + yres / 2, xres, yres)
    out = np.where(np.isfinite(arr), arr, nodata).astype("float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=out.shape[0],
        width=out.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
        compress="deflate",
    ) as dst:
        dst.write(out, 1)


def load_change_map(index_name, scenario, period):
    path = ROOT / "output" / "etccdi_spatial_trends" / "maps" / f"etccdi_{index_name}_{scenario}_{period}_ensemble_change.nc"
    with xr.open_dataset(path) as ds:
        var = first_data_var(ds)
        da = standardise_xy(ds[var]).load()
    return da


def load_aoi_mask(target):
    with xr.open_dataset(ROOT / "output" / "zones" / "hydroclimatic_zones_SA.nc") as ds:
        var = "zone" if "zone" in ds.data_vars else first_data_var(ds)
        zones = standardise_xy(ds[var]).load()
    zones = zones.interp(lat=target["lat"], lon=target["lon"], method="nearest")
    return np.isfinite(zones) & (zones >= 0)


def mask_to_aoi(da):
    return da.where(load_aoi_mask(da))


def scenario_emergence_table():
    annual = pd.read_csv(ROOT / "output" / "etccdi_fast" / "tables" / "fast_etccdi_zone_annual_means.csv")
    annual = annual[annual["index"].isin(KEY_INDICES)].copy()
    ens = annual.groupby(["scenario", "index", "year", "zone", "units"], as_index=False)["zone_mean"].mean()

    hist = ens[ens["scenario"] == "historical"]
    noise = (
        hist.groupby(["index", "zone"], as_index=False)["zone_mean"]
        .agg(baseline_mean="mean", baseline_sd="std")
    )

    fut = ens[ens["scenario"].isin(["ssp245", "ssp585"])].copy()
    wide = fut.pivot_table(index=["index", "year", "zone", "units"], columns="scenario", values="zone_mean").reset_index()
    wide = wide.dropna(subset=["ssp245", "ssp585"])
    wide["scenario_difference_585_minus_245"] = wide["ssp585"] - wide["ssp245"]
    wide["scenario_percent_difference"] = 100 * wide["scenario_difference_585_minus_245"] / wide["ssp245"].replace(0, np.nan)
    wide = wide.merge(noise, on=["index", "zone"], how="left")
    wide["signal_to_noise"] = wide["scenario_difference_585_minus_245"].abs() / wide["baseline_sd"].replace(0, np.nan)
    wide.to_csv(TABLE_DIR / "o7_annual_scenario_difference_by_zone.csv", index=False)

    emergence_rows = []
    for (idx, zone), sub in wide.groupby(["index", "zone"]):
        sub = sub.sort_values("year").copy()
        rolling = sub["scenario_difference_585_minus_245"].rolling(20, min_periods=15).mean()
        snr = rolling.abs() / sub["baseline_sd"].replace(0, np.nan)
        emerged = sub.loc[(sub["year"] >= 2021) & (snr >= 1.0)]
        emergence_year = int(emerged["year"].iloc[0]) if not emerged.empty else np.nan
        for period, (start, end) in PERIODS.items():
            psub = sub[(sub["year"] >= start) & (sub["year"] <= end)]
            if psub.empty:
                continue
            emergence_rows.append(
                {
                    "index": idx,
                    "zone": zone,
                    "period": period,
                    "ssp245_mean": psub["ssp245"].mean(),
                    "ssp585_mean": psub["ssp585"].mean(),
                    "difference_585_minus_245": psub["scenario_difference_585_minus_245"].mean(),
                    "percent_difference": psub["scenario_percent_difference"].mean(),
                    "baseline_sd": psub["baseline_sd"].iloc[0],
                    "signal_to_noise": psub["signal_to_noise"].mean(),
                    "emergence_year_snr1_20yr": emergence_year,
                    "units": psub["units"].iloc[0],
                }
            )

    summary = pd.DataFrame(emergence_rows)
    summary.to_csv(TABLE_DIR / "o7_scenario_emergence_summary_by_zone.csv", index=False)
    return wide, summary


def create_spatial_amplification_maps():
    inventory = []
    for idx in KEY_INDICES:
        for period in PERIODS:
            da245 = load_change_map(idx, "ssp245", period)
            da585 = load_change_map(idx, "ssp585", period)
            diff = mask_to_aoi(da585 - da245).rename(f"{idx}_ssp585_minus_ssp245_change")
            diff.attrs.update(
                units=UNITS[idx],
                description=f"{idx} scenario amplification: SSP5-8.5 change minus SSP2-4.5 change for {period}",
            )
            nc_path = MAP_DIR / f"o7_{idx}_{period}_ssp585_minus_ssp245_change.nc"
            tif_path = TIF_DIR / f"o7_{idx}_{period}_ssp585_minus_ssp245_change.tif"
            diff.to_dataset(name=diff.name).to_netcdf(nc_path)
            write_geotiff(tif_path, diff)
            inventory.append(
                {
                    "index": idx,
                    "period": period,
                    "units": UNITS[idx],
                    "mean_amplification": float(diff.mean(skipna=True).values),
                    "min_amplification": float(diff.min(skipna=True).values),
                    "max_amplification": float(diff.max(skipna=True).values),
                    "netcdf": str(nc_path.relative_to(ROOT)),
                    "geotiff": str(tif_path.relative_to(ROOT)),
                }
            )
    inv = pd.DataFrame(inventory)
    inv.to_csv(TABLE_DIR / "o7_spatial_amplification_geotiff_inventory.csv", index=False)
    return inv


def plot_spatial_panel():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), constrained_layout=True)
    axes = axes.ravel()
    letters = list("ABCDE")
    cmaps = {"PRCPTOT": "YlGnBu", "RX1day": "YlGnBu", "CDD": "BrBG_r", "TXx": "YlOrRd", "TNn": "YlOrRd"}
    for i, idx in enumerate(KEY_INDICES):
        ax = axes[i]
        da = mask_to_aoi(load_change_map(idx, "ssp585", "far_future_2061_2100") - load_change_map(idx, "ssp245", "far_future_2061_2100"))
        vals = da.values
        vmax = np.nanpercentile(np.abs(vals), 98)
        if idx in ["CDD"]:
            vmin, vmax_plot = -vmax, vmax
        else:
            vmin, vmax_plot = 0, vmax
        cmap = plt.get_cmap(cmaps[idx]).copy()
        cmap.set_bad("white", 0.0)
        im = ax.pcolormesh(da["lon"], da["lat"], np.ma.masked_invalid(da.values), cmap=cmap, shading="auto", vmin=vmin, vmax=vmax_plot)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.text(
            -0.12,
            1.10,
            letters[i],
            transform=ax.transAxes,
            fontsize=20,
            fontweight="bold",
            ha="left",
            va="bottom",
            clip_on=False,
            bbox=dict(facecolor="white", edgecolor="black", linewidth=0.8, pad=4),
        )
        ax.text(0.5, 1.08, idx, transform=ax.transAxes, ha="center", va="bottom", fontsize=16, fontweight="bold", clip_on=False)
        cb = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.02)
        cb.set_label(UNITS[idx], fontweight="bold")
        for label in cb.ax.get_yticklabels():
            label.set_fontweight("bold")
    axes[-1].axis("off")
    out = FIG_DIR / "MANUSCRIPT_PUBLICATION_o7_far_future_scenario_amplification_panel.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_emergence_heatmap(summary):
    far = summary[summary["period"] == "far_future_2061_2100"].copy()
    far["abs_snr"] = far["signal_to_noise"].abs()
    pivot = far.pivot(index="index", columns="zone", values="signal_to_noise").loc[KEY_INDICES]
    fig, ax = plt.subplots(figsize=(8.6, 4.6), constrained_layout=True)
    im = ax.imshow(pivot.values, cmap="magma", aspect="auto", vmin=0)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, fontweight="bold")
    ax.set_yticks(range(len(pivot.index)), pivot.index, fontweight="bold")
    ax.set_xlabel("Hydroclimatic zone", fontweight="bold")
    ax.set_ylabel("Index", fontweight="bold")
    for y in range(pivot.shape[0]):
        for x in range(pivot.shape[1]):
            val = pivot.values[y, x]
            ax.text(x, y, f"{val:.1f}", ha="center", va="center", color="white" if val > np.nanmax(pivot.values) * 0.55 else "black", fontsize=9, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Scenario separation / baseline SD", fontweight="bold")
    out = FIG_DIR / "SUPPLEMENT_PUBLICATION_o7_scenario_separation_heatmap.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_methods_results(summary, inv):
    far = summary[summary["period"] == "far_future_2061_2100"].copy()
    strongest = far.loc[far["signal_to_noise"].abs().idxmax()]
    earliest = far.dropna(subset=["emergence_year_snr1_20yr"]).sort_values("emergence_year_snr1_20yr").head(5)
    lines = [
        "# O7 Scenario Emergence and Amplification",
        "",
        "This analysis uses the completed index-first ETCCDI annual outputs and existing spatial change maps. It does not require the full daily archive.",
        "",
        "## Methods text for manuscript",
        "Scenario emergence was quantified from annual zone-mean ETCCDI index series by comparing SSP5-8.5 and SSP2-4.5 ensemble means for each year, hydroclimatic zone, and key index. Baseline variability was estimated as the 1985-2014 interannual standard deviation of the historical ensemble-mean series. Scenario separation was expressed as the absolute SSP5-8.5 minus SSP2-4.5 difference divided by baseline standard deviation. Emergence year was defined as the first year after 2020 when the 20-year rolling scenario difference exceeded one baseline standard deviation. Far-future spatial amplification was mapped as the SSP5-8.5 change minus the SSP2-4.5 change for 2061-2100.",
        "",
        "## Results text for manuscript",
        f"The strongest far-future scenario separation occurred for {strongest['index']} in {strongest['zone']} (signal-to-noise ratio = {strongest['signal_to_noise']:.2f}). Scenario amplification maps show where SSP5-8.5 adds the largest change beyond SSP2-4.5, with GeoTIFFs exported for QGIS.",
        "",
        "Earliest emergence examples:",
    ]
    for _, r in earliest.iterrows():
        lines.append(f"- {r['index']} {r['zone']}: {int(r['emergence_year_snr1_20yr'])}")
    lines.extend(
        [
            "",
            "## Output files",
            "- tables/o7_annual_scenario_difference_by_zone.csv",
            "- tables/o7_scenario_emergence_summary_by_zone.csv",
            "- tables/o7_spatial_amplification_geotiff_inventory.csv",
            "- figures/MANUSCRIPT_PUBLICATION_o7_far_future_scenario_amplification_panel.png",
            "- figures/SUPPLEMENT_PUBLICATION_o7_scenario_separation_heatmap.png",
            "- geotiff/o7_*_ssp585_minus_ssp245_change.tif",
        ]
    )
    (OUT / "O7_SCENARIO_EMERGENCE_METHODS_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    annual_diff, summary = scenario_emergence_table()
    inv = create_spatial_amplification_maps()
    plot_spatial_panel()
    plot_emergence_heatmap(summary)
    write_methods_results(summary, inv)
    print("O7 outputs written:", OUT)
    print("Rows:", len(summary), "GeoTIFFs:", len(list(TIF_DIR.glob("*.tif"))))


if __name__ == "__main__":
    main()
