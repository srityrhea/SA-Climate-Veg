from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]

ZONE_COLORS = ["#8c510a", "#d98c24", "#f6c65b", "#b8de6f", "#2a9d8f", "#5ab4ac", "#084c4f"]
SCENARIO_COLORS = {"ssp245": "#2c7fb8", "ssp585": "#d95f0e", "historical": "#4d4d4d"}
PERIOD_COLORS = {"near_future": "#4C78A8", "mid_future": "#F58518", "far_future": "#54A24B"}
INDEX_UNITS = {"PRCPTOT": "mm", "RX1day": "mm", "CDD": "days", "TXx": "degC", "TNn": "degC"}
KEY_INDICES = ["PRCPTOT", "RX1day", "CDD", "TXx", "TNn"]
_ZONE_CACHE = None


def set_style():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "font.weight": "bold",
            "axes.labelsize": 15,
            "axes.labelweight": "bold",
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "legend.title_fontsize": 11,
            "axes.linewidth": 1.1,
            "savefig.dpi": 350,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


def first_var(ds):
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


def load_zones():
    global _ZONE_CACHE
    if _ZONE_CACHE is not None:
        return _ZONE_CACHE
    with xr.open_dataset(ROOT / "output/zones/hydroclimatic_zones_SA.nc") as ds:
        _ZONE_CACHE = standardise_xy(ds["zone"]).load()
        return _ZONE_CACHE


def aoi_mask(target):
    z = load_zones().interp(lat=target["lat"], lon=target["lon"], method="nearest")
    return z >= 0


def masked(da):
    da = standardise_xy(da)
    return da.where(aoi_mask(da))


def add_panel_label(ax, letter, x=-0.13, y=1.10):
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
        bbox=dict(facecolor="white", edgecolor="black", linewidth=1.0, pad=4),
    )


def bold_ticks(ax):
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")


def save(fig, path):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=350, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def pcolormesh_da(ax, da, cmap, vmin=None, vmax=None, norm=None):
    cmap_obj = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap
    cmap_obj.set_bad("white", 0.0)
    arr = np.ma.masked_invalid(da.values)
    if norm is not None:
        im = ax.pcolormesh(da["lon"], da["lat"], arr, cmap=cmap_obj, shading="auto", norm=norm)
    else:
        im = ax.pcolormesh(da["lon"], da["lat"], arr, cmap=cmap_obj, shading="auto", vmin=vmin, vmax=vmax)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    bold_ticks(ax)
    return im


def read_tif_as_da(path, name):
    path = ROOT / path
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        with xr.open_dataset(ROOT / "output/zones/hydroclimatic_zones_SA.nc") as zds:
            raw_lat = zds["lat"].values
            raw_lon = zds["lon"].values
        if arr.shape == (len(raw_lat), len(raw_lon)):
            ys = raw_lat
            xs = raw_lon
        else:
            transform = src.transform
            xs = transform.c + (np.arange(src.width) + 0.5) * transform.a
            ys = transform.f + (np.arange(src.height) + 0.5) * transform.e
    return xr.DataArray(arr, coords={"lat": ys, "lon": xs}, dims=("lat", "lon"), name=name).sortby("lat")


def figure1_zones():
    zones = load_zones()
    pann = read_tif_as_da("output/zones/tif/zones_p_ann.tif", "Pann")
    agree = read_tif_as_da("output/zones/tif/zones_agreement.tif", "agreement")
    ai = read_tif_as_da("output/zones/tif/zones_aridity_index.tif", "AI")
    ai = ai.where(ai < 10)
    layers = [
        ("A", "Hydroclimatic zones", zones.where(zones >= 0), ListedColormap(ZONE_COLORS), None, None, "zone"),
        ("B", "Algorithm agreement", masked(agree), "RdYlGn", 0, 1, "agreement"),
        ("C", "Mean annual precipitation", masked(pann), "YlGnBu", 0, float(np.nanpercentile(pann, 98)), "mm yr-1"),
        ("D", "Aridity index", masked(ai), "YlOrRd_r", 0, float(np.nanpercentile(ai, 98)), "P/PET"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.2), constrained_layout=True)
    for ax, (letter, title, da, cmap, vmin, vmax, label) in zip(axes.ravel(), layers):
        if letter == "A":
            norm = BoundaryNorm(np.arange(-0.5, 7.5, 1), 7)
            im = pcolormesh_da(ax, da, cmap, norm=norm)
            handles = [Patch(facecolor=ZONE_COLORS[i], edgecolor="black", label=f"Z{i+1}") for i in range(7)]
            ax.legend(handles=handles, loc="lower left", ncol=1, frameon=True, facecolor="white", edgecolor="black")
        else:
            im = pcolormesh_da(ax, da, cmap, vmin=vmin, vmax=vmax)
            cb = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.02)
            cb.set_label(label, fontweight="bold")
            for tick in cb.ax.get_yticklabels():
                tick.set_fontweight("bold")
        add_panel_label(ax, letter)
        ax.text(0.5, 1.07, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=16, fontweight="bold")
    save(fig, "output/zones/hydroclimatic_zones_SA.png")


def figure2_ranking():
    df = pd.read_csv(ROOT / "output/cmip6_eval/tables/model_ranking.csv").sort_values("composite_score", ascending=True)
    colors = ["#2c7fb8" if v else "#bdbdbd" for v in df["in_ensemble_pool"]]
    fig, ax = plt.subplots(figsize=(9.5, 6.2), constrained_layout=True)
    ax.barh(df["model"], df["composite_score"], color=colors, edgecolor="black", linewidth=0.5)
    med = df["composite_score"].median()
    ax.axvline(med, color="black", linestyle="--", linewidth=1.2)
    ax.text(med + 0.004, len(df) - 0.3, "median", fontsize=11, fontweight="bold", va="top")
    ax.set_xlabel("Composite skill score")
    ax.set_ylabel("CMIP6 model")
    bold_ticks(ax)
    ax.grid(axis="x", alpha=0.25)
    save(fig, "output/cmip6_eval/figures/model_ranking.png")


def figure3_projection_bars(variable, out_path):
    df = pd.read_csv(ROOT / "output/o3a_skill_weighted_ensemble/tables/o3a_period_changes_by_zone.csv")
    sub = df[df["variable"] == variable].copy()
    value_col = "percent_change_pr_only" if variable == "pr" else "absolute_change"
    ylabel = "Precipitation change (%)" if variable == "pr" else "Temperature change (degC)"
    title = {"pr": "PR", "tasmax": "Tmax", "tasmin": "Tmin"}[variable]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True, constrained_layout=True)
    periods = ["near_future", "mid_future", "far_future"]
    period_labels = ["Near", "Mid", "Far"]
    x = np.arange(7)
    width = 0.24
    for ax, scenario, letter in zip(axes, ["ssp245", "ssp585"], ["A", "B"]):
        s = sub[sub["scenario"] == scenario]
        for j, period in enumerate(periods):
            vals = (
                s[s["period"] == period]
                .set_index("zone")
                .reindex([f"Z{i}" for i in range(1, 8)])[value_col]
                .astype(float)
                .values
            )
            ax.bar(x + (j - 1) * width, vals, width=width, label=period_labels[j], color=list(PERIOD_COLORS.values())[j], edgecolor="black", linewidth=0.4)
        add_panel_label(ax, letter, x=-0.10, y=1.05)
        ax.text(0.5, 1.05, scenario.upper().replace("SSP", "SSP"), transform=ax.transAxes, ha="center", va="bottom", fontsize=15, fontweight="bold")
        ax.set_xticks(x, [f"Z{i}" for i in range(1, 8)])
        ax.set_xlabel("Hydroclimatic zone")
        ax.grid(axis="y", alpha=0.25)
        bold_ticks(ax)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(title="Period", frameon=True, edgecolor="black", loc="upper left")
    fig.suptitle(f"Zone-level projected changes: {title}", fontsize=17, fontweight="bold")
    save(fig, out_path)


def figure4_etccdi_change():
    fig, axes = plt.subplots(2, 3, figsize=(14.6, 8.4), constrained_layout=True)
    axes = axes.ravel()
    cmaps = {"PRCPTOT": "BrBG", "RX1day": "BrBG", "CDD": "BrBG_r", "TXx": "RdYlBu_r", "TNn": "RdYlBu_r"}
    labels = {"PRCPTOT": "PRCPTOT", "RX1day": "RX1day", "CDD": "CDD", "TXx": "TXx", "TNn": "TNn"}
    for i, idx in enumerate(KEY_INDICES):
        with xr.open_dataset(ROOT / f"output/etccdi_spatial_trends/maps/etccdi_{idx}_ssp585_far_future_2061_2100_ensemble_change.nc") as ds:
            da = standardise_xy(ds["ensemble_mean_change"]).load()
        da = masked(da)
        vals = da.values
        if idx in ["TXx", "TNn"]:
            vmin, vmax, norm = 0, float(np.nanpercentile(vals, 98)), None
        else:
            lim = float(np.nanpercentile(np.abs(vals), 98))
            vmin, vmax, norm = -lim, lim, TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
        im = pcolormesh_da(axes[i], da, cmaps[idx], vmin=vmin, vmax=vmax, norm=norm)
        add_panel_label(axes[i], chr(65 + i))
        axes[i].text(0.5, 1.08, labels[idx], transform=axes[i].transAxes, ha="center", va="bottom", fontsize=16, fontweight="bold")
        cb = fig.colorbar(im, ax=axes[i], shrink=0.86, pad=0.02)
        cb.set_label(INDEX_UNITS[idx], fontweight="bold")
        for tick in cb.ax.get_yticklabels():
            tick.set_fontweight("bold")
    axes[-1].axis("off")
    save(fig, "output/etccdi_spatial_trends/figures/MANUSCRIPT_Figure_ETCCDI_spatial_change_SSP585_far_future.png")


def heatmap_from_metrics(metric, out_path, label):
    df = pd.read_csv(ROOT / "output/cmip6_eval/tables/evaluation_metrics_by_model_variable_zone.csv")
    ranking = pd.read_csv(ROOT / "output/cmip6_eval/tables/model_ranking.csv")
    order = ranking.sort_values("composite_score", ascending=False)["model"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.4), sharey=True, constrained_layout=True)
    for ax, var, letter in zip(axes, ["pr", "tasmax", "tasmin"], ["A", "B", "C"]):
        sub = df[df["variable"] == var].pivot_table(index="model", columns="zone", values=metric).reindex(order)
        arr = sub.values
        cmap = "RdYlBu_r" if metric in ["rmse", "bias"] else "viridis"
        im = ax.imshow(arr, aspect="auto", cmap=cmap)
        ax.set_xticks(range(sub.shape[1]), sub.columns, rotation=0, fontweight="bold")
        ax.set_yticks(range(sub.shape[0]), sub.index, fontweight="bold")
        ax.set_xlabel("Zone")
        if ax is axes[0]:
            ax.set_ylabel("Model")
        add_panel_label(ax, letter, x=-0.18, y=1.04)
        ax.text(0.5, 1.04, var.upper(), transform=ax.transAxes, ha="center", va="bottom", fontsize=15, fontweight="bold")
        cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
        cb.set_label(label, fontweight="bold")
        for tick in cb.ax.get_yticklabels():
            tick.set_fontweight("bold")
    save(fig, out_path)


def supplementary_o1_validation():
    df = pd.read_csv(ROOT / "output/zones/tables/o1_kmeans_k_selection_validation_metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), constrained_layout=True)
    cols = [("silhouette_score_sample", "Silhouette"), ("davies_bouldin_index_sample", "Davies-Bouldin"), ("calinski_harabasz_score_sample", "Calinski-Harabasz")]
    for ax, (col, lab), letter in zip(axes, cols, ["A", "B", "C"]):
        ax.plot(df["k"], df[col], color="#2c7fb8", linewidth=2.4, marker="o", markersize=6)
        ax.set_xlabel("Number of zones (k)")
        ax.set_ylabel(lab)
        add_panel_label(ax, letter, x=-0.16, y=1.05)
        ax.grid(alpha=0.25)
        bold_ticks(ax)
    save(fig, "output/zones/figures/o1_zone_validation_metrics_panel.png")


def o3a_weights_and_series():
    weights = pd.read_csv(ROOT / "output/o3a_skill_weighted_ensemble/tables/o3a_model_pools_and_weights.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0), sharey=True, constrained_layout=True)
    for ax, var, letter in zip(axes, ["pr", "tasmax", "tasmin"], ["A", "B", "C"]):
        sub = weights[weights["variable"] == var].sort_values("weight")
        ax.barh(sub["model"], sub["weight"], color="#2c7fb8", edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Normalized weight")
        add_panel_label(ax, letter, x=-0.18, y=1.05)
        ax.text(0.5, 1.05, var.upper(), transform=ax.transAxes, ha="center", va="bottom", fontsize=15, fontweight="bold")
        ax.grid(axis="x", alpha=0.25)
        bold_ticks(ax)
    save(fig, "output/o3a_skill_weighted_ensemble/figures/o3a_model_weights_by_variable.png")

    ts = pd.read_csv(ROOT / "output/o3a_skill_weighted_ensemble/tables/o3a_zone_annual_ensemble_timeseries.csv")
    for var, out, ylabel in [
        ("pr", "output/o3a_skill_weighted_ensemble/figures/o3a_annual_series_pr.png", "Precipitation (mm yr-1)"),
        ("tasmax", "output/o3a_skill_weighted_ensemble/figures/o3a_annual_series_tasmax.png", "Tmax (degC)"),
        ("tasmin", "output/o3a_skill_weighted_ensemble/figures/o3a_annual_series_tasmin.png", "Tmin (degC)"),
    ]:
        sub = ts[(ts["variable"] == var) & (ts["stat"] == "mean")]
        dom = sub.groupby(["scenario", "year"], as_index=False)["value"].mean()
        fig, ax = plt.subplots(figsize=(10.5, 5.3), constrained_layout=True)
        for scenario, g in dom.groupby("scenario"):
            ax.plot(g["year"], g["value"], lw=2.3, label=scenario.upper(), color=SCENARIO_COLORS.get(scenario, "black"))
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=True, edgecolor="black")
        ax.grid(alpha=0.25)
        bold_ticks(ax)
        save(fig, out)


def five_map_panel_from_tifs(tif_paths, out_path, units, cmaps, titles=None, divergent=None):
    titles = titles or list(tif_paths.keys())
    divergent = divergent or set()
    fig, axes = plt.subplots(2, 3, figsize=(14.6, 8.4), constrained_layout=True)
    axes = axes.ravel()
    for i, (idx, tif) in enumerate(tif_paths.items()):
        da = masked(read_tif_as_da(tif, idx))
        vals = da.values
        if idx in divergent:
            lim = float(np.nanpercentile(np.abs(vals), 98))
            norm = TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
            vmin = vmax = None
        else:
            vmin = 0 if np.nanmin(vals) >= 0 else float(np.nanpercentile(vals, 2))
            vmax = float(np.nanpercentile(vals, 98))
            norm = None
        im = pcolormesh_da(axes[i], da, cmaps[idx], vmin=vmin, vmax=vmax, norm=norm)
        add_panel_label(axes[i], chr(65 + i))
        axes[i].text(0.5, 1.08, idx, transform=axes[i].transAxes, ha="center", va="bottom", fontsize=16, fontweight="bold")
        cb = fig.colorbar(im, ax=axes[i], shrink=0.86, pad=0.02)
        cb.set_label(units[idx], fontweight="bold")
        for tick in cb.ax.get_yticklabels():
            tick.set_fontweight("bold")
    axes[-1].axis("off")
    save(fig, out_path)


def restyle_o5_o6_main_maps():
    o5_tifs = {
        "PRCPTOT": "output/o5_pettitt_prewhitened_trends/geotiff/o5_ext_ssp585_PRCPTOT_sen_slope_decade.tif",
        "RX1day": "output/o5_pettitt_prewhitened_trends/geotiff/o5_ext_ssp585_RX1day_sen_slope_decade.tif",
        "CDD": "output/o5_pettitt_prewhitened_trends/geotiff/o5_ext_ssp585_CDD_sen_slope_decade.tif",
        "TXx": "output/o5_pettitt_prewhitened_trends/geotiff/o5_ext_ssp585_TXx_sen_slope_decade.tif",
        "TNn": "output/o5_pettitt_prewhitened_trends/geotiff/o5_ext_ssp585_TNn_sen_slope_decade.tif",
    }
    five_map_panel_from_tifs(
        o5_tifs,
        "output/o5_pettitt_prewhitened_trends/figures/MANUSCRIPT_PUBLICATION_o5_ssp585_sen_slope_panel.png",
        {"PRCPTOT": "mm decade-1", "RX1day": "mm decade-1", "CDD": "days decade-1", "TXx": "degC decade-1", "TNn": "degC decade-1"},
        {"PRCPTOT": "YlGnBu", "RX1day": "YlGnBu", "CDD": "BrBG_r", "TXx": "YlOrRd", "TNn": "YlOrRd"},
        divergent={"CDD"},
    )

    trend_tifs = {
        "NDVI": "output/o6_vegetation_climate_refinement/geotiff/o6_NDVI_sen_slope_decade.tif",
        "EVI": "output/o6_vegetation_climate_refinement/geotiff/o6_EVI_sen_slope_decade.tif",
        "GPP": "output/o6_vegetation_climate_refinement/geotiff/o6_GPP_sen_slope_decade.tif",
        "NPP": "output/o6_vegetation_climate_refinement/geotiff/o6_NPP_sen_slope_decade.tif",
    }
    four_map_panel(
        trend_tifs,
        "output/o6_vegetation_climate_refinement/figures/MANUSCRIPT_PUBLICATION_o6_vegetation_trend_panel.png",
        "Sen slope decade-1",
        "YlGn",
        divergent=False,
    )

    link_tifs = {
        "NDVI": "output/o6_vegetation_climate_refinement/geotiff/o6_NDVI_PRCPTOT_lag0_spearman_r.tif",
        "EVI": "output/o6_vegetation_climate_refinement/geotiff/o6_EVI_PRCPTOT_lag0_spearman_r.tif",
        "GPP": "output/o6_vegetation_climate_refinement/geotiff/o6_GPP_PRCPTOT_lag0_spearman_r.tif",
        "NPP": "output/o6_vegetation_climate_refinement/geotiff/o6_NPP_PRCPTOT_lag0_spearman_r.tif",
    }
    four_map_panel(
        link_tifs,
        "output/o6_vegetation_climate_refinement/figures/MANUSCRIPT_PUBLICATION_o6_prcptot_vegetation_linkage_panel.png",
        "Spearman r",
        "RdBu",
        divergent=True,
    )


def four_map_panel(tif_paths, out_path, cbar_label, cmap, divergent=False):
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.8), constrained_layout=True)
    axes = axes.ravel()
    for i, (title, tif) in enumerate(tif_paths.items()):
        da = masked(read_tif_as_da(tif, title))
        vals = da.values
        if divergent:
            norm = TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
            vmin = vmax = None
        else:
            vmin = 0 if np.nanmin(vals) >= 0 else float(np.nanpercentile(vals, 2))
            vmax = float(np.nanpercentile(vals, 98))
            norm = None
        im = pcolormesh_da(axes[i], da, cmap, vmin=vmin, vmax=vmax, norm=norm)
        add_panel_label(axes[i], chr(65 + i))
        axes[i].text(0.5, 1.08, title, transform=axes[i].transAxes, ha="center", va="bottom", fontsize=16, fontweight="bold")
        cb = fig.colorbar(im, ax=axes[i], shrink=0.86, pad=0.02)
        cb.set_label(cbar_label, fontweight="bold")
        for tick in cb.ax.get_yticklabels():
            tick.set_fontweight("bold")
    save(fig, out_path)


def supplementary_s2_algorithm_comparison():
    saved = pd.read_csv(ROOT / "output/zones/tables/o1_multi_algorithm_7zone_validation_metrics.csv")
    metrics = ["silhouette_score", "davies_bouldin_index", "calinski_harabasz_score"]
    labels = ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.6), constrained_layout=True)
    for ax, metric, label, letter in zip(axes, metrics, labels, ["A", "B", "C"]):
        ax.bar(saved["algorithm"], saved[metric], color="#2c7fb8", edgecolor="black", linewidth=0.5)
        ax.set_ylabel(label)
        ax.set_xlabel("Partition")
        ax.tick_params(axis="x", rotation=25)
        add_panel_label(ax, letter, x=-0.16, y=1.05)
        ax.grid(axis="y", alpha=0.25)
        bold_ticks(ax)
    save(fig, "output/zones/figures/o1_multi_algorithm_validation_comparison.png")


def supplementary_etccdi_spatial(period, out_path):
    period_label = "near_future_2021_2060" if period == "near" else "far_future_2061_2100"
    fig, axes = plt.subplots(2, 5, figsize=(18.2, 7.5), constrained_layout=True)
    cmaps = {"PRCPTOT": "BrBG", "RX1day": "BrBG", "CDD": "BrBG_r", "TXx": "RdYlBu_r", "TNn": "RdYlBu_r"}
    for row, scenario in enumerate(["ssp245", "ssp585"]):
        for col, idx in enumerate(KEY_INDICES):
            ax = axes[row, col]
            with xr.open_dataset(ROOT / f"output/etccdi_spatial_trends/maps/etccdi_{idx}_{scenario}_{period_label}_ensemble_change.nc") as ds:
                da = masked(standardise_xy(ds["ensemble_mean_change"]).load())
            vals = da.values
            if idx in ["TXx", "TNn"]:
                vmin, vmax, norm = 0, float(np.nanpercentile(vals, 98)), None
            else:
                lim = float(np.nanpercentile(np.abs(vals), 98))
                vmin, vmax, norm = None, None, TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
            im = pcolormesh_da(ax, da, cmaps[idx], vmin=vmin, vmax=vmax, norm=norm)
            add_panel_label(ax, chr(65 + row * 5 + col), x=-0.14, y=1.08)
            ax.text(0.5, 1.08, f"{scenario.upper()} {idx}", transform=ax.transAxes, ha="center", va="bottom", fontsize=14, fontweight="bold")
            cb = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
            cb.set_label(INDEX_UNITS[idx], fontweight="bold")
            for tick in cb.ax.get_yticklabels():
                tick.set_fontweight("bold")
    save(fig, out_path)


def supplementary_o5_maps_and_heatmaps():
    plot_o5_pettitt_year_panel()

    df = pd.read_csv(ROOT / "output/o5_pettitt_prewhitened_trends/tables/o5_ext_key_indices_pettitt_prewhitened_mk_by_zone.csv")
    df = df[(df["scenario"] == "ssp585") & (df["index"].isin(KEY_INDICES))].copy()
    for value_col, out, cbar in [
        ("sen_slope_per_decade", "output/o5_pettitt_prewhitened_trends/figures/o5_ext_ssp585_sen_slope_heatmap.png", "Sen slope decade-1"),
        ("pettitt_change_year", "output/o5_pettitt_prewhitened_trends/figures/o5_ext_ssp585_pettitt_year_heatmap.png", "Pettitt year"),
    ]:
        mat = df.pivot(index="index", columns="zone", values=value_col).reindex(KEY_INDICES)
        fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
        if value_col == "sen_slope_per_decade":
            lim = np.nanpercentile(np.abs(mat.values), 98)
            im = ax.imshow(mat.values, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim), aspect="auto")
        else:
            im = ax.imshow(mat.values, cmap="viridis", aspect="auto")
        ax.set_xticks(range(mat.shape[1]), mat.columns, fontweight="bold")
        ax.set_yticks(range(mat.shape[0]), mat.index, fontweight="bold")
        ax.set_xlabel("Hydroclimatic zone")
        ax.set_ylabel("Index")
        for y in range(mat.shape[0]):
            for x in range(mat.shape[1]):
                val = mat.values[y, x]
                if np.isfinite(val):
                    txt = f"{val:.1f}" if value_col == "sen_slope_per_decade" else f"{int(val)}"
                    ax.text(x, y, txt, ha="center", va="center", fontsize=9, fontweight="bold")
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label(cbar, fontweight="bold")
        save(fig, out)


def plot_o5_pettitt_year_panel():
    fig, axes = plt.subplots(2, 3, figsize=(14.6, 8.4), constrained_layout=True)
    axes = axes.ravel()
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white", 0.0)
    for i, idx in enumerate(KEY_INDICES):
        year = read_tif_as_da(
            f"output/o5_pettitt_prewhitened_trends/geotiff/o5_ext_ssp585_{idx}_pettitt_year.tif",
            idx,
        )
        sig_path = ROOT / f"output/o5_pettitt_prewhitened_trends/geotiff/o5_ext_ssp585_{idx}_pettitt_sig.tif"
        if sig_path.exists():
            sig = read_tif_as_da(sig_path.relative_to(ROOT), f"{idx}_sig")
            year = year.where(sig > 0)
        year = masked(year.where(year >= 1900))
        im = pcolormesh_da(axes[i], year, cmap, vmin=2040, vmax=2095)
        add_panel_label(axes[i], chr(65 + i))
        axes[i].text(0.5, 1.08, idx, transform=axes[i].transAxes, ha="center", va="bottom", fontsize=16, fontweight="bold")
        cb = fig.colorbar(im, ax=axes[i], shrink=0.86, pad=0.02)
        cb.set_label("year", fontweight="bold")
        cb.set_ticks([2040, 2050, 2060, 2070, 2080, 2090])
        for tick in cb.ax.get_yticklabels():
            tick.set_fontweight("bold")
    axes[-1].axis("off")
    save(fig, "output/o5_pettitt_prewhitened_trends/figures/SUPPLEMENT_PUBLICATION_o5_ssp585_pettitt_year_panel.png")


def supplementary_vegetation_heatmaps():
    trend = pd.read_csv(ROOT / "output/vegetation_climate_linkage/tables/modis_vegetation_zone_trends.csv")
    if "sen_slope_per_decade" in trend.columns:
        mat = trend.pivot(index="vegetation_metric", columns="zone", values="sen_slope_per_decade")
        fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
        lim = np.nanpercentile(np.abs(mat.values), 98)
        im = ax.imshow(mat.values, cmap="BrBG", norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim), aspect="auto")
        ax.set_xticks(range(mat.shape[1]), mat.columns, fontweight="bold")
        ax.set_yticks(range(mat.shape[0]), mat.index, fontweight="bold")
        ax.set_xlabel("Hydroclimatic zone")
        ax.set_ylabel("Vegetation metric")
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label("Sen slope decade-1", fontweight="bold")
        save(fig, "output/vegetation_climate_linkage/figures/modis_vegetation_zone_trend_heatmap.png")

    corr = pd.read_csv(ROOT / "output/vegetation_climate_linkage/tables/vegetation_key_etccdi_correlations_by_zone.csv")
    for metric in ["NDVI", "EVI", "GPP", "NPP"]:
        sub = corr[(corr["vegetation_metric"] == metric) & (corr["lag_years"] == 0)].copy()
        if sub.empty:
            continue
        mat = sub.pivot_table(index="zone", columns="climate_index", values="spearman_r", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
        im = ax.imshow(mat.values, cmap="RdBu", norm=TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1), aspect="auto")
        ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=35, ha="right", fontweight="bold")
        ax.set_yticks(range(mat.shape[0]), mat.index, fontweight="bold")
        ax.set_xlabel("Climate index")
        ax.set_ylabel("Hydroclimatic zone")
        for y in range(mat.shape[0]):
            for x in range(mat.shape[1]):
                val = mat.values[y, x]
                if np.isfinite(val):
                    ax.text(x, y, f"{val:.2f}", ha="center", va="center", fontsize=8.5, fontweight="bold")
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label("Spearman r", fontweight="bold")
        save(fig, f"output/vegetation_climate_linkage/figures/vegetation_climate_correlation_heatmap_{metric}.png")


def supplementary_etccdi_timeseries_and_trends():
    annual = pd.read_csv(ROOT / "output/etccdi_fast/tables/fast_etccdi_zone_annual_means.csv")
    sub = annual[annual["index"].isin(KEY_INDICES)].copy()
    ens = sub.groupby(["scenario", "index", "year"], as_index=False)["zone_mean"].mean()
    fig, axes = plt.subplots(3, 2, figsize=(12.8, 10.2), constrained_layout=True)
    axes = axes.ravel()
    for i, idx in enumerate(KEY_INDICES):
        ax = axes[i]
        for scenario, g in ens[ens["index"] == idx].groupby("scenario"):
            g = g.sort_values("year")
            ax.plot(g["year"], g["zone_mean"], lw=2.1, label=scenario.upper(), color=SCENARIO_COLORS.get(scenario, None))
        add_panel_label(ax, chr(65 + i), x=-0.15, y=1.05)
        ax.text(0.5, 1.05, idx, transform=ax.transAxes, ha="center", va="bottom", fontsize=15, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel(INDEX_UNITS[idx])
        ax.grid(alpha=0.25)
        bold_ticks(ax)
    axes[-1].axis("off")
    axes[0].legend(frameon=True, edgecolor="black", loc="upper left")
    save(fig, "output/etccdi_fast/figures/fast_etccdi_main_indices_panel.png")

    trends = pd.read_csv(ROOT / "output/etccdi_spatial_trends/tables/etccdi_key_indices_ensemble_zone_trends.csv")
    trends = trends[(trends["series_type"] == "ensemble_mean") & (trends["index"].isin(KEY_INDICES))]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)
    for ax, scenario, letter in zip(axes, ["historical", "ssp245", "ssp585"], ["A", "B", "C"]):
        mat = trends[trends["scenario"] == scenario].pivot(index="index", columns="zone", values="sen_slope_per_decade").reindex(KEY_INDICES)
        lim = np.nanpercentile(np.abs(mat.values), 98)
        im = ax.imshow(mat.values, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim), aspect="auto")
        ax.set_xticks(range(mat.shape[1]), mat.columns, fontweight="bold")
        ax.set_yticks(range(mat.shape[0]), mat.index, fontweight="bold")
        ax.set_xlabel("Hydroclimatic zone")
        if ax is axes[0]:
            ax.set_ylabel("Index")
        add_panel_label(ax, letter, x=-0.18, y=1.05)
        ax.text(0.5, 1.05, scenario.upper(), transform=ax.transAxes, ha="center", va="bottom", fontsize=15, fontweight="bold")
        for y in range(mat.shape[0]):
            for x in range(mat.shape[1]):
                val = mat.values[y, x]
                if np.isfinite(val):
                    ax.text(x, y, f"{val:.1f}", ha="center", va="center", fontsize=8.3, fontweight="bold")
        cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
        cb.set_label("Sen slope decade-1", fontweight="bold")
    save(fig, "output/etccdi_spatial_trends/figures/SUPPLEMENT_Figure_ETCCDI_zone_trends_all_scenarios.png")


def main():
    set_style()
    figure1_zones()
    figure2_ranking()
    figure3_projection_bars("pr", "output/o3a_skill_weighted_ensemble/figures/o3a_zone_period_changes_pr.png")
    figure3_projection_bars("tasmax", "output/o3a_skill_weighted_ensemble/figures/o3a_zone_period_changes_tasmax.png")
    figure3_projection_bars("tasmin", "output/o3a_skill_weighted_ensemble/figures/o3a_zone_period_changes_tasmin.png")
    figure4_etccdi_change()
    supplementary_o1_validation()
    supplementary_s2_algorithm_comparison()
    heatmap_from_metrics("kge", "output/cmip6_eval/figures/kge_heatmaps.png", "KGE")
    heatmap_from_metrics("rmse", "output/cmip6_eval/figures/rmse_heatmaps.png", "RMSE")
    heatmap_from_metrics("bias", "output/cmip6_eval/figures/bias_heatmaps.png", "Bias")
    heatmap_from_metrics("r", "output/cmip6_eval/figures/r_heatmaps.png", "Correlation")
    o3a_weights_and_series()
    restyle_o5_o6_main_maps()
    supplementary_etccdi_spatial("near", "output/etccdi_spatial_trends/figures/SUPPLEMENT_Figure_ETCCDI_spatial_change_near_future_2021_2060.png")
    supplementary_etccdi_spatial("far", "output/etccdi_spatial_trends/figures/SUPPLEMENT_Figure_ETCCDI_spatial_change_far_future_2061_2100.png")
    supplementary_o5_maps_and_heatmaps()
    supplementary_vegetation_heatmaps()
    supplementary_etccdi_timeseries_and_trends()
    print("Publication figure restyle completed.")


if __name__ == "__main__":
    main()
