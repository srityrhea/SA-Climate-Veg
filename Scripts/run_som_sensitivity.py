from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import rasterio
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "o1_som_sensitivity"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
TIF_DIR = OUT / "geotiff"
for d in [TABLE_DIR, FIG_DIR, TIF_DIR]:
    d.mkdir(parents=True, exist_ok=True)

YEARS = list(range(1985, 2015))
ZONE_NC = ROOT / "output" / "zones" / "hydroclimatic_zones_SA.nc"
ZONE_TIF = ROOT / "output" / "zones" / "tif" / "zones_consensus.tif"
FEATURE_NPZ = TABLE_DIR / "o1_som_reconstructed_feature_arrays.npz"
FEATURE_CSV = TABLE_DIR / "o1_som_feature_matrix_sample.csv"


def load_zone():
    ds = xr.open_dataset(ZONE_NC)
    zone = ds["zone"].values.astype("int16")
    lat = ds["lat"].values
    lon = ds["lon"].values
    ds.close()
    return zone, lat, lon


def reconstruct_features():
    if FEATURE_NPZ.exists():
        z = np.load(FEATURE_NPZ)
        return {k: z[k] for k in z.files}

    zone, lat, lon = load_zone()
    shape = zone.shape
    annual_pr_sum = np.zeros(shape, dtype="float64")
    monsoon_pr_sum = np.zeros(shape, dtype="float64")
    monthly_totals = np.zeros((12,) + shape, dtype="float64")
    dtr_sum = np.zeros(shape, dtype="float64")
    dtr_count = 0

    for year in YEARS:
        pr_path = ROOT / "output" / "era5" / "pr" / f"era5_land_pr_{year}_clipped.nc"
        tx_path = ROOT / "output" / "era5" / "tasmax" / f"era5_land_tasmax_{year}_clipped.nc"
        tn_path = ROOT / "output" / "era5" / "tasmin" / f"era5_land_tasmin_{year}_clipped.nc"

        with xr.open_dataset(pr_path) as ds:
            pr = ds["pr"]
            annual_pr_sum += pr.sum("time", skipna=True).values
            monsoon_pr_sum += pr.sel(time=pr["time"].dt.month.isin([6, 7, 8, 9])).sum("time", skipna=True).values
            monthly = pr.groupby("time.month").sum("time", skipna=True)
            for month in monthly["month"].values:
                monthly_totals[int(month) - 1] += monthly.sel(month=month).values

        with xr.open_dataset(tx_path) as txds, xr.open_dataset(tn_path) as tnds:
            dtr = txds["tasmax"] - tnds["tasmin"]
            dtr_sum += dtr.sum("time", skipna=True).values
            dtr_count += dtr.sizes["time"]

    p_ann = annual_pr_sum / len(YEARS)
    monsoon_frac = np.divide(monsoon_pr_sum, annual_pr_sum, out=np.full(shape, np.nan), where=annual_pr_sum != 0)
    monthly_clim = monthly_totals / len(YEARS)
    cv_monthly = np.divide(
        np.nanstd(monthly_clim, axis=0),
        np.nanmean(monthly_clim, axis=0),
        out=np.full(shape, np.nan),
        where=np.nanmean(monthly_clim, axis=0) != 0,
    )
    dtr_mean = dtr_sum / dtr_count

    with xr.open_dataset(ROOT / "output" / "ancillary" / "srtm" / "srtm_elevation_1km_clipped.nc") as elev_ds:
        elev = elev_ds["elevation"].interp(lat=lat, lon=lon, method="linear").values

    lat_grid = np.repeat(lat[:, None], len(lon), axis=1)

    arrays = {
        "zone": zone,
        "lat": lat,
        "lon": lon,
        "p_ann": p_ann.astype("float32"),
        "cv_monthly": cv_monthly.astype("float32"),
        "monsoon_frac": monsoon_frac.astype("float32"),
        "dtr_mean": dtr_mean.astype("float32"),
        "elevation": elev.astype("float32"),
        "latitude": lat_grid.astype("float32"),
    }
    np.savez_compressed(FEATURE_NPZ, **arrays)
    return arrays


def make_matrix(arrays):
    zone = arrays["zone"]
    features = ["p_ann", "cv_monthly", "monsoon_frac", "dtr_mean", "elevation", "latitude"]
    stack = np.column_stack([arrays[f].ravel() for f in features])
    consensus = zone.ravel().astype(int)
    valid = (consensus > 0) & np.all(np.isfinite(stack), axis=1)
    matrix = stack[valid]
    labels = consensus[valid]
    rows, cols = np.where(valid.reshape(zone.shape))
    sample_df = pd.DataFrame(matrix, columns=features)
    sample_df.insert(0, "consensus_zone", labels)
    sample_df.insert(0, "col", cols)
    sample_df.insert(0, "row", rows)
    sample_df.to_csv(FEATURE_CSV, index=False)
    return matrix, labels, valid, features


def reorder_by_precip(labels, p_ann_values):
    new = np.zeros_like(labels)
    order = []
    for lab in sorted(np.unique(labels)):
        order.append((lab, np.nanmean(p_ann_values[labels == lab])))
    order = sorted(order, key=lambda x: x[1])
    mapping = {old: i + 1 for i, (old, _) in enumerate(order)}
    for old, new_lab in mapping.items():
        new[labels == old] = new_lab
    return new, mapping


def run_som(matrix, consensus, features):
    scaler = StandardScaler()
    x = scaler.fit_transform(matrix)
    som = MiniSom(
        x=5,
        y=5,
        input_len=x.shape[1],
        sigma=1.2,
        learning_rate=0.5,
        neighborhood_function="gaussian",
        random_seed=42,
    )
    som.random_weights_init(x)
    som.train_random(x, num_iteration=50000, verbose=False)

    bmus = np.array([som.winner(row) for row in x])
    neuron_ids = bmus[:, 0] * 5 + bmus[:, 1]
    weights = som.get_weights().reshape(-1, x.shape[1])
    km = KMeans(n_clusters=7, n_init=50, random_state=42)
    neuron_clusters = km.fit_predict(weights) + 1
    raw_labels = neuron_clusters[neuron_ids]
    som_labels, mapping = reorder_by_precip(raw_labels, matrix[:, 0])

    sample_n = min(10000, len(x))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(x), size=sample_n, replace=False)
    metrics = {
        "algorithm": "SOM_5x5_neuron_kmeans_7",
        "n_clusters": 7,
        "n_pixels": int(len(x)),
        "sample_n": int(sample_n),
        "silhouette_score_sample": float(silhouette_score(x[sample_idx], som_labels[sample_idx])),
        "davies_bouldin_index_sample": float(davies_bouldin_score(x[sample_idx], som_labels[sample_idx])),
        "calinski_harabasz_score_sample": float(calinski_harabasz_score(x[sample_idx], som_labels[sample_idx])),
        "ari_vs_consensus": float(adjusted_rand_score(consensus, som_labels)),
        "quantization_error": float(som.quantization_error(x)),
        "topographic_error": float(som.topographic_error(x)),
    }
    pd.DataFrame([metrics]).to_csv(TABLE_DIR / "o1_som_validation_metrics.csv", index=False)
    return som_labels, metrics


def write_geotiffs(valid, som_labels, consensus):
    with rasterio.open(ZONE_TIF) as src:
        profile = src.profile.copy()
        base = src.read(1)
    profile.update(dtype="float32", count=1, nodata=-9999.0, compress="lzw")
    som_arr = np.full(base.shape, -9999.0, dtype="float32")
    consensus_flat = np.full(base.size, -9999.0, dtype="float32")
    som_arr.ravel()[valid] = som_labels.astype("float32")
    consensus_flat[valid] = consensus.astype("float32")
    consensus_arr = consensus_flat.reshape(base.shape)
    agree = np.where((som_arr != -9999.0) & (consensus_arr != -9999.0), (som_arr == consensus_arr).astype("float32"), -9999.0)
    for name, arr in [
        ("o1_som_7zone_sensitivity.tif", som_arr),
        ("o1_som_vs_consensus_label_agreement.tif", agree),
    ]:
        with rasterio.open(TIF_DIR / name, "w", **profile) as dst:
            dst.write(arr.astype("float32"), 1)


def plot_panel(valid, som_labels, consensus, metrics):
    with rasterio.open(ZONE_TIF) as src:
        bounds = src.bounds
        base = src.read(1)
    som_arr = np.full(base.shape, np.nan)
    con_arr = np.full(base.shape, np.nan)
    som_arr.ravel()[valid] = som_labels
    con_arr.ravel()[valid] = consensus
    agree = np.where(np.isfinite(som_arr) & np.isfinite(con_arr), som_arr == con_arr, np.nan)
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    panels = [
        ("A  Consensus", con_arr, "tab20", 1, 7, "Zone"),
        ("B  SOM sensitivity", som_arr, "tab20", 1, 7, "Zone"),
        ("C  Label agreement", agree, "Greens", 0, 1, "Agreement"),
    ]
    for ax, (title, arr, cmap, vmin, vmax, label) in zip(axes, panels):
        im = ax.imshow(arr, extent=extent, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, loc="left", fontsize=15, fontweight="bold")
        ax.set_xlabel("Longitude", fontsize=11, fontweight="bold")
        ax.set_ylabel("Latitude", fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=9)
        cb = fig.colorbar(im, ax=ax, shrink=0.75)
        cb.set_label(label, fontsize=10, fontweight="bold")
    fig.savefig(FIG_DIR / "SUPPLEMENT_PUBLICATION_o1_som_sensitivity_panel.png", dpi=300)
    fig.savefig(FIG_DIR / "SUPPLEMENT_PUBLICATION_o1_som_sensitivity_panel.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    names = ["Silhouette", "Davies-Bouldin", "ARI"]
    vals = [metrics["silhouette_score_sample"], metrics["davies_bouldin_index_sample"], metrics["ari_vs_consensus"]]
    ax.bar(names, vals, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_title("SOM sensitivity diagnostics", fontsize=14, fontweight="bold")
    ax.tick_params(labelsize=11)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontweight="bold")
    fig.savefig(FIG_DIR / "SUPPLEMENT_o1_som_validation_metrics.png", dpi=300)
    fig.savefig(FIG_DIR / "SUPPLEMENT_o1_som_validation_metrics.pdf")
    plt.close(fig)


def write_summary(metrics):
    text = f"""# O1 SOM Sensitivity Extension

## Purpose

This extension adds the Self-Organizing Map sensitivity test that was listed in the proposal but missing from the earlier O1 validation.

## Method

The original six O1 feature types were reconstructed from the existing ERA5-Land and SRTM archive for 1985-2014: annual precipitation, monthly precipitation coefficient of variation, JJAS monsoon precipitation fraction, mean diurnal temperature range, elevation, and latitude. Features were z-score standardized. A 5 x 5 SOM was trained using MiniSom, and SOM neuron weights were clustered into seven classes using K-Means. Final SOM classes were reordered by increasing annual precipitation to match the dry-to-wet interpretation of the consensus zones.

## Key Metrics

- Sample Silhouette Score: {metrics['silhouette_score_sample']:.3f}
- Sample Davies-Bouldin Index: {metrics['davies_bouldin_index_sample']:.3f}
- Sample Calinski-Harabasz Score: {metrics['calinski_harabasz_score_sample']:.1f}
- Adjusted Rand Index versus final consensus: {metrics['ari_vs_consensus']:.3f}
- SOM quantization error: {metrics['quantization_error']:.3f}
- SOM topographic error: {metrics['topographic_error']:.3f}

## Manuscript Interpretation

The SOM sensitivity test should be reported as an additional robustness check rather than as a replacement for the validated consensus map. It closes the proposal-method gap and supports a transparent statement that the final hydroclimatic zones were evaluated against K-Means, Ward/HCA, GMM, and SOM sensitivity diagnostics.
"""
    (OUT / "O1_SOM_SENSITIVITY_METHODS_RESULTS.md").write_text(text, encoding="utf-8")


def main():
    arrays = reconstruct_features()
    matrix, consensus, valid, features = make_matrix(arrays)
    som_labels, metrics = run_som(matrix, consensus, features)
    write_geotiffs(valid, som_labels, consensus)
    plot_panel(valid, som_labels, consensus, metrics)
    write_summary(metrics)
    pd.DataFrame([{"item": "feature_cache", "path": str(FEATURE_NPZ), "exists": FEATURE_NPZ.exists()},
                  {"item": "som_metrics", "path": str(TABLE_DIR / "o1_som_validation_metrics.csv"), "exists": True},
                  {"item": "geotiff_count", "path": str(TIF_DIR), "exists": len(list(TIF_DIR.glob("*.tif")))},
                  {"item": "figure_count", "path": str(FIG_DIR), "exists": len(list(FIG_DIR.glob("*.png")))}]).to_csv(TABLE_DIR / "o1_som_completion_checklist.csv", index=False)
    print("Wrote", OUT)
    print(pd.DataFrame([metrics]).to_string(index=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
