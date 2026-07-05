from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "o5_pettitt_prewhitened_trends"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
TIF_DIR = OUT / "geotiff"
for d in [TABLE_DIR, FIG_DIR, TIF_DIR]:
    d.mkdir(parents=True, exist_ok=True)

INPUT = ROOT / "output" / "etccdi_fast" / "tables" / "fast_etccdi_zone_annual_means.csv"
ZONE_TIF = ROOT / "output" / "zones" / "tif" / "zones_consensus.tif"

KEY_INDICES = ["PRCPTOT", "RX1day", "CDD", "TXx", "TNn"]
SCENARIOS = ["historical", "ssp245", "ssp585"]


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mk_test(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 4:
        return np.nan, np.nan, np.nan
    s = 0
    for k in range(n - 1):
        s += np.sign(x[k + 1:] - x[k]).sum()
    unique, counts = np.unique(x, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if var_s <= 0:
        return np.nan, np.nan, np.nan
    if s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0
    p = 2 * (1 - norm_cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1))
    return tau, z, p


def sen_slope(years, x):
    years = np.asarray(years, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(years) & np.isfinite(x)
    years = years[mask]
    x = x[mask]
    n = len(x)
    if n < 2:
        return np.nan
    slopes = []
    for i in range(n - 1):
        dy = x[i + 1:] - x[i]
        dt = years[i + 1:] - years[i]
        valid = dt != 0
        slopes.extend((dy[valid] / dt[valid]).tolist())
    return float(np.median(slopes)) if slopes else np.nan


def lag1_autocorr(x):
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x)
    x = x[mask]
    if len(x) < 4:
        return np.nan
    x0 = x[:-1]
    x1 = x[1:]
    if np.std(x0) == 0 or np.std(x1) == 0:
        return 0.0
    return float(np.corrcoef(x0, x1)[0, 1])


def prewhiten_series(years, x):
    years = np.asarray(years, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(years) & np.isfinite(x)
    years = years[mask]
    x = x[mask]
    if len(x) < 5:
        return years, x, np.nan
    r1 = lag1_autocorr(x)
    if not np.isfinite(r1):
        return years, x, r1
    # Conservative lag-1 pre-whitening. Negative autocorrelation is not removed.
    r_use = r1 if r1 > 0 else 0.0
    xp = x[1:] - r_use * x[:-1]
    yp = years[1:]
    return yp, xp, r1


def pettitt_test(years, x):
    years = np.asarray(years, dtype=int)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x)
    years = years[mask]
    x = x[mask]
    n = len(x)
    if n < 8:
        return np.nan, np.nan, np.nan
    ranks = pd.Series(x).rank(method="average").to_numpy()
    u = np.array([2 * np.sum(ranks[:t]) - t * (n + 1) for t in range(1, n + 1)], dtype=float)
    k_idx = int(np.argmax(np.abs(u)))
    k_stat = float(np.abs(u[k_idx]))
    p = min(1.0, 2.0 * math.exp((-6.0 * k_stat * k_stat) / (n**3 + n**2)))
    year = int(years[k_idx])
    before = float(np.nanmean(x[: k_idx + 1]))
    after = float(np.nanmean(x[k_idx + 1 :])) if k_idx + 1 < n else np.nan
    return year, k_stat, p, before, after, after - before if np.isfinite(after) else np.nan


def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan, dtype=float)
    mask = np.isfinite(pvals)
    p = pvals[mask]
    m = len(p)
    if m == 0:
        return out
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * m / (np.arange(1, m + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    tmp = np.empty(m)
    tmp[order] = adj
    out[mask] = tmp
    return out


def scenario_year_filter(df):
    mask_hist = (df["scenario"] == "historical") & df["year"].between(1985, 2014)
    mask_future = (df["scenario"].isin(["ssp245", "ssp585"])) & df["year"].between(2015, 2100)
    return df[mask_hist | mask_future].copy()


def compute_ensemble(df):
    group_cols = ["scenario", "index", "year", "zone", "zone_id", "units"]
    ens = df.groupby(group_cols, as_index=False)["zone_mean"].mean()
    ens["series_type"] = "ensemble_mean"
    ens["model"] = "ensemble_mean"
    return ens


def make_heatmap(df, metric, scenario, indices, path, title, cmap="RdBu_r"):
    sub = df[(df["scenario"] == scenario) & (df["index"].isin(indices))].copy()
    if sub.empty:
        return
    pivot = sub.pivot(index="index", columns="zone", values=metric).reindex(indices)
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    im = ax.imshow(pivot.values.astype(float), aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2g}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def zone_to_raster(zone_arr, zone_values, nodata=-9999.0):
    out = np.full(zone_arr.shape, nodata, dtype="float32")
    for zid, value in zone_values.items():
        if np.isfinite(value):
            out[zone_arr == zid] = float(value)
    return out


def write_tif(template_path, out_path, arr, nodata=-9999.0):
    with rasterio.open(template_path) as src:
        profile = src.profile.copy()
        profile.update(dtype="float32", count=1, nodata=nodata, compress="lzw")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(arr.astype("float32"), 1)


def export_geotiffs(results):
    with rasterio.open(ZONE_TIF) as src:
        zones = src.read(1)
    valid_zone_ids = [1, 2, 3, 4, 5, 6, 7]
    for scenario in SCENARIOS:
        for idx in results["index"].unique():
            sub = results[(results["scenario"] == scenario) & (results["index"] == idx)]
            if sub.empty:
                continue
            zid = sub["zone"].str.replace("Z", "", regex=False).astype(int)
            metrics = {
                "sen_slope_per_decade": "sen_slope_decade",
                "pw_mk_p_fdr": "pw_mk_fdr_p",
                "pw_mk_significant_fdr_0_05": "pw_mk_sig",
                "pettitt_change_year": "pettitt_year",
                "pettitt_p_fdr": "pettitt_fdr_p",
                "pettitt_significant_fdr_0_05": "pettitt_sig",
            }
            for col, name in metrics.items():
                vals = {}
                for zone_num, val in zip(zid, sub[col]):
                    if col.endswith("significant_fdr_0_05"):
                        vals[zone_num] = 1.0 if bool(val) else 0.0
                    else:
                        vals[zone_num] = float(val) if pd.notna(val) else np.nan
                arr = zone_to_raster(zones, vals)
                out = TIF_DIR / f"o5_ext_{scenario}_{idx}_{name}.tif"
                write_tif(ZONE_TIF, out, arr)


def main():
    df = pd.read_csv(INPUT)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["zone_mean"] = pd.to_numeric(df["zone_mean"], errors="coerce")
    df = scenario_year_filter(df)
    ens = compute_ensemble(df)
    ens.to_csv(TABLE_DIR / "o5_ext_ensemble_zone_annual_series.csv", index=False)

    records = []
    for (scenario, index, zone), g in ens.groupby(["scenario", "index", "zone"]):
        g = g.sort_values("year")
        years = g["year"].astype(int).to_numpy()
        x = g["zone_mean"].astype(float).to_numpy()
        units = g["units"].iloc[0]
        slope = sen_slope(years, x)
        tau, z, p_mk = mk_test(x)
        py, px, r1 = prewhiten_series(years, x)
        pw_tau, pw_z, pw_p = mk_test(px)
        pw_slope = sen_slope(py, px)
        pet_year, pet_k, pet_p, pet_before, pet_after, pet_delta = pettitt_test(years, x)
        records.append({
            "series_type": "ensemble_mean",
            "scenario": scenario,
            "index": index,
            "zone": zone,
            "start_year": int(np.nanmin(years)),
            "end_year": int(np.nanmax(years)),
            "n_years": int(len(years)),
            "units": units,
            "lag1_autocorrelation": r1,
            "sen_slope_per_year": slope,
            "sen_slope_per_decade": slope * 10 if np.isfinite(slope) else np.nan,
            "mk_tau_original": tau,
            "mk_z_original": z,
            "mk_p_original": p_mk,
            "pw_mk_tau": pw_tau,
            "pw_mk_z": pw_z,
            "pw_mk_p": pw_p,
            "pw_sen_slope_per_year": pw_slope,
            "pw_sen_slope_per_decade": pw_slope * 10 if np.isfinite(pw_slope) else np.nan,
            "pettitt_change_year": pet_year,
            "pettitt_k": pet_k,
            "pettitt_p": pet_p,
            "pettitt_before_mean": pet_before,
            "pettitt_after_mean": pet_after,
            "pettitt_delta_after_minus_before": pet_delta,
        })

    res = pd.DataFrame(records)
    for scenario in res["scenario"].unique():
        for index in res["index"].unique():
            mask = (res["scenario"] == scenario) & (res["index"] == index)
            res.loc[mask, "pw_mk_p_fdr"] = bh_fdr(res.loc[mask, "pw_mk_p"].to_numpy())
            res.loc[mask, "pettitt_p_fdr"] = bh_fdr(res.loc[mask, "pettitt_p"].to_numpy())
    res["pw_mk_significant_fdr_0_05"] = res["pw_mk_p_fdr"] < 0.05
    res["pettitt_significant_fdr_0_05"] = res["pettitt_p_fdr"] < 0.05
    res["pw_mk_trend"] = np.where(
        res["pw_mk_significant_fdr_0_05"],
        np.where(res["sen_slope_per_decade"] > 0, "increasing", "decreasing"),
        "not_significant",
    )
    res.to_csv(TABLE_DIR / "o5_ext_pettitt_prewhitened_mk_by_zone.csv", index=False)

    key = res[res["index"].isin(KEY_INDICES)].copy()
    key.to_csv(TABLE_DIR / "o5_ext_key_indices_pettitt_prewhitened_mk_by_zone.csv", index=False)

    summary_rows = []
    for (scenario, index), g in key.groupby(["scenario", "index"]):
        sig_mk = int(g["pw_mk_significant_fdr_0_05"].sum())
        sig_pet = int(g["pettitt_significant_fdr_0_05"].sum())
        max_abs = g.loc[g["sen_slope_per_decade"].abs().idxmax()]
        years = g.loc[g["pettitt_significant_fdr_0_05"], "pettitt_change_year"].dropna()
        common_year = int(years.mode().iloc[0]) if len(years) else np.nan
        summary_rows.append({
            "scenario": scenario,
            "index": index,
            "n_zones": len(g),
            "prewhitened_mk_fdr_significant_zones": sig_mk,
            "pettitt_fdr_significant_zones": sig_pet,
            "largest_abs_sen_slope_zone": max_abs["zone"],
            "largest_abs_sen_slope_per_decade": max_abs["sen_slope_per_decade"],
            "common_significant_pettitt_year": common_year,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLE_DIR / "o5_ext_key_index_summary.csv", index=False)

    for scenario in SCENARIOS:
        make_heatmap(
            key,
            "sen_slope_per_decade",
            scenario,
            KEY_INDICES,
            FIG_DIR / f"o5_ext_{scenario}_sen_slope_heatmap.png",
            f"{scenario}: Sen slope per decade",
        )
        make_heatmap(
            key,
            "pettitt_change_year",
            scenario,
            KEY_INDICES,
            FIG_DIR / f"o5_ext_{scenario}_pettitt_year_heatmap.png",
            f"{scenario}: Pettitt change-point year",
            cmap="viridis",
        )

    export_geotiffs(res)

    checklist = pd.DataFrame([
        {"item": "input_zone_annual_series", "path": str(INPUT), "exists": INPUT.exists()},
        {"item": "full_results_table", "path": str(TABLE_DIR / "o5_ext_pettitt_prewhitened_mk_by_zone.csv"), "exists": True},
        {"item": "key_results_table", "path": str(TABLE_DIR / "o5_ext_key_indices_pettitt_prewhitened_mk_by_zone.csv"), "exists": True},
        {"item": "summary_table", "path": str(TABLE_DIR / "o5_ext_key_index_summary.csv"), "exists": True},
        {"item": "geotiff_count", "path": str(TIF_DIR), "exists": len(list(TIF_DIR.glob("*.tif")))},
        {"item": "figure_count", "path": str(FIG_DIR), "exists": len(list(FIG_DIR.glob("*.png")))},
    ])
    checklist.to_csv(TABLE_DIR / "o5_ext_completion_checklist.csv", index=False)
    print("Wrote", OUT)
    print(checklist.to_string(index=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
