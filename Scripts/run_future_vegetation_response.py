from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from matplotlib.colors import TwoSlopeNorm
from rasterio.transform import from_origin
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "o8_future_vegetation_response"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
TIF_DIR = OUT / "geotiff"
for d in [TABLE_DIR, FIG_DIR, TIF_DIR]:
    d.mkdir(parents=True, exist_ok=True)

KEY_INDICES = ["PRCPTOT", "RX1day", "CDD", "TXx", "TNn"]
VEG_METRICS = ["NDVI", "EVI", "GPP", "NPP"]
PERIODS = {
    "near_future_2021_2060": (2021, 2060),
    "far_future_2061_2100": (2061, 2100),
}
ALPHAS = np.logspace(-3, 3, 25)
ZONE_IDS = {f"Z{i}": i - 1 for i in range(1, 8)}


def set_style():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "savefig.dpi": 350,
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
    return da


def load_zone_grid():
    with xr.open_dataset(ROOT / "output/zones/hydroclimatic_zones_SA.nc") as ds:
        z = standardise_xy(ds["zone"]).load()
    return z


def write_zone_geotiff(path, zone_values):
    zones = load_zone_grid()
    arr = np.full(zones.shape, np.nan, dtype="float32")
    for zone, val in zone_values.items():
        zid = ZONE_IDS[zone]
        arr = np.where(zones.values == zid, val, arr)
    lat = zones["lat"].values
    lon = zones["lon"].values
    out = arr.copy()
    if lat[0] < lat[-1]:
        out = out[::-1, :]
        lat = lat[::-1]
    xres = float(abs(lon[1] - lon[0]))
    yres = float(abs(lat[1] - lat[0]))
    transform = from_origin(float(lon.min()) - xres / 2, float(lat.max()) + yres / 2, xres, yres)
    nodata = -9999.0
    data = np.where(np.isfinite(out), out, nodata).astype("float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
        compress="deflate",
    ) as dst:
        dst.write(data, 1)


def load_training_table():
    train = pd.read_csv(ROOT / "output/vegetation_climate_linkage/tables/vegetation_climate_linkage_dataset_zone_year.csv")
    keep = ["year", "zone"] + VEG_METRICS + KEY_INDICES
    return train[keep].copy()


def load_future_climate():
    fut = pd.read_csv(ROOT / "output/etccdi_fast/tables/fast_etccdi_zone_annual_means.csv")
    fut = fut[fut["index"].isin(KEY_INDICES)].copy()
    ens = fut.groupby(["scenario", "index", "year", "zone"], as_index=False)["zone_mean"].mean()
    wide = ens.pivot_table(index=["scenario", "year", "zone"], columns="index", values="zone_mean").reset_index()
    return wide


def confidence_class(r, skill_ratio, n):
    if n < 12 or not np.isfinite(r) or not np.isfinite(skill_ratio):
        return "low"
    if r >= 0.55 and skill_ratio < 0.85:
        return "high"
    if r >= 0.30 and skill_ratio < 1.00:
        return "moderate"
    return "low"


def fit_models(train):
    model_rows = []
    pred_rows = []
    fitted = {}
    for zone in sorted(train["zone"].unique()):
        zdf = train[train["zone"] == zone].copy()
        for metric in VEG_METRICS:
            sub = zdf.dropna(subset=[metric] + KEY_INDICES).copy()
            if len(sub) < 10:
                continue
            X = sub[KEY_INDICES].astype(float).values
            y = sub[metric].astype(float).values
            years = sub["year"].astype(int).values
            loo = LeaveOneOut()
            yhat = np.full_like(y, np.nan, dtype=float)
            for tr, te in loo.split(X):
                model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
                model.fit(X[tr], y[tr])
                yhat[te] = model.predict(X[te])
            rmse = mean_squared_error(y, yhat) ** 0.5
            mae = mean_absolute_error(y, yhat)
            clim_rmse = mean_squared_error(y, np.repeat(y.mean(), len(y))) ** 0.5
            skill_ratio = rmse / clim_rmse if clim_rmse > 0 else np.nan
            r = pearsonr(y, yhat).statistic if len(y) > 2 and np.nanstd(yhat) > 0 else np.nan
            r2 = r2_score(y, yhat)
            final_model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
            final_model.fit(X, y)
            fitted[(zone, metric)] = {
                "model": final_model,
                "baseline_mean": float(np.mean(y)),
                "train_year_start": int(years.min()),
                "train_year_end": int(years.max()),
            }
            coef = final_model.named_steps["ridgecv"].coef_
            model_rows.append(
                {
                    "zone": zone,
                    "vegetation_metric": metric,
                    "n_years": len(y),
                    "train_year_start": int(years.min()),
                    "train_year_end": int(years.max()),
                    "loyo_r": r,
                    "loyo_r2": r2,
                    "loyo_rmse": rmse,
                    "loyo_mae": mae,
                    "rmse_climatology_ratio": skill_ratio,
                    "confidence": confidence_class(r, skill_ratio, len(y)),
                    "baseline_observed_mean": float(np.mean(y)),
                    **{f"coef_{idx}": float(c) for idx, c in zip(KEY_INDICES, coef)},
                }
            )
            for yr, obs, pred in zip(years, y, yhat):
                pred_rows.append({"zone": zone, "vegetation_metric": metric, "year": int(yr), "observed": obs, "loyo_predicted": pred})
    return fitted, pd.DataFrame(model_rows), pd.DataFrame(pred_rows)


def project_future(fitted, future_clim):
    annual_rows = []
    period_rows = []
    for (zone, metric), info in fitted.items():
        model = info["model"]
        baseline = info["baseline_mean"]
        zf = future_clim[(future_clim["zone"] == zone) & (future_clim["scenario"].isin(["ssp245", "ssp585"]))].dropna(subset=KEY_INDICES)
        if zf.empty:
            continue
        X = zf[KEY_INDICES].astype(float).values
        pred = model.predict(X)
        temp = zf[["scenario", "year", "zone"]].copy()
        temp["vegetation_metric"] = metric
        temp["predicted_potential_response"] = pred
        temp["delta_vs_observed_baseline"] = pred - baseline
        annual_rows.extend(temp.to_dict("records"))
        for scenario in ["ssp245", "ssp585"]:
            sf = temp[temp["scenario"] == scenario]
            for period, (start, end) in PERIODS.items():
                pf = sf[(sf["year"] >= start) & (sf["year"] <= end)]
                if pf.empty:
                    continue
                period_rows.append(
                    {
                        "zone": zone,
                        "vegetation_metric": metric,
                        "scenario": scenario,
                        "period": period,
                        "baseline_observed_mean": baseline,
                        "future_predicted_mean": pf["predicted_potential_response"].mean(),
                        "delta_vs_observed_baseline": pf["delta_vs_observed_baseline"].mean(),
                        "n_years": len(pf),
                    }
                )
    return pd.DataFrame(annual_rows), pd.DataFrame(period_rows)


def add_confidence(period, model_metrics):
    cols = ["zone", "vegetation_metric", "loyo_r", "loyo_rmse", "rmse_climatology_ratio", "confidence"]
    return period.merge(model_metrics[cols], on=["zone", "vegetation_metric"], how="left")


def create_geotiffs(period):
    rows = []
    for metric in VEG_METRICS:
        for scenario in ["ssp245", "ssp585"]:
            for period_name in PERIODS:
                sub = period[(period["vegetation_metric"] == metric) & (period["scenario"] == scenario) & (period["period"] == period_name)]
                if sub.empty:
                    continue
                vals = dict(zip(sub["zone"], sub["delta_vs_observed_baseline"]))
                path = TIF_DIR / f"o8_{metric}_{scenario}_{period_name}_potential_response_delta.tif"
                write_zone_geotiff(path, vals)
                rows.append(
                    {
                        "vegetation_metric": metric,
                        "scenario": scenario,
                        "period": period_name,
                        "geotiff": str(path.relative_to(ROOT)),
                    }
                )
    return pd.DataFrame(rows)


def plot_main_panel(period):
    far = period[period["period"] == "far_future_2061_2100"].copy()
    plot_metrics = ["NDVI", "EVI", "NPP"]
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.9), constrained_layout=True)
    axes = axes.ravel()
    zones = load_zone_grid()
    lon = zones["lon"].values
    lat = zones["lat"].values
    letters = list("ABC")
    for ax, metric, letter in zip(axes, plot_metrics, letters):
        sub = far[
            (far["vegetation_metric"] == metric)
            & (far["scenario"] == "ssp585")
            & (far["confidence"].isin(["moderate", "high"]))
        ]
        zone_vals = dict(zip(sub["zone"], sub["delta_vs_observed_baseline"]))
        arr = np.full(zones.shape, np.nan, dtype="float32")
        for zone, val in zone_vals.items():
            arr = np.where(zones.values == ZONE_IDS[zone], val, arr)
        if np.isfinite(arr).any():
            lim = max(abs(np.nanpercentile(arr, 2)), abs(np.nanpercentile(arr, 98)))
        else:
            lim = 0.01
        im = ax.pcolormesh(lon, lat, np.ma.masked_invalid(arr), cmap="BrBG", norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim), shading="auto")
        ax.text(-0.13, 1.08, letter, transform=ax.transAxes, fontsize=20, fontweight="bold", bbox=dict(facecolor="white", edgecolor="black", pad=4), clip_on=False)
        ax.text(0.5, 1.08, metric, transform=ax.transAxes, ha="center", va="bottom", fontsize=16, fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")
        cb = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.02)
        cb.set_label("SSP585 response vs baseline", fontweight="bold")
    out = FIG_DIR / "MANUSCRIPT_PUBLICATION_o8_future_vegetation_response_panel.png"
    fig.savefig(out, dpi=350, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_validation_and_summary(model_metrics, period):
    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    mat = model_metrics.pivot(index="vegetation_metric", columns="zone", values="loyo_r").reindex(VEG_METRICS)
    im = ax.imshow(mat.values, cmap="RdBu", norm=TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1), aspect="auto")
    ax.set_xticks(range(mat.shape[1]), mat.columns, fontweight="bold")
    ax.set_yticks(range(mat.shape[0]), mat.index, fontweight="bold")
    ax.set_xlabel("Hydroclimatic zone", fontweight="bold")
    ax.set_ylabel("Vegetation metric", fontweight="bold")
    for y in range(mat.shape[0]):
        for x in range(mat.shape[1]):
            val = mat.values[y, x]
            if np.isfinite(val):
                ax.text(x, y, f"{val:.2f}", ha="center", va="center", fontsize=9, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Leave-one-year-out r", fontweight="bold")
    out = FIG_DIR / "SUPPLEMENT_PUBLICATION_o8_model_validation_heatmap.png"
    fig.savefig(out, dpi=350, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    far = period[
        (period["period"] == "far_future_2061_2100")
        & (period["scenario"] == "ssp585")
        & (period["confidence"].isin(["moderate", "high"]))
    ].copy()
    mat = far.pivot(index="vegetation_metric", columns="zone", values="delta_vs_observed_baseline").reindex(["NDVI", "EVI", "NPP"])
    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    lim = max(abs(np.nanpercentile(mat.values, 2)), abs(np.nanpercentile(mat.values, 98)))
    im = ax.imshow(mat.values, cmap="BrBG", norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim), aspect="auto")
    ax.set_xticks(range(mat.shape[1]), mat.columns, fontweight="bold")
    ax.set_yticks(range(mat.shape[0]), mat.index, fontweight="bold")
    ax.set_xlabel("Hydroclimatic zone", fontweight="bold")
    ax.set_ylabel("Vegetation metric", fontweight="bold")
    for y in range(mat.shape[0]):
        for x in range(mat.shape[1]):
            val = mat.values[y, x]
            if np.isfinite(val):
                ax.text(x, y, f"{val:.3f}", ha="center", va="center", fontsize=8.5, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Potential response vs observed baseline", fontweight="bold")
    out = FIG_DIR / "SUPPLEMENT_PUBLICATION_o8_far_future_response_heatmap.png"
    fig.savefig(out, dpi=350, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_note(model_metrics, period):
    high_mod = model_metrics[model_metrics["confidence"].isin(["high", "moderate"])]
    far = period[(period["scenario"] == "ssp585") & (period["period"] == "far_future_2061_2100")]
    strongest = far.iloc[far["delta_vs_observed_baseline"].abs().argmax()]
    text = [
        "# O8 Future Vegetation Response/Exposure",
        "",
        "This analysis is a constrained future vegetation-climate linkage extension. It should be described as potential vegetation response or exposure, not as a deterministic vegetation forecast.",
        "",
        f"Moderate/high validation confidence models: {len(high_mod)} of {len(model_metrics)} zone-metric models.",
        f"Largest far-future SSP5-8.5 potential response: {strongest['vegetation_metric']} in {strongest['zone']} ({strongest['delta_vs_observed_baseline']:.3f} relative to observed baseline).",
        "",
        "Outputs:",
        "- tables/o8_zone_metric_model_validation.csv",
        "- tables/o8_future_vegetation_annual_predictions.csv",
        "- tables/o8_future_vegetation_period_response.csv",
        "- figures/MANUSCRIPT_PUBLICATION_o8_future_vegetation_response_panel.png",
        "- figures/SUPPLEMENT_PUBLICATION_o8_model_validation_heatmap.png",
        "- figures/SUPPLEMENT_PUBLICATION_o8_far_future_response_heatmap.png",
        "- geotiff/o8_*_potential_response_delta.tif",
    ]
    (OUT / "O8_FUTURE_VEGETATION_RESPONSE_METHODS_RESULTS.md").write_text("\n".join(text), encoding="utf-8")


def main():
    set_style()
    train = load_training_table()
    future = load_future_climate()
    fitted, metrics, cv_preds = fit_models(train)
    annual, period = project_future(fitted, future)
    period = add_confidence(period, metrics)
    inventory = create_geotiffs(period)
    metrics.to_csv(TABLE_DIR / "o8_zone_metric_model_validation.csv", index=False)
    cv_preds.to_csv(TABLE_DIR / "o8_observed_leave_one_year_out_predictions.csv", index=False)
    annual.to_csv(TABLE_DIR / "o8_future_vegetation_annual_predictions.csv", index=False)
    period.to_csv(TABLE_DIR / "o8_future_vegetation_period_response.csv", index=False)
    inventory.to_csv(TABLE_DIR / "o8_geotiff_inventory.csv", index=False)
    plot_main_panel(period)
    plot_validation_and_summary(metrics, period)
    write_note(metrics, period)
    print("O8 completed:", OUT)
    print("models:", len(metrics), "geotiffs:", len(inventory))


if __name__ == "__main__":
    main()
