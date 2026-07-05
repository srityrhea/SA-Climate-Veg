from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
GEO = ROOT / "output" / "o5_pettitt_prewhitened_trends" / "geotiff"
FIG = ROOT / "output" / "o5_pettitt_prewhitened_trends" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

INDICES = ["PRCPTOT", "RX1day", "CDD", "TXx", "TNn"]
LETTERS = ["A", "B", "C", "D", "E"]
TITLES = {
    "PRCPTOT": "PRCPTOT",
    "RX1day": "RX1day",
    "CDD": "CDD",
    "TXx": "TXx",
    "TNn": "TNn",
}
UNITS = {
    "PRCPTOT": "mm decade-1",
    "RX1day": "mm decade-1",
    "CDD": "days decade-1",
    "TXx": "degC decade-1",
    "TNn": "degC decade-1",
}


def read_tif(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        bounds = src.bounds
    return arr, bounds


def add_map(ax, arr, bounds, title, letter, cmap, norm=None, vmin=None, vmax=None):
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    im = ax.imshow(arr, extent=extent, origin="upper", cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
    ax.set_title(f"{letter}  {title}", fontsize=18, weight="bold", pad=10, loc="left")
    ax.set_xlabel("Longitude", fontsize=13, fontweight="bold")
    ax.set_ylabel("Latitude", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=11, width=1.0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.set_aspect("equal")
    return im


def panel_sen_slope():
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    axes = axes.ravel()
    for i, idx in enumerate(INDICES):
        tif = GEO / f"o5_ext_ssp585_{idx}_sen_slope_decade.tif"
        arr, bounds = read_tif(tif)
        valid = arr[np.isfinite(arr)]
        if idx == "CDD":
            limit = max(abs(np.nanmin(valid)), abs(np.nanmax(valid)))
            norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
            cmap = "BrBG"
            im = add_map(axes[i], arr, bounds, f"{TITLES[idx]}", LETTERS[i], cmap, norm=norm)
        elif idx in ["TXx", "TNn"]:
            im = add_map(axes[i], arr, bounds, f"{TITLES[idx]}", LETTERS[i], "YlOrRd", vmin=0, vmax=np.nanmax(valid))
        else:
            im = add_map(axes[i], arr, bounds, f"{TITLES[idx]}", LETTERS[i], "YlGnBu", vmin=0, vmax=np.nanmax(valid))
        cb = fig.colorbar(im, ax=axes[i], shrink=0.75)
        cb.ax.tick_params(labelsize=11, width=1.0)
        cb.set_label(UNITS[idx], fontsize=12, fontweight="bold")
        for label in cb.ax.get_yticklabels():
            label.set_fontweight("bold")
    axes[-1].axis("off")
    out_png = FIG / "MANUSCRIPT_PUBLICATION_o5_ssp585_sen_slope_panel.png"
    out_pdf = FIG / "MANUSCRIPT_PUBLICATION_o5_ssp585_sen_slope_panel.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    return out_png, out_pdf


def panel_pettitt_year():
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    axes = axes.ravel()
    for i, idx in enumerate(INDICES):
        tif = GEO / f"o5_ext_ssp585_{idx}_pettitt_year.tif"
        sig_tif = GEO / f"o5_ext_ssp585_{idx}_pettitt_sig.tif"
        arr, bounds = read_tif(tif)
        sig, _ = read_tif(sig_tif)
        # Keep non-significant zones visible but muted.
        masked = arr.copy()
        masked[np.isfinite(sig) & (sig < 0.5)] = np.nan
        im = add_map(
            axes[i],
            masked,
            bounds,
            f"{TITLES[idx]}",
            LETTERS[i],
            "viridis",
            vmin=2035,
            vmax=2095,
        )
        cb = fig.colorbar(im, ax=axes[i], shrink=0.75)
        cb.ax.tick_params(labelsize=11, width=1.0)
        cb.set_label("Change year", fontsize=12, fontweight="bold")
        for label in cb.ax.get_yticklabels():
            label.set_fontweight("bold")
    axes[-1].axis("off")
    out_png = FIG / "SUPPLEMENT_PUBLICATION_o5_ssp585_pettitt_year_panel.png"
    out_pdf = FIG / "SUPPLEMENT_PUBLICATION_o5_ssp585_pettitt_year_panel.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    return out_png, out_pdf


if __name__ == "__main__":
    outputs = []
    outputs.extend(panel_sen_slope())
    outputs.extend(panel_pettitt_year())
    for out in outputs:
        print(out)
