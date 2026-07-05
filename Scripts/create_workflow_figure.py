from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
import matplotlib.patheffects as pe


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "workflow_figure"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "savefig.dpi": 450,
})

COLORS = {
    "data": "#E8F3FF",
    "prep": "#EAF7EF",
    "zone": "#FFF4D6",
    "skill": "#F1ECFF",
    "proj": "#EDF2FF",
    "extreme": "#FFEDE8",
    "veg": "#E7F7F5",
    "out": "#F3F4F6",
    "edge": "#243447",
    "text": "#18212F",
    "muted": "#586174",
}


def box(ax, x, y, w, h, title, lines, fc, fs=8.4):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.15,
        edgecolor=COLORS["edge"],
        facecolor=fc,
        zorder=2,
    )
    patch.set_path_effects([
        pe.SimplePatchShadow(offset=(1.2, -1.2), shadow_rgbFace=(0, 0, 0), alpha=0.10),
        pe.Normal(),
    ])
    ax.add_patch(patch)
    ax.text(x + 0.018, y + h - 0.030, title, ha="left", va="top",
            fontsize=10.8, fontweight="bold", color="#101827", zorder=3)
    ax.text(x + 0.018, y + h - 0.080, "\n".join(lines), ha="left", va="top",
            fontsize=fs, color=COLORS["text"], linespacing=1.18, zorder=3)


def arrow(ax, start, end, rad=0.0, color="#334155", lw=1.25):
    arr = FancyArrowPatch(
        start, end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=1,
    )
    ax.add_patch(arr)


def stage_label(ax, x, y, label):
    patch = FancyBboxPatch(
        (x, y), 0.065, 0.032,
        boxstyle="round,pad=0.005,rounding_size=0.010",
        linewidth=0.7,
        edgecolor="#94A3B8",
        facecolor="#FFFFFF",
        zorder=4,
    )
    ax.add_patch(patch)
    ax.text(x + 0.0325, y + 0.016, label, ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="#475569", zorder=5)


def mini_map(ax, x, y, scale=1.0):
    pts = [
        (x + 0.000 * scale, y + 0.055 * scale),
        (x + 0.030 * scale, y + 0.108 * scale),
        (x + 0.070 * scale, y + 0.113 * scale),
        (x + 0.107 * scale, y + 0.095 * scale),
        (x + 0.123 * scale, y + 0.055 * scale),
        (x + 0.094 * scale, y + 0.043 * scale),
        (x + 0.070 * scale, y + 0.014 * scale),
        (x + 0.063 * scale, y - 0.030 * scale),
        (x + 0.044 * scale, y + 0.004 * scale),
        (x + 0.028 * scale, y + 0.032 * scale),
    ]
    poly = Polygon(pts, closed=True, facecolor="#B9DCC4", edgecolor="#3E6B55", lw=0.9, zorder=3)
    ax.add_patch(poly)
    ax.text(x + 0.066 * scale, y - 0.045 * scale, "South Asia\nanalysis domain",
            ha="center", va="top", fontsize=6.8, fontweight="bold", color="#315B47", zorder=4)


def main():
    fig, ax = plt.subplots(figsize=(17.0, 10.2))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.965,
            "Workflow for Zone-Aware CMIP6 Climate-Extreme and Vegetation-Response Analysis",
            ha="center", va="top", fontsize=17.5, fontweight="bold", color="#111827")
    ax.text(0.5, 0.928,
            "Hydroclimatic zoning, model screening, skill-weighted projections, extremes, trends, and vegetation exposure for South Asia",
            ha="center", va="top", fontsize=10.2, color=COLORS["muted"])

    # Top row
    box(ax, 0.055, 0.695, 0.205, 0.175, "Input Datasets", [
        "ERA5-Land: precipitation, Tmax, Tmin",
        "CMIP6 historical + ScenarioMIP",
        "SRTM elevation",
        "MODIS: NDVI, EVI, GPP, NPP",
        "Common South Asian land mask",
    ], COLORS["data"], fs=8.2)
    mini_map(ax, 0.168, 0.810, scale=0.62)

    box(ax, 0.300, 0.695, 0.190, 0.175, "Preprocessing", [
        "Clip to study domain",
        "Regrid to 0.25 deg grid",
        "Unit conversion and quality checks",
        "Annual aggregation",
        "Zone-year summaries",
    ], COLORS["prep"], fs=8.2)

    box(ax, 0.535, 0.695, 0.205, 0.175, "Hydroclimatic Zoning", [
        "Features: Pann, rainfall variability,",
        "JJAS fraction, DTR, elevation, latitude",
        "K-Means + Ward + GMM consensus",
        "Seven dry-to-wet zones",
        "SOM sensitivity validation",
    ], COLORS["zone"], fs=8.0)

    box(ax, 0.785, 0.695, 0.170, 0.175, "Zone Outputs", [
        "Hydroclimatic zone map",
        "Zone characterization table",
        "Validation metrics",
        "SOM robustness evidence",
        "AOI masks for maps",
    ], COLORS["out"], fs=8.0)

    # Middle row
    box(ax, 0.055, 0.435, 0.205, 0.175, "CMIP6 Skill Evaluation", [
        "Historical period: 1985-2014",
        "Metrics: r, RMSE, bias, KGE",
        "Variable-specific model screening",
        "Top pools for precipitation, Tmax, Tmin",
        "Skill weights for ensemble",
    ], COLORS["skill"], fs=8.0)

    box(ax, 0.300, 0.435, 0.190, 0.175, "Projection Ensemble", [
        "SSP2-4.5 and SSP5-8.5",
        "Zone-level bias correction",
        "Skill-weighted annual ensemble",
        "5th-95th percentile spread",
        "Near, mid, far-future periods",
    ], COLORS["proj"], fs=8.0)

    box(ax, 0.535, 0.435, 0.205, 0.175, "Extreme Diagnostics", [
        "ETCCDI-style annual indices",
        "PRCPTOT, RX1day, CDD, TXx, TNn",
        "Spatial change maps",
        "Sen slope + MK/PW-MK tests",
        "Pettitt change points + FDR",
    ], COLORS["extreme"], fs=8.0)

    box(ax, 0.785, 0.435, 0.170, 0.175, "Scenario Emergence", [
        "SSP5-8.5 minus SSP2-4.5",
        "Signal-to-noise separation",
        "Far-future amplification maps",
        "QGIS-ready GeoTIFF layers",
        "Hotspot interpretation",
    ], "#FFF8E7", fs=8.0)

    # Bottom row
    box(ax, 0.055, 0.190, 0.205, 0.170, "Vegetation Linkage", [
        "MODIS trends: 2000/2001-2025",
        "NDVI, EVI, GPP, NPP by zone",
        "Lag-0 and lag-1 Spearman links",
        "Precipitation and heat controls",
        "FDR-screened relationships",
    ], COLORS["veg"], fs=7.9)

    box(ax, 0.300, 0.190, 0.190, 0.170, "Future Vegetation Response", [
        "Ridge response models",
        "Leave-one-year-out validation",
        "Confidence-qualified responses",
        "NDVI, EVI, NPP maps retained",
        "Low-confidence GPP excluded",
    ], "#EAF7F2", fs=7.9)

    box(ax, 0.535, 0.190, 0.420, 0.170, "Final Manuscript Evidence", [
        "Zone-specific climate-extreme risk and vegetation-response interpretation",
        "Main figures/tables plus supplementary diagnostics and inventories",
        "Publication maps and QGIS-ready GeoTIFFs for cartographic refinement",
        "Transparent separation of robust climate signals and lower-confidence ecological extrapolations",
    ], "#F8FAFC", fs=8.0)

    # Stage tags
    for i, x in enumerate([0.058, 0.303, 0.538, 0.788], start=1):
        stage_label(ax, x, 0.878, f"Stage {i}")
    for i, x in enumerate([0.058, 0.303, 0.538, 0.788], start=5):
        stage_label(ax, x, 0.618, f"Stage {i}")
    for i, x in enumerate([0.058, 0.303, 0.538], start=9):
        stage_label(ax, x, 0.368, f"Stage {i}")

    # Arrows top row
    arrow(ax, (0.260, 0.782), (0.300, 0.782))
    arrow(ax, (0.490, 0.782), (0.535, 0.782))
    arrow(ax, (0.740, 0.782), (0.785, 0.782))

    # Main process down and across
    arrow(ax, (0.158, 0.695), (0.158, 0.610))

    arrow(ax, (0.260, 0.522), (0.300, 0.522))
    arrow(ax, (0.490, 0.522), (0.535, 0.522))
    arrow(ax, (0.740, 0.522), (0.785, 0.522))

    # Cross-row dependencies
    arrow(ax, (0.638, 0.695), (0.158, 0.610), rad=0.08, color="#64748B", lw=1.0)
    arrow(ax, (0.395, 0.695), (0.158, 0.610), rad=0.02, color="#64748B", lw=1.0)
    arrow(ax, (0.395, 0.435), (0.158, 0.360), rad=0.05, color="#64748B", lw=1.0)
    arrow(ax, (0.638, 0.435), (0.395, 0.360), rad=-0.05, color="#64748B", lw=1.0)
    arrow(ax, (0.870, 0.435), (0.745, 0.360), rad=-0.08, color="#64748B", lw=1.0)

    # Bottom row arrows
    arrow(ax, (0.260, 0.275), (0.300, 0.275))
    arrow(ax, (0.490, 0.275), (0.535, 0.275))

    # Scenario pill
    pill = FancyBboxPatch((0.318, 0.620), 0.152, 0.034,
                          boxstyle="round,pad=0.006,rounding_size=0.010",
                          linewidth=0.9, edgecolor="#6D5BD0", facecolor="#FFFFFF", zorder=5)
    ax.add_patch(pill)
    ax.text(0.394, 0.637, "SSP2-4.5 vs SSP5-8.5", ha="center", va="center",
            fontsize=7.7, fontweight="bold", color="#4C3FB3", zorder=6)

    ax.text(0.5, 0.065,
            "KGE = Kling-Gupta Efficiency; MK = Mann-Kendall; PW-MK = pre-whitened Mann-Kendall; FDR = false discovery rate.",
            ha="center", va="center", fontsize=8.2, color=COLORS["muted"])

    png = OUT / "MANUSCRIPT_Figure_2_workflow_chart.png"
    pdf = OUT / "MANUSCRIPT_Figure_2_workflow_chart.pdf"
    svg = OUT / "MANUSCRIPT_Figure_2_workflow_chart.svg"
    fig.savefig(png, bbox_inches="tight", dpi=450)
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(png)
    print(pdf)
    print(svg)


if __name__ == "__main__":
    main()
