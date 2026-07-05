from pathlib import Path
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pyproj import Geod


ROOT = Path(__file__).resolve().parents[1]
COUNTRIES_SHP = ROOT / "shp" / "SAsiaFinal.shp"
DISSOLVED_SHP = ROOT / "shp" / "SAsiaFinalD.shp"
OUT = ROOT / "output" / "study_area_map"
OUT.mkdir(parents=True, exist_ok=True)


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.linewidth": 0.8,
    "savefig.dpi": 450,
})


COLORS = {
    "ocean": "#DCEFF7",
    "land": "#F4F1E8",
    "neighbor": "#DDD8CC",
    "boundary": "#8B1E3F",
    "country_edge": "#253245",
    "grid": "#8DA4B8",
    "text": "#17202A",
}


COUNTRY_COLORS = {
    "Afghanistan": "#E9C46A",
    "Pakistan": "#8AB17D",
    "India": "#F4A261",
    "Nepal": "#A8DADC",
    "Bhutan": "#B8E0D2",
    "Bangladesh": "#90BE6D",
    "Sri Lanka": "#CDB4DB",
    "Maldives": "#80CED7",
}


def add_north_arrow(ax, x=0.935, y=0.865):
    ax.annotate(
        "N", xy=(x, y + 0.075), xytext=(x, y),
        xycoords="axes fraction", textcoords="axes fraction",
        ha="center", va="center", fontsize=14, fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", lw=1.8, color=COLORS["text"]),
        color=COLORS["text"], zorder=10,
    )


def add_scale_bar(ax, lon0=62.0, lat0=2.0, length_km=1000):
    geod = Geod(ellps="WGS84")
    lon1, lat1, _ = geod.fwd(lon0, lat0, 90, length_km * 1000)
    trans = ccrs.PlateCarree()
    ax.plot([lon0, lon1], [lat0, lat1], transform=trans, color="black", lw=3.0, zorder=9)
    tick = 0.45
    for lon in [lon0, lon1]:
        ax.plot([lon, lon], [lat0 - tick, lat0 + tick], transform=trans, color="black", lw=2.0, zorder=9)
    mid_lon, mid_lat, _ = geod.fwd(lon0, lat0, 90, length_km * 500)
    ax.text(mid_lon, lat0 - 1.05, f"{length_km:,} km", transform=trans,
            ha="center", va="top", fontsize=9.5, fontweight="bold", color="black", zorder=9)


def label_country(ax, row, dx=0, dy=0, size=8.6):
    point = row.geometry.representative_point()
    ax.text(
        point.x + dx, point.y + dy, row["NAME"],
        transform=ccrs.PlateCarree(),
        ha="center", va="center",
        fontsize=size, fontweight="bold",
        color="#1F2937",
        zorder=8,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.72),
    )


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    countries = gpd.read_file(COUNTRIES_SHP).to_crs("EPSG:4326")
    dissolved = gpd.read_file(DISSOLVED_SHP).to_crs("EPSG:4326")
    minx, miny, maxx, maxy = dissolved.total_bounds

    fig = plt.figure(figsize=(10.8, 8.2), facecolor="white")
    proj = ccrs.PlateCarree()
    ax = fig.add_axes([0.055, 0.075, 0.89, 0.825], projection=proj)
    ax.set_extent([58, 100, -3, 41], crs=proj)
    ax.set_facecolor(COLORS["ocean"])

    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=COLORS["land"], edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=COLORS["ocean"], edgecolor="none", zorder=0)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#8B8F95", linewidth=0.45, zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#67717E", linewidth=0.55, zorder=2)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor=COLORS["ocean"], edgecolor="#A7BBC8", linewidth=0.35, zorder=1)

    for _, row in countries.iterrows():
        name = row["NAME"]
        ax.add_geometries(
            [row.geometry],
            crs=proj,
            facecolor=COUNTRY_COLORS.get(name, "#BFD7EA"),
            edgecolor=COLORS["country_edge"],
            linewidth=0.75,
            alpha=0.92,
            zorder=4,
        )

    ax.add_geometries(
        dissolved.geometry,
        crs=proj,
        facecolor="none",
        edgecolor=COLORS["boundary"],
        linewidth=2.0,
        zorder=6,
    )

    label_offsets = {
        "Afghanistan": (0.0, 0.0, 8.2),
        "Pakistan": (-0.3, 0.0, 8.6),
        "India": (0.5, -0.2, 9.2),
        "Nepal": (0.0, 0.7, 7.6),
        "Bhutan": (0.4, 0.35, 7.2),
        "Bangladesh": (0.7, -0.25, 7.4),
        "Sri Lanka": (0.0, -0.25, 7.4),
        "Maldives": (1.9, 1.5, 7.0),
    }
    for _, row in countries.iterrows():
        dx, dy, size = label_offsets.get(row["NAME"], (0, 0, 8))
        label_country(ax, row, dx, dy, size)

    gl = ax.gridlines(
        crs=proj, draw_labels=True,
        linewidth=0.45, color=COLORS["grid"], alpha=0.65, linestyle="--",
        xlocs=[60, 70, 80, 90, 100],
        ylocs=[0, 10, 20, 30, 40],
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8.5, "fontweight": "bold", "color": "#263446"}
    gl.ylabel_style = {"size": 8.5, "fontweight": "bold", "color": "#263446"}

    add_north_arrow(ax, x=0.955, y=0.535)
    add_scale_bar(ax)

    ax.text(
        0.02, 0.975, "Study Area: South Asia",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=16, fontweight="bold", color=COLORS["text"],
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CBD5E1", alpha=0.92),
        zorder=12,
    )
    ax.text(
        0.02, 0.918,
        "Countries included: Afghanistan, Pakistan, India, Nepal, Bhutan,\nBangladesh, Sri Lanka, and Maldives",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=8.6, color="#334155",
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#E2E8F0", alpha=0.90),
        zorder=12,
    )

    # World inset
    inset = fig.add_axes([0.655, 0.625, 0.255, 0.225], projection=proj)
    inset.set_global()
    inset.set_facecolor(COLORS["ocean"])
    inset.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#E6E1D8", edgecolor="#B5AEA2", linewidth=0.25, zorder=0)
    inset.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor=COLORS["ocean"], edgecolor="none", zorder=0)
    inset.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="#888", linewidth=0.25, zorder=1)
    inset.add_geometries(dissolved.geometry, crs=proj, facecolor="#D94E5D", edgecolor="#7A1230", linewidth=0.7, zorder=3)
    inset.add_patch(Rectangle((minx, miny), maxx - minx, maxy - miny,
                              transform=proj, fill=False, edgecolor="#7A1230", linewidth=1.1, zorder=4))
    inset.set_extent([-180, 180, -60, 85], crs=proj)
    inset.set_title("Location in the world", fontsize=8.5, fontweight="bold", pad=3)
    for spine in inset.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#334155")

    # Legend-like note
    ax.plot([], [], color=COLORS["boundary"], lw=2.2, label="Dissolved study-area boundary")
    leg = ax.legend(loc="lower right", frameon=True, framealpha=0.94, fontsize=8.5)
    leg.get_frame().set_edgecolor("#CBD5E1")

    fig.text(
        0.5, 0.026,
        "Projection: geographic coordinates (WGS84). Country polygons are from the project shapefile; dissolved boundary highlights the final study area.",
        ha="center", va="center", fontsize=8.2, color="#475569",
    )

    png = OUT / "MANUSCRIPT_Figure_1_study_area_map.png"
    pdf = OUT / "MANUSCRIPT_Figure_1_study_area_map.pdf"
    svg = OUT / "MANUSCRIPT_Figure_1_study_area_map.svg"
    tif = OUT / "MANUSCRIPT_Figure_1_study_area_map.tif"
    fig.savefig(png, bbox_inches="tight", dpi=450)
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(tif, bbox_inches="tight", dpi=450)
    plt.close(fig)
    print(png)
    print(pdf)
    print(svg)
    print(tif)


if __name__ == "__main__":
    main()
