"""South Polar Stereographic rendering for Antarctic temperature snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.path import Path as MplPath
from shapely.geometry import box

from antarctica_temp_map.colormap import COLORMAP, NORM, colorbar_ticks

# Landmark sites for orientation on the overview (name, lat, lon).
LANDMARKS: list[tuple[str, float, float]] = [
    ("South Pole", -90.0, 0.0),
    ("Vostok", -78.464, 106.837),
    ("Dome A", -80.367, 77.367),
    ("Dome C", -75.100, 123.350),
]

# Large canvas so the overview stays sharp when zoomed.
FIGSIZE = (16.0, 16.5)
DPI = 360
# Upsample the 0.25° GFS grid before fill so cells read more continuously.
SMOOTH_ZOOM = 4
# NSIDC-style sea-ice extent threshold (15% concentration).
ICE_EDGE_FRACTION = 0.15
# Midway between pack-ice grey (#c4c8d0) and continent outline (#4a4f59).
ICE_EDGE_COLOR = "#878c95"
ICE_EDGE_WIDTH = 1.6


@dataclass(frozen=True)
class FrameSpec:
    name: str
    title: str
    circle_lat: float
    show_landmarks: bool = True


def build_overview_frame() -> tuple[FrameSpec, float]:
    """Return the overview `FrameSpec` and central longitude."""
    return (
        FrameSpec(
            name="overview",
            title="Antarctica · 2 m Temperature",
            circle_lat=-60.0,
            show_landmarks=True,
        ),
        0.0,
    )


def render_frame(
    t2m: xr.DataArray,
    frame: FrameSpec,
    *,
    valid_time: datetime,
    cycle_time: datetime,
    forecast_hour: int,
    out_path: Path,
    central_longitude: float = 0.0,
    icec: xr.DataArray | None = None,
    land: xr.DataArray | None = None,
) -> Path:
    """Render the polar stereographic temperature overview PNG."""
    lats, lons, data = _smooth_temperature_field(
        t2m["latitude"].values,
        t2m["longitude"].values,
        t2m.values,
    )

    proj = ccrs.SouthPolarStereo(central_longitude=central_longitude)
    fig = plt.figure(figsize=FIGSIZE, facecolor="#070b14")
    ax = fig.add_axes([0.0375, 0.105, 0.925, 0.7975], projection=proj)
    ax.set_facecolor("#070b14")

    mesh = ax.pcolormesh(
        lons,
        lats,
        data,
        transform=ccrs.PlateCarree(),
        cmap=COLORMAP,
        norm=NORM,
        shading="auto",
        zorder=1,
    )

    if icec is not None:
        _add_sea_ice_edge(ax, icec, land)

    _add_antarctic_coastline(ax)
    _set_circular_boundary(ax, frame.circle_lat)
    ax.spines["geo"].set_visible(False)
    _add_gridlines(ax)

    if frame.show_landmarks:
        _add_landmarks(ax)

    _add_colorbar(fig, mesh)
    _annotate_figure(
        fig,
        frame,
        t2m,
        valid_time=valid_time,
        cycle_time=cycle_time,
        forecast_hour=forecast_hour,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return out_path


def _smooth_temperature_field(
    lats: np.ndarray,
    lons: np.ndarray,
    data: np.ndarray,
    *,
    factor: int = SMOOTH_ZOOM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Upsample the discrete GFS grid so the filled field reads more continuously
    without changing the fixed color scale.
    """
    if factor <= 1:
        return lats, lons, data

    fine = _light_blur(_upsample_bilinear(data, factor))
    lats_fine = np.linspace(float(lats[0]), float(lats[-1]), fine.shape[0])
    lons_fine = np.linspace(float(lons[0]), float(lons[-1]), fine.shape[1])
    return lats_fine, lons_fine, fine


def _upsample_bilinear(data: np.ndarray, factor: int) -> np.ndarray:
    """Bilinear upsample on a regular lat/lon grid (numpy-only)."""
    src = np.asarray(data, dtype=float)
    nlat, nlon = src.shape
    lat_dst = np.linspace(0.0, nlat - 1, nlat * factor)
    lon_dst = np.linspace(0.0, nlon - 1, nlon * factor)

    i0 = np.floor(lat_dst).astype(int)
    j0 = np.floor(lon_dst).astype(int)
    i1 = np.minimum(i0 + 1, nlat - 1)
    j1 = np.minimum(j0 + 1, nlon - 1)
    di = (lat_dst - i0)[:, None]
    dj = (lon_dst - j0)[None, :]

    v00 = src[i0][:, j0]
    v01 = src[i0][:, j1]
    v10 = src[i1][:, j0]
    v11 = src[i1][:, j1]
    return (
        v00 * (1.0 - di) * (1.0 - dj)
        + v01 * (1.0 - di) * dj
        + v10 * di * (1.0 - dj)
        + v11 * di * dj
    )


def _light_blur(data: np.ndarray) -> np.ndarray:
    """3×3 mean filter to soften blockiness after upsampling."""
    padded = np.pad(data, 1, mode="edge")
    return (
        padded[0:-2, 0:-2]
        + padded[0:-2, 1:-1]
        + padded[0:-2, 2:]
        + padded[1:-1, 0:-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, 0:-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 9.0


def _add_sea_ice_edge(
    ax,
    icec: xr.DataArray,
    land: xr.DataArray | None,
) -> None:
    """Upsample ICEC, mask land, and draw the outer 15% pack-ice edge."""
    ice_lats, ice_lons, ice_data = _smooth_temperature_field(
        icec["latitude"].values,
        icec["longitude"].values,
        icec.values,
    )
    if land is not None:
        # Mask continent so coastal land/ocean transitions are not contoured.
        land_fine = _upsample_nearest(land.values, SMOOTH_ZOOM)
        ice_data = np.array(ice_data, dtype=float, copy=True)
        ice_data[land_fine > 0.5] = np.nan
    _draw_pack_ice_edge(ax, ice_lons, ice_lats, ice_data)


def _upsample_nearest(data: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbor upsample for categorical masks (e.g. land/sea)."""
    src = np.asarray(data, dtype=float)
    nlat, nlon = src.shape
    lat_idx = np.linspace(0.0, nlat - 1, nlat * factor)
    lon_idx = np.linspace(0.0, nlon - 1, nlon * factor)
    i = np.clip(np.rint(lat_idx).astype(int), 0, nlat - 1)
    j = np.clip(np.rint(lon_idx).astype(int), 0, nlon - 1)
    return src[i][:, j]


def _draw_pack_ice_edge(
    ax, ice_lons: np.ndarray, ice_lats: np.ndarray, ice_data: np.ndarray
) -> None:
    """
    Draw the outer 15% pack-ice edge only.

    A raw 0.15 contour also traces coastal/ice-shelf gradients. Keep segments that
    reach the open Southern Ocean (north of ~−65°S) — the true extent boundary.
    """
    # Close the longitude ring so the contour does not break at 0°/360°.
    lon_pad = np.concatenate([ice_lons, ice_lons[:1] + 360.0])
    ice_pad = np.concatenate([ice_data, ice_data[:, :1]], axis=1)

    fig_tmp, ax_tmp = plt.subplots()
    cs_tmp = ax_tmp.contour(lon_pad, ice_lats, ice_pad, levels=[ICE_EDGE_FRACTION])
    outer_segs = [
        seg
        for seg in cs_tmp.allsegs[0]
        if len(seg) >= 8 and float(np.nanmax(seg[:, 1])) >= -65.0
    ]
    plt.close(fig_tmp)

    for seg in outer_segs:
        ax.plot(
            seg[:, 0],
            seg[:, 1],
            color=ICE_EDGE_COLOR,
            linewidth=ICE_EDGE_WIDTH,
            transform=ccrs.PlateCarree(),
            zorder=4.5,
            solid_capstyle="round",
        )


def _add_antarctic_coastline(ax) -> None:
    """
    Draw Natural Earth coastlines clipped to the Antarctic domain.

    Without the clip, tips of South America / Africa appear on the map rim.
    """
    path = shpreader.natural_earth(
        resolution="50m",
        category="physical",
        name="coastline",
    )
    clip = box(-180.0, -90.0, 180.0, -60.0)
    geoms = []
    for geom in shpreader.Reader(path).geometries():
        clipped = geom.intersection(clip)
        if not clipped.is_empty:
            geoms.append(clipped)
    if geoms:
        ax.add_geometries(
            geoms,
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="#4a4f59",
            linewidth=4.0,
            zorder=5,
        )


def _set_circular_boundary(ax, lat_edge: float) -> None:
    """Clip the polar axes to a circle whose rim is `lat_edge` (e.g. −60)."""
    theta = np.linspace(0, 2 * np.pi, 361)
    center = ax.projection.transform_point(0.0, -90.0, ccrs.PlateCarree())
    rim_pt = ax.projection.transform_point(0.0, lat_edge, ccrs.PlateCarree())
    radius = float(np.hypot(rim_pt[0] - center[0], rim_pt[1] - center[1]))
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T * radius + np.asarray(center)
    verts = np.vstack([verts, verts[:1]])
    codes = np.full(len(verts), MplPath.LINETO, dtype=np.uint8)
    codes[0] = MplPath.MOVETO
    codes[-1] = MplPath.CLOSEPOLY
    circle = MplPath(verts, codes)
    ax.set_boundary(circle, transform=ax.transData)
    ax.set_extent([-180, 180, -90, lat_edge], crs=ccrs.PlateCarree())
    ax.set_xlim(center[0] - radius * 1.02, center[0] + radius * 1.02)
    ax.set_ylim(center[1] - radius * 1.02, center[1] + radius * 1.02)


def _add_gridlines(ax) -> None:
    gl = ax.gridlines(
        draw_labels=False,
        linewidth=0.4,
        color=(1, 1, 1, 0.18),
        linestyle="--",
        zorder=3,
    )
    gl.xlocator = plt.FixedLocator(np.arange(-180, 181, 30))
    gl.ylocator = plt.FixedLocator([-80, -70, -60, -50])


def _add_landmarks(ax) -> None:
    for name, lat, lon in LANDMARKS:
        ax.plot(
            lon,
            lat,
            marker="o",
            markersize=4.0,
            color="white",
            markeredgecolor="#0a0e18",
            markeredgewidth=0.7,
            transform=ccrs.PlateCarree(),
            zorder=6,
        )
        # Nudge labels off the marker; leave the Pole label on the point.
        label_lat = lat + (1.4 if lat > -89 else 0.0)
        label_lon = lon + (4.0 if abs(lat + 90) > 0.5 else 0.0)
        ax.text(
            label_lon,
            label_lat,
            name,
            transform=ccrs.PlateCarree(),
            fontsize=9,
            color="#0a0e18",
            ha="left",
            va="bottom",
            zorder=7,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": (1, 1, 1, 0.62),
                "edgecolor": "none",
            },
        )


def _add_colorbar(fig, mesh) -> None:
    cax = fig.add_axes([0.14, 0.0515, 0.72, 0.023])
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="both")
    cbar.set_ticks(colorbar_ticks())
    cbar.ax.tick_params(labelsize=9, colors="#e8eef8")
    cbar.outline.set_edgecolor("#8a93a8")
    cbar.set_label("2 m air temperature (°F)", color="#e8eef8", fontsize=10, labelpad=7)
    for spine in cbar.ax.spines.values():
        spine.set_edgecolor("#8a93a8")


def _annotate_figure(
    fig,
    frame: FrameSpec,
    t2m: xr.DataArray,
    *,
    valid_time: datetime,
    cycle_time: datetime,
    forecast_hour: int,
) -> None:
    """Title, valid/cycle stamp with domain stats, and footer attribution."""
    stats = _stats_south_of(t2m, lat_max=-60.0)
    valid_str = valid_time.strftime("%Y-%m-%d %H:%MZ")
    cycle_str = cycle_time.strftime("%Y-%m-%d %HZ")
    fig.text(
        0.05,
        0.960,
        frame.title,
        color="#f4f7fb",
        fontsize=20,
        fontweight="bold",
        fontfamily="sans-serif",
    )
    fig.text(
        0.05,
        0.9315,
        (
            f"Valid {valid_str}  ·  GFS cycle {cycle_str}  f{forecast_hour:03d}  ·  "
            f"min {_format_temp(stats['min'])}  mean {_format_temp(stats['mean'])}  "
            f"max {_format_temp(stats['max'])}"
        ),
        color="#b7c0d0",
        fontsize=10,
    )
    fig.text(
        0.05,
        0.015,
        "NOAA GFS 0.25°  ·  linear −140°F→+40°F  ·  light grey = 15% sea-ice edge",
        color="#8a93a8",
        fontsize=8.5,
    )


def _stats_south_of(t2m: xr.DataArray, lat_max: float = -60.0) -> dict[str, float]:
    """Min/mean/max over cells at or south of `lat_max`."""
    subset = t2m.where(t2m["latitude"] <= lat_max)
    return {
        "min": float(subset.min()),
        "max": float(subset.max()),
        "mean": float(subset.mean()),
    }


def _format_temp(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.0f}°F"
