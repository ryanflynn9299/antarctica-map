"""Fetch NOAA GFS 0.25° fields for the Antarctic domain."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import gribberish
import numpy as np
import requests
import xarray as xr

NOMADS_FILTER = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
# Antarctic domain for NOMADS subregion requests (full longitude).
TOP_LAT = -50.0
BOTTOM_LAT = -90.0
# Short-range forecast hours to try when discovering the newest available grid.
FORECAST_HOURS = (0, 1, 3, 6)
USER_AGENT = "antarctica-temp-map/0.1 (research snapshot tool)"


@dataclass(frozen=True)
class GfsSnapshot:
    path: Path
    cycle_time: datetime
    forecast_hour: int
    valid_time: datetime
    ice_path: Path | None = None
    land_path: Path | None = None


def download_latest_antarctic_t2m(
    dest_dir: Path,
    *,
    timeout: float = 120.0,
    session: requests.Session | None = None,
) -> GfsSnapshot:
    """
    Discover the newest available GFS cycle/forecast and download Antarctic
    2 m temperature, plus matching ICEC and LAND fields when available.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    http = session or requests.Session()

    last_error: Exception | None = None
    for cycle in _candidate_cycles():
        for fhour in FORECAST_HOURS:
            url = _build_url(
                cycle,
                fhour,
                var="var_TMP",
                level="lev_2_m_above_ground",
            )
            out = _grib_path(dest_dir, "t2m", cycle, fhour)
            try:
                if not _download_grib(url, out, http=http, timeout=timeout):
                    continue
                return GfsSnapshot(
                    path=out,
                    cycle_time=cycle,
                    forecast_hour=fhour,
                    valid_time=cycle + timedelta(hours=fhour),
                    ice_path=download_icec_for_cycle(
                        dest_dir,
                        cycle,
                        fhour,
                        timeout=timeout,
                        session=http,
                    ),
                    land_path=download_land_for_cycle(
                        dest_dir,
                        cycle,
                        fhour,
                        timeout=timeout,
                        session=http,
                    ),
                )
            except (requests.RequestException, OSError, ValueError) as exc:
                last_error = exc
                if out.exists():
                    out.unlink(missing_ok=True)
                continue

    detail = f" Last error: {last_error}" if last_error else ""
    raise RuntimeError(
        "Could not download Antarctic GFS 2 m temperature from NOMADS." + detail
    )


def download_icec_for_cycle(
    dest_dir: Path,
    cycle: datetime,
    forecast_hour: int,
    *,
    timeout: float = 120.0,
    session: requests.Session | None = None,
) -> Path | None:
    """Download GFS surface ice concentration for a known cycle."""
    return download_surface_field_for_cycle(
        dest_dir,
        cycle,
        forecast_hour,
        var="var_ICEC",
        stem="icec",
        timeout=timeout,
        session=session,
    )


def download_land_for_cycle(
    dest_dir: Path,
    cycle: datetime,
    forecast_hour: int,
    *,
    timeout: float = 120.0,
    session: requests.Session | None = None,
) -> Path | None:
    """Download GFS land-sea mask for a known cycle."""
    return download_surface_field_for_cycle(
        dest_dir,
        cycle,
        forecast_hour,
        var="var_LAND",
        stem="land",
        timeout=timeout,
        session=session,
    )


def download_surface_field_for_cycle(
    dest_dir: Path,
    cycle: datetime,
    forecast_hour: int,
    *,
    var: str,
    stem: str,
    timeout: float = 120.0,
    session: requests.Session | None = None,
) -> Path | None:
    """Download a GFS surface field for a known cycle, or None on failure."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    http = session or requests.Session()
    out = _grib_path(dest_dir, stem, cycle, forecast_hour)
    if out.exists() and out.stat().st_size > 5_000:
        return out
    url = _build_url(cycle, forecast_hour, var=var, level="lev_surface")
    try:
        if _download_grib(url, out, http=http, timeout=timeout):
            return out
    except (requests.RequestException, OSError, ValueError):
        if out.exists():
            out.unlink(missing_ok=True)
    return None


def open_t2m_fahrenheit(path: Path) -> xr.DataArray:
    """Open a GFS TMP:2m GRIB and return Fahrenheit on a lat/lon grid."""
    t2m_k = _open_latlon_field(path, name="t2m")
    t2m_f = _kelvin_to_fahrenheit(t2m_k)
    t2m_f.attrs.update(
        {
            "units": "degF",
            "long_name": "2 metre temperature",
            "standard_name": "air_temperature",
        }
    )
    return t2m_f.load()


def open_icec_fraction(path: Path) -> xr.DataArray:
    """Open GFS surface ice concentration (0–1 fraction)."""
    ice = _open_latlon_field(path, name="icec")
    # GFS ICEC is typically 0–1; some products encode percent instead.
    if float(np.nanmax(ice.values)) > 1.5:
        ice = ice / 100.0
    ice.attrs.update(
        {
            "units": "1",
            "long_name": "sea ice concentration",
            "standard_name": "sea_ice_area_fraction",
        }
    )
    return ice.load()


def open_land_fraction(path: Path) -> xr.DataArray:
    """Open GFS land-sea mask (1 = land, 0 = ocean)."""
    land = _open_latlon_field(path, name="land")
    land.attrs.update(
        {
            "units": "1",
            "long_name": "land-sea mask",
            "standard_name": "land_binary_mask",
        }
    )
    return land.load()


def _grib_path(dest_dir: Path, stem: str, cycle: datetime, forecast_hour: int) -> Path:
    """Local cache path for a downloaded GFS field."""
    return dest_dir / (
        f"gfs_{stem}_{cycle.strftime('%Y%m%d')}_t{cycle.hour:02d}z_"
        f"f{forecast_hour:03d}.grib2"
    )


def _candidate_cycles(now: datetime | None = None) -> list[datetime]:
    """
    Recent GFS cycles, newest first (UTC).

    GFS runs at 00/06/12/18Z; NOMADS often lags a few hours after cycle time, so
    the search starts from the previous 6-hour boundary offset by 3 hours.
    """
    now = now or datetime.now(timezone.utc)
    anchor = now - timedelta(hours=3)
    hour = (anchor.hour // 6) * 6
    latest = anchor.replace(hour=hour, minute=0, second=0, microsecond=0)
    return [latest - timedelta(hours=6 * i) for i in range(12)]


def _build_url(
    cycle: datetime,
    forecast_hour: int,
    *,
    var: str,
    level: str,
) -> str:
    """Build a NOMADS filter URL for one GFS 0.25° field over the Antarctic box."""
    ymd = cycle.strftime("%Y%m%d")
    hh = f"{cycle.hour:02d}"
    fxx = f"{forecast_hour:03d}"
    file_name = f"gfs.t{hh}z.pgrb2.0p25.f{fxx}"
    directory = f"/gfs.{ymd}/{hh}/atmos"
    params = {
        "file": file_name,
        level: "on",
        var: "on",
        "subregion": "",
        "leftlon": "0",
        "rightlon": "360",
        "toplat": str(TOP_LAT),
        "bottomlat": str(BOTTOM_LAT),
        "dir": directory,
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{NOMADS_FILTER}?{query}"


def _download_grib(
    url: str,
    out: Path,
    *,
    http: requests.Session,
    timeout: float,
) -> bool:
    """
    Download a GRIB to `out`.

    Returns False for missing/unavailable products (non-200, HTML error pages,
    or tiny payloads). Raises on truncated HTML disguised as a download.
    """
    headers = {"User-Agent": USER_AGENT}
    with http.get(url, stream=True, timeout=timeout, headers=headers) as resp:
        if resp.status_code != 200:
            return False
        content_type = (resp.headers.get("Content-Type") or "").lower()
        if "html" in content_type or "text/" in content_type:
            return False
        chunks: list[bytes] = []
        size = 0
        for chunk in resp.iter_content(chunk_size=1 << 16):
            if not chunk:
                continue
            chunks.append(chunk)
            size += len(chunk)
            if size < 512 and b"<html" in b"".join(chunks).lower():
                raise ValueError("NOMADS returned HTML instead of GRIB")
        if size < 10_000:
            return False
        out.write_bytes(b"".join(chunks))
    return True


def _open_latlon_field(path: Path, *, name: str) -> xr.DataArray:
    """
    Decode a single-message GFS GRIB into a lat/lon DataArray via gribberish.

    Arrays are forced south→north so pcolormesh and contour agree on orientation.
    """
    raw = Path(path).read_bytes()
    if raw[:4] != b"GRIB":
        raise ValueError(f"Not a GRIB file: {path}")

    dataset = gribberish.parse_grib_dataset(raw)
    instant = None
    for group in dataset.get("groups", {}).values():
        subgroups = group.get("groups", {})
        if "instant" in subgroups:
            instant = subgroups["instant"]
            break
    if instant is None:
        raise ValueError(f"Unexpected gribberish dataset structure in {path}")

    lats = np.asarray(instant["coords"]["latitude"]["values"], dtype=float)
    lons = np.asarray(instant["coords"]["longitude"]["values"], dtype=float)
    values = np.asarray(
        gribberish.parse_grib_array(raw, 0, north_up=False),
        dtype=float,
    ).reshape(lats.size, lons.size)

    if lats[0] > lats[-1]:
        lats = lats[::-1]
        values = values[::-1]

    return xr.DataArray(
        values,
        coords={"latitude": lats, "longitude": lons},
        dims=("latitude", "longitude"),
        name=name,
    )


def _kelvin_to_fahrenheit(kelvin: xr.DataArray) -> xr.DataArray:
    return (kelvin - 273.15) * (9.0 / 5.0) + 32.0
