#!/usr/bin/env python3
"""Fetch latest Antarctic GFS 2 m temps and write a polar overview PNG."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from antarctica_temp_map.fetch_gfs import (
    download_icec_for_cycle,
    download_land_for_cycle,
    download_latest_antarctic_t2m,
    open_icec_fraction,
    open_land_fraction,
    open_t2m_fahrenheit,
)
from antarctica_temp_map.render import (
    ICE_EDGE_FRACTION,
    build_overview_frame,
    render_frame,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data"
DEFAULT_OUTPUT = ROOT / "output"


@dataclass(frozen=True)
class SnapshotInputs:
    grib_path: Path
    cycle_time: datetime
    forecast_hour: int
    valid_time: datetime
    ice_path: Path | None
    land_path: Path | None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = _resolve_inputs(args)

    print("Opening GRIB and converting to °F…")
    t2m = open_t2m_fahrenheit(inputs.grib_path)
    print(
        f"Domain °F  min={float(t2m.min()):.1f}  "
        f"mean={float(t2m.mean()):.1f}  max={float(t2m.max()):.1f}"
    )

    icec, land = _open_overlays(inputs.ice_path, inputs.land_path)

    stamp = inputs.valid_time.strftime("%Y%m%dT%H%MZ")
    frame, central_lon = build_overview_frame()
    out = args.output_dir / f"antarctica_t2m_{frame.name}_{stamp}.png"
    print(f"Rendering {frame.name} → {out.name}")
    render_frame(
        t2m,
        frame,
        valid_time=inputs.valid_time,
        cycle_time=inputs.cycle_time,
        forecast_hour=inputs.forecast_hour,
        out_path=out,
        central_longitude=central_lon,
        icec=icec,
        land=land,
    )

    print("Done.")
    print(f"  {out}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download NOAA GFS 0.25° 2 m temperature over Antarctica and render "
            "a high-resolution polar stereographic overview with sea-ice edge."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA,
        help="Directory for downloaded GRIB files (default: ./data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for PNG snapshots (default: ./output)",
    )
    parser.add_argument(
        "--grib",
        type=Path,
        default=None,
        help="Use an existing temperature GRIB instead of downloading",
    )
    return parser.parse_args(argv)


def _resolve_inputs(args: argparse.Namespace) -> SnapshotInputs:
    """Download the latest cycle, or reuse `--grib` and matching overlays."""
    if args.grib is not None:
        grib_path = args.grib
        parsed = _parse_grib_name(grib_path)
        if parsed is not None:
            cycle_time, forecast_hour = parsed
            valid_time = cycle_time + timedelta(hours=forecast_hour)
            ice_path = download_icec_for_cycle(
                args.data_dir, cycle_time, forecast_hour
            )
            land_path = download_land_for_cycle(
                args.data_dir, cycle_time, forecast_hour
            )
        else:
            cycle_time = datetime.fromtimestamp(
                grib_path.stat().st_mtime, tz=timezone.utc
            )
            forecast_hour = 0
            valid_time = cycle_time
            ice_path = None
            land_path = None
        print(f"Using existing GRIB: {grib_path}")
        return SnapshotInputs(
            grib_path=grib_path,
            cycle_time=cycle_time,
            forecast_hour=forecast_hour,
            valid_time=valid_time,
            ice_path=ice_path,
            land_path=land_path,
        )

    print("Discovering latest NOAA GFS Antarctic 2 m temperature…")
    snapshot = download_latest_antarctic_t2m(args.data_dir)
    print(
        f"Downloaded {snapshot.path.name}  "
        f"(cycle {snapshot.cycle_time.strftime('%Y-%m-%d %HZ')}, "
        f"f{snapshot.forecast_hour:03d}, "
        f"valid {snapshot.valid_time.strftime('%Y-%m-%d %H:%MZ')})"
    )
    return SnapshotInputs(
        grib_path=snapshot.path,
        cycle_time=snapshot.cycle_time,
        forecast_hour=snapshot.forecast_hour,
        valid_time=snapshot.valid_time,
        ice_path=snapshot.ice_path,
        land_path=snapshot.land_path,
    )


def _open_overlays(
    ice_path: Path | None,
    land_path: Path | None,
):
    """Load sea-ice and land masks used by the renderer, if present."""
    icec = None
    land = None
    if ice_path is not None:
        print(f"Opening sea-ice concentration: {ice_path.name}")
        icec = open_icec_fraction(ice_path)
        covered = float(
            np.mean(np.asarray(icec.values, dtype=float) >= ICE_EDGE_FRACTION)
        )
        pct = int(ICE_EDGE_FRACTION * 100)
        print(f"Sea-ice extent (≥{pct}%): {covered:.1%} of domain cells")
    else:
        print("Sea-ice concentration unavailable for this cycle; skipping ice edge.")
    if land_path is not None:
        land = open_land_fraction(land_path)
    return icec, land


def _parse_grib_name(path: Path) -> tuple[datetime, int] | None:
    """Parse cycle time and forecast hour from a `gfs_t2m_*` filename."""
    match = re.search(r"gfs_t2m_(\d{8})_t(\d{2})z_f(\d{3})", path.name)
    if not match:
        return None
    cycle_time = datetime.strptime(
        f"{match.group(1)}{match.group(2)}",
        "%Y%m%d%H",
    ).replace(tzinfo=timezone.utc)
    return cycle_time, int(match.group(3))


if __name__ == "__main__":
    sys.exit(main())
