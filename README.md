# Antarctica Temperature Map

A personal high-resolution polar view of Antarctic **2m air temperature** inspired by Apple's Weather app with enough color range to tell deep colds apart, rendered in continental view rather than along the bottom of Mercator-projection world maps.

![Antarctica 2 m temperature overview](overview.png)

## Why this exists

I’ve had a long fascination with Antarctica, especially after visiting in 2023. That curiosity spilled into Apple Weather: I’d drop pins into the darkest purple patches of the interior, hunting for the lowest reading I could find — a kind of cold “high score.”

That game breaks down in winter: once you’re deep cold, the app runs out of colors, so −60°F and −100°F look the same.

Worse, Apple shows the world on a Mercator-style map. Antarctica is a thin strip along the bottom.

This project is the map I wished I had:

- Colors that keep distinguishing extreme cold instead of crushing the interior into one blob
- Antarctica as a **continent** — South Polar Stereographic, not a flattened footer under the rest of the world
- Near-real-time temperatures you can compare day to day

## How it works

### Data Source

The temperature field comes from **NOAA/NCEP’s Global Forecast System (GFS)** at **0.25°** (~25 km), pulled through [NCEP NOMADS](https://nomads.ncep.noaa.gov/). I chose GFS because it covers the whole Antarctic domain at useful resolution, updates several times a day, and is free to download without an API key or account — good enough for a personal snapshot tool.

Each run uses three fields from the same cycle. Together they build the image:

| Field | GFS variable | Role on the map |
| --- | --- | --- |
| 2 m air temperature | `TMP` at 2 m | The temperature fill |
| Sea-ice concentration | `ICEC` (surface) | Outer **15%** pack-ice edge (NSIDC-style threshold) |
| Land–sea mask | `LAND` (surface) | Keeps ice contours off the continent |

The download is clipped to **50°S–90°S**, full longitude, so only the Antarctic domain is fetched.

### Data Freshness

GFS cycles every **6 hours** (00Z / 06Z / 12Z / 18Z). The tool walks recent cycles and short forecast hours (`f000`, `f001`, `f003`, `f006`) and takes the newest grid that NOMADS actually serves — usually within a few hours of “now.” Each PNG is stamped with **valid time**, **cycle**, and **forecast hour** so you know exactly what you’re looking at.

Re-run the script anytime for a fresh snapshot; previously downloaded GRIBs in `data/` are reused when present.

### About the render

1. Decode the GRIB lat/lon grids (temperature in Kelvin → °F).
2. Lightly upsample and smooth the temperature field so the coarse 0.25° cells read more continuously on a large canvas.
3. Project onto **South Polar Stereographic**, clipped to a circle near **60°S**.
4. Overlay Natural Earth coastlines (Antarctic domain only), the 15% sea-ice edge, gridlines, and a few landmarks (South Pole, Vostok, Dome A, Dome C).
5. Write a large overview PNG under `output/` (~5800×5800 px) meant for zooming and poking around.

The finished image also prints domain min / mean / max south of 60°S in the title bar. The color scale is a personal preference and intentionally omitted here — swap it in `antarctica_temp_map/colormap.py` if you want something else.

### Caveats

- This is a **model** field (GFS analysis/short-range forecast), not station observations. Interior Antarctica has sparse verification.
- At 0.25°, individual valleys and ice rises are below the grid scale. Good for continent-scale patterns, not pin-point microclimates.
- NOMADS can lag or miss a cycle; the downloader simply falls back to the next-newest available file.

## How to use this project

### Project layout

```text
generate_snapshots.py      # entry point
antarctica_temp_map/       # library (fetch, colormap, render)
data/                      # downloaded GRIBs (gitignored)
output/                    # PNG snapshots (gitignored)
```

### Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+. Network access is needed for the first download (and whenever you want a new cycle).

```bash
uv sync
```

GRIB decoding uses **gribberish** (no system eccodes install required). Cartopy pulls in the usual geospatial wheels on macOS/Linux; on first run it may also fetch Natural Earth coastline data.

### Generate a snapshot

```bash
uv run python generate_snapshots.py
```

Writes one PNG under `output/`:

- **overview** — circular South Polar Stereographic view clipped near 60°S (~5800×5800 px)

Typical filename: `antarctica_t2m_overview_YYYYMMDDTHHMMZ.png`.

Options:

```bash
# Re-render from a GRIB you already have
uv run python generate_snapshots.py --grib data/some_file.grib2

# Custom directories
uv run python generate_snapshots.py --output-dir output --data-dir data
```

## Data attribution

Temperature and ice fields are from the **NOAA/NCEP Global Forecast System (GFS)** 0.25° product, accessed through NOMADS. This project is not affiliated with NOAA.
