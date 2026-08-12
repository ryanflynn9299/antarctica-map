"""Linear Antarctic temperature colormap (−140°F → +40°F)."""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap, Normalize

TEMP_MIN_F = -140.0
TEMP_MAX_F = 40.0

# Fixed linear °F scale. Stops are ≤10°F apart so cold detail stays resolvable;
# purple→blue and blue→sky use slight off-line bisects so the hue path feels
# continuous. Deep blue holds through ~−20°F; cyan covers ~0…+10; mint is held
# flat from ~+37…+40 so the warm end does not keep climbing.


def _pos(temp_f: float) -> float:
    """Map °F onto [0, 1] within TEMP_MIN_F…TEMP_MAX_F."""
    return (temp_f - TEMP_MIN_F) / (TEMP_MAX_F - TEMP_MIN_F)


_COLOR_STOPS: list[tuple[float, str]] = [
    # Deep purple → purple
    (_pos(-140), "#1a1028"),
    (_pos(-130), "#1d122e"),
    (_pos(-120), "#1f1334"),
    (_pos(-110), "#281642"),
    (_pos(-100), "#321850"),
    (_pos(-90), "#3c2166"),
    (_pos(-80), "#442a7c"),
    # Purple → blue
    (_pos(-70), "#3f3486"),
    (_pos(-60), "#3a4894"),
    (_pos(-50), "#384c9a"),
    (_pos(-40), "#3650a0"),
    (_pos(-30), "#3656a5"),
    (_pos(-20), "#355caa"),
    # Sky bridge → cyan
    (_pos(-10), "#3c74b8"),
    (_pos(0), "#4a98c0"),
    (_pos(10), "#56acb0"),
    # Cyan → mint
    (_pos(16), "#5eb0b2"),
    (_pos(20), "#62b7ae"),
    (_pos(24), "#66bca8"),
    (_pos(30), "#6bbfaa"),
    (_pos(37), "#72c49a"),
    (_pos(40), "#72c49a"),
]


def build_colormap() -> LinearSegmentedColormap:
    """Build the project colormap from `_COLOR_STOPS`."""
    return LinearSegmentedColormap.from_list(
        "antarctica_blue_purple_mint",
        _COLOR_STOPS,
        N=1024,
    )


def colorbar_ticks() -> list[float]:
    """Even 20°F ticks across the fixed scale."""
    return [-140, -120, -100, -80, -60, -40, -20, 0, 20, 40]


COLORMAP = build_colormap()
NORM = Normalize(vmin=TEMP_MIN_F, vmax=TEMP_MAX_F, clip=False)
