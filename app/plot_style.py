#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared figure styling: font sizes and fallback station colours.

Presentation only - nothing here changes data, calculations or layout.
Figures decorated with font_sizes_plus(N) use FONT_SIZES + N points.
"""

from functools import wraps

import matplotlib.pyplot as plt

# Font sizes in points.
FONT_SIZES = {
    "title": 12,
    "axis_label": 11,  # axis labels and colorbar label
    "tick_label": 11,  # axis and colorbar tick labels
    "legend": 11,
    "text": 11,        # annotations and figure footers
}

# Step 6 heatmaps.
HEATMAP_FONT_SIZES = {
    "title": 19,
    "axis_label": 18,
    "tick_label": 18,
    "legend": 18,
    "text": 18,
}


def font_sizes_plus(points, base=None):
    """Copy of `base` (default FONT_SIZES) with every size increased by `points`."""
    base = FONT_SIZES if base is None else base
    return {key: size + points for key, size in base.items()}


# Colours for stations without an entry in a script's STATION_COLORS (e.g. a
# station added to stations.json). None of them matches the four study stations.
FALLBACK_STATION_COLORS = ["tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]


def station_color(station, known_colors, stations):
    """
    Colour of `station` in a comparison figure: its entry in known_colors or,
    failing that, a fallback colour assigned in alphabetical order among the
    stations of the figure that have no entry.
    """
    if station in known_colors:
        return known_colors[station]
    unknown = sorted(s for s in stations if s not in known_colors)
    return FALLBACK_STATION_COLORS[unknown.index(station) % len(FALLBACK_STATION_COLORS)]


def with_font_sizes(plot_function=None, *, sizes=None):
    """
    Decorator: draw the figure with FONT_SIZES, or with `sizes` if given.
    plt.rc_context restores the settings afterwards, so other figures and
    other scripts in the same process (e.g. web_server.py) are not affected.
    """
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            active = sizes if sizes is not None else FONT_SIZES
            with plt.rc_context({
                "font.size": active["text"],
                "axes.titlesize": active["title"],
                "figure.titlesize": active["title"],
                "axes.labelsize": active["axis_label"],
                "xtick.labelsize": active["tick_label"],
                "ytick.labelsize": active["tick_label"],
                "legend.fontsize": active["legend"],
                "legend.title_fontsize": active["legend"],
            }):
                return func(*args, **kwargs)
        return wrapper

    if plot_function is not None:
        return decorate(plot_function)
    return decorate
