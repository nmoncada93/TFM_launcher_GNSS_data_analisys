#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tamaños de letra de las figuras, en un solo sitio.

Para probar otro tamaño: cambia un valor de FONT_SIZES (o de
HEATMAP_FONT_SIZES para los heatmaps del Step 6) y vuelve a generar la
figura. Solo afecta a la presentación: no cambia datos, cálculos, figsize,
layout, ticks, textos ni colores.
"""

from functools import wraps

import matplotlib.pyplot as plt

# Tamaños en puntos de Matplotlib (los valores por defecto de Matplotlib son
# 12 para títulos y 10 para el resto).
FONT_SIZES = {
    "title": 12,       # títulos
    "axis_label": 11,  # nombres de los ejes y etiqueta de la colorbar
    "tick_label": 11,  # números de los ejes y de la colorbar
    "legend": 11,      # leyendas
    "text": 11,        # anotaciones y pies de figura
}

# Heatmaps del Step 6: algo más grandes que el resto.
HEATMAP_FONT_SIZES = {
    "title": 14,
    "axis_label": 13,
    "tick_label": 13,
    "legend": 13,
    "text": 13,
}


def with_font_sizes(plot_function=None, *, sizes=None):
    """
    Dibuja la figura con los tamaños de FONT_SIZES, o con los de `sizes` si
    se indica (p. ej. @with_font_sizes(sizes=HEATMAP_FONT_SIZES)). Usa
    plt.rc_context, que restaura la configuración al terminar, así que no
    afecta a otras figuras ni a otros scripts cargados en el mismo proceso
    (p. ej. web_server.py).
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
