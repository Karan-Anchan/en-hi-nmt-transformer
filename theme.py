"""Shared minimalist-luxury palette for plots.

Two themes — ``light`` and ``dark`` — that the plotting modules can opt
into without duplicating colour definitions. Same accent in both modes so
the brand feels consistent across the README.
"""
from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap


LIGHT = {
    'ink': '#1A1A1A',          # primary text
    'soft_ink': '#5C5C5C',     # muted text
    'hairline': '#E5E1D8',     # subtle separators
    'canvas': '#FBFAF6',       # background
    'primary': '#1F2937',      # primary series (deep slate)
    'accent': '#B8956A',       # accent (champagne)
    'whisper': '#C9C2B0',      # auxiliary lines
}

DARK = {
    'ink': '#EFEAD8',          # warm off-white text
    'soft_ink': '#8F8B7D',     # muted warm gray
    'hairline': '#262422',     # subtle separators
    'canvas': '#0E0D0B',       # deep warm-black background
    'primary': '#E8DCC4',      # warm cream as primary on dark
    'accent': '#C9A961',       # champagne gold (same)
    'whisper': '#3A352D',      # auxiliary lines
}


def palette(dark: bool = False) -> dict[str, str]:
    return DARK if dark else LIGHT


def metric_cmap(dark: bool = False) -> LinearSegmentedColormap:
    """Ramp used to color line plots by value — both themes are colourful,
    flowing cool → warm. Direction of *visibility* flips per theme:

    * **Dark**:  low = cool/dark (recedes into canvas), high = warm/bright (pops).
    * **Light**: low = cool/pale (recedes into canvas), high = warm/deep (pops).

    Same hue family in both, just adjusted brightness, so the metric lines
    and the attention heatmaps feel like part of the same visual system.
    """
    if dark:
        stops = ['#1E1B3A', '#4A1F5C', '#8A2D5A', '#C8513F', '#E89B4A', '#F2D67E']
    else:
        stops = ['#D9D4E0', '#B89DBA', '#B47F8E', '#A8624E', '#7A3E26', '#3A1E1A']
    return LinearSegmentedColormap.from_list('minlux_metric', stops, N=256)


def attention_cmap(dark: bool = False) -> LinearSegmentedColormap:
    """Attention ramp from canvas (low) to a vivid high.

    Both themes use the same sunset/magma family — cool/cool-pale on the
    low end (fades into the canvas) flowing through warm hues to a
    high-end that pops against its background.

    * Dark: warm-black → indigo → violet → mulberry → coral → amber → gold.
    * Light: warm cream → dusty lavender → mauve → rose → terracotta → wine.
    """
    p = palette(dark)
    if dark:
        stops = [
            p['canvas'],   # #0E0D0B  warm-black
            '#1E1B3A',     # midnight indigo
            '#4A1F5C',     # deep violet
            '#8A2D5A',     # mulberry / wine
            '#C8513F',     # warm coral / brick
            '#E89B4A',     # amber
            '#F2D67E',     # pale champagne gold
        ]
    else:
        stops = [
            p['canvas'],   # #FBFAF6  warm cream — empty cells fade away
            '#E8DDE6',     # whisper of lavender
            '#D9D4E0',     # pale lavender
            '#B89DBA',     # mauve
            '#B47F8E',     # warm rose
            '#A8624E',     # terracotta
            '#7A3E26',     # deep amber-brown
            '#3A1E1A',     # deep wine-ink
        ]
    return LinearSegmentedColormap.from_list('minlux_attention', stops, N=256)


def apply_rc(dark: bool = False) -> dict[str, str]:
    """Set matplotlib rcParams for the chosen theme and return the palette."""
    import matplotlib as mpl
    p = palette(dark)
    mpl.rcParams.update({
        'figure.dpi': 140,
        'savefig.dpi': 200,
        'savefig.bbox': 'tight',
        'savefig.facecolor': p['canvas'],
        'figure.facecolor': p['canvas'],
        'axes.facecolor': p['canvas'],
        'font.family': ['Inter', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10.5,
        'text.color': p['ink'],
        'axes.labelcolor': p['soft_ink'],
        'axes.edgecolor': p['hairline'],
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.bottom': True,
        'axes.spines.left': True,
        'axes.titlesize': 13,
        'axes.titleweight': 'medium',
        'axes.titlepad': 24,
        'axes.titlelocation': 'left',
        'axes.labelpad': 12,
        'axes.labelsize': 10,
        'xtick.color': p['soft_ink'],
        'ytick.color': p['soft_ink'],
        'xtick.major.size': 0,
        'ytick.major.size': 0,
        'xtick.major.pad': 8,
        'ytick.major.pad': 8,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
        'axes.grid': False,
        'legend.frameon': False,
        'legend.fontsize': 9.5,
        'lines.solid_capstyle': 'round',
        'lines.solid_joinstyle': 'round',
    })
    return p
