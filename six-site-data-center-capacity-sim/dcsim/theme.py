"""Shared visual theme with two modes:

* dark (default): the Windows Terminal "Campbell PowerShell" scheme -
  navy #012456 background, #CCCCCC text, Campbell ANSI colours.
* light: white background with xterm-256 grays.

Data colours (sites, status, series) are the same in both modes, so the app's
charts and HTML can follow Streamlit's live theme switch without knowing which
mode is active. Each colour was checked with the dataviz palette validator on
both surfaces (#012456 dark, #ffffff light): lightness band, chroma floor,
normal-vision floor and contrast pass in both; the worst adjacent pair under
deuteranopia is ΔE 6.1 (floor band), which is legal because every site line
also carries its own dash pattern (SITE_DASH) and a legend.

"Low" risk is drawn in neutral gray: normal needs no colour, and that frees
green for a site. Medium / High / Critical use yellow / orange / red.
"""

from __future__ import annotations

# Campbell PowerShell (Windows Terminal) reference values.
CAMPBELL = {
    "background": "#012456", "foreground": "#CCCCCC",
    "black": "#0C0C0C", "red": "#C50F1F", "green": "#13A10E", "yellow": "#C19C00",
    "blue": "#0037DA", "purple": "#881798", "cyan": "#3A96DD", "white": "#CCCCCC",
    "brightBlack": "#767676", "brightRed": "#E74856", "brightGreen": "#16C60C", "brightYellow": "#F9F1A5",
    "brightBlue": "#3B78FF", "brightPurple": "#B4009E", "brightCyan": "#61D6D6", "brightWhite": "#F2F2F2",
}

MODES = {
    "dark": {  # Campbell PowerShell
        "SURFACE": "#012456", "SURFACE_RAISED": "#0a2d5e", "PANEL_HEAD": "#123a70",
        "GRID": "#1d4378", "BORDER": "#4a6a99",
        "TEXT_PRIMARY": "#f2f2f2", "TEXT_SECONDARY": "#cccccc", "TEXT_MUTED": "#9aa7bb",
    },
    "light": {  # xterm-256 grays
        "SURFACE": "#ffffff", "SURFACE_RAISED": "#f3f3f3", "PANEL_HEAD": "#dadada",
        "GRID": "#e4e4e4", "BORDER": "#8a8a8a",
        "TEXT_PRIMARY": "#121212", "TEXT_SECONDARY": "#444444", "TEXT_MUTED": "#626262",
    },
}
DEFAULT_MODE = "dark"

# Mode-dependent neutrals (module globals so the matplotlib visuals can switch with set_mode()).
SURFACE = SURFACE_RAISED = PANEL_HEAD = GRID = BORDER = TEXT_PRIMARY = TEXT_SECONDARY = TEXT_MUTED = ""


def set_mode(mode: str = DEFAULT_MODE) -> None:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}")
    globals().update(MODES[mode])


set_mode(DEFAULT_MODE)

# ---- Mode-independent data colours (readable on navy and on white) ----------
ACCENT = CAMPBELL["cyan"]   # primary series (portfolio demand, current-utilization bars)
NEUTRAL = "#8a8a8a"         # capacity lines and horizon markers: >= 3:1 on both surfaces

# Fixed site -> colour mapping; colour follows the site everywhere, never its rank.
SITE_COLORS = {
    "DC1": "#3A96DD",  # Campbell cyan
    "DC2": "#5F8700",  # xterm 64  olive
    "DC3": "#AF5FFF",  # xterm 135 violet
    "DC4": "#00875F",  # xterm 29  green
    "DC5": "#D7008F",  # xterm 162 pink
    "DC6": "#3B78FF",  # Campbell bright blue
}
SITE_DASH = {"DC1": "solid", "DC2": "dash", "DC3": "solid", "DC4": "dot", "DC5": "dashdot", "DC6": "longdash"}

STATUS = {"good": NEUTRAL, "warning": CAMPBELL["yellow"], "serious": "#E07B00", "critical": CAMPBELL["brightRed"]}
RISK_COLORS = {
    "Low": STATUS["good"],
    "Medium": STATUS["warning"],
    "High": STATUS["serious"],
    "Critical": STATUS["critical"],
}
# Filled status tags: (background, text) - each pair >= 4.5:1.
RISK_TAG = {
    "Low": ("#767676", "#ffffff"),
    "Medium": ("#F9F1A5", "#0C0C0C"),
    "High": ("#E07B00", "#0C0C0C"),
    "Critical": ("#C50F1F", "#ffffff"),
}
RISK_ICONS = {"Low": "●", "Medium": "▲", "High": "◆", "Critical": "■"}

FONT_FAMILY = ["Ubuntu", "DejaVu Sans", "Segoe UI", "Arial"]
MONO_FAMILY = ["Ubuntu Mono", "DejaVu Sans Mono", "Consolas", "Menlo", "monospace"]


def apply_matplotlib_theme(mode: str | None = None) -> None:
    import matplotlib as mpl

    if mode:
        set_mode(mode)
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BORDER,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.color": BORDER,
        "ytick.color": BORDER,
        "xtick.labelcolor": TEXT_SECONDARY,
        "ytick.labelcolor": TEXT_SECONDARY,
        "text.color": TEXT_PRIMARY,
        "legend.frameon": False,
        "legend.labelcolor": TEXT_SECONDARY,
        "font.family": "monospace",
        "font.monospace": MONO_FAMILY,
        "font.size": 11,
        "lines.solid_capstyle": "butt",
        "lines.solid_joinstyle": "miter",
    })
