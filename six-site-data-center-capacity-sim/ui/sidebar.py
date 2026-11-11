"""Shared sidebar: predefined scenarios, reset, and planning assumptions (every page)."""

from __future__ import annotations

import streamlit as st

from dcsim import config as C

from . import state
from .components import default_dark_theme

PRESET_HELP = {
    "Baseline": "Current operating assumptions and planned expansions.",
    "High Growth": "Every site's demand grows 25% per year.",
    "Infrastructure Expansion": "Adds the sized expansion to the site with the earliest constraint (editable on Scenario Planning).",
    "Efficiency Optimization": "Reduces modeled power and cooling demand by 10%.",
    "Budget-Constrained": "Funds ranked expansion projects within a $60M capital budget.",
}


def render() -> None:
    with st.sidebar:
        st.markdown("#### Scenario")
        active = st.session_state["preset"]
        for name in state.PRESETS:
            st.button(
                name, key=f"preset_{name}", width="stretch", help=PRESET_HELP[name],
                type="primary" if active == name else "secondary",
                on_click=state.apply_preset, args=(name,),
            )
        if active == "Custom":
            st.caption("Custom scenario: levers changed on the Scenario Planning page.")
        st.button("↺ Reset to Baseline", key="reset", width="stretch", type="tertiary",
                  on_click=state.reset_to_baseline,
                  help="Restore every lever, threshold and the horizon to the baseline defaults.")

        st.markdown("#### Planning assumptions")
        st.select_slider("Forecast horizon (months)", options=list(C.HORIZON_CHOICES), key="horizon",
                         help="Controls charts, tables and expansion sizing. Risk status always looks 36 months ahead.")
        st.slider("Warning threshold (%)", 50, 99, key="warn_pct", step=1,
                  help="Utilization at which expansion planning should start; also the sizing target.")
        st.slider("Critical threshold (%)", 51, 100, key="crit_pct", step=1,
                  help="Utilization at which a site is treated as high operational risk.")
        st.divider()
        st.caption("Light / dark: app menu ⋮ (top right). Dark is the Campbell PowerShell scheme.")
        st.caption("Synthetic data — demonstration only. Calculations: see the Methodology on GitHub.")
        default_dark_theme()
