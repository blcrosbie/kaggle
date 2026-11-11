"""Scenario Planning (Milestone 3)."""

import streamlit as st

from ui import components as ui
from ui import state

ctx = state.context()
ui.page_header("Scenario Planning", ctx)
st.info("This page is being built in Milestone 3. Coming here: Full what-if levers: portfolio and site-specific growth, power/cooling/rack expansion with activation month, efficiency improvement, capital budget, and scenario comparison.", icon=":material/construction:")
