"""Capacity Forecasting (Milestone 3)."""

import streamlit as st

from ui import components as ui
from ui import state

ctx = state.context()
ui.page_header("Capacity Forecasting", ctx)
st.info("This page is being built in Milestone 3. Coming here: Monthly power, cooling and rack projections per site, exhaustion dates, first limiting resource, actual vs projected views, and CSV export.", icon=":material/construction:")
