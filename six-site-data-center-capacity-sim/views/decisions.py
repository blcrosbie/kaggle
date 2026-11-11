"""Executive Decisions (Milestone 4)."""

import streamlit as st

from ui import components as ui
from ui import state

ctx = state.context()
ui.page_header("Executive Decisions", ctx)
st.info("This page is being built in Milestone 4. Coming here: Capital investment ranking vs budget, recommended expansion timing, deferral consequences, optimization opportunities, and the executive recommendations panel.", icon=":material/construction:")
