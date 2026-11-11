"""Six-Site Data Center Capacity Simulator - Streamlit entry point.

    streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Six-Site DC Capacity Simulator",
    page_icon=":material/dns:",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui import components, sidebar, state  # noqa: E402

components.inject_css()
state.ensure_state()

nav = st.navigation([
    st.Page("views/overview.py", title="Executive Overview", icon=":material/dashboard:", default=True),
    st.Page("views/forecasting.py", title="Capacity Forecasting", icon=":material/show_chart:"),
    st.Page("views/scenarios.py", title="Scenario Planning", icon=":material/tune:"),
    st.Page("views/decisions.py", title="Executive Decisions", icon=":material/gavel:"),
])
sidebar.render()
components.synthetic_banner()
nav.run()
