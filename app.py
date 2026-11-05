"""
Complaint & Fraud Ops Workbench - IR Assignment 2 (Text Mining & Document Intelligence)

Run with:   python -m streamlit run app.py

The six pages follow the six workflow parts of the assignment brief, in order.
"""

import os

import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Complaint & Fraud Ops Workbench", layout="wide",
                   page_icon=os.path.join(HERE, "assets", "icon.svg"))

from views import common
from views import dashboard, ingestion, preprocessing, mining, recommend, performance

common.init_state()

st.logo(os.path.join(HERE, "assets", "logo.svg"), size="large")

pages = [
    st.Page(dashboard.render, title="Dashboard", icon=":material/space_dashboard:",
            url_path="dashboard", default=True),
    st.Page(ingestion.render, title="Data ingestion", icon=":material/move_to_inbox:", url_path="ingestion"),
    st.Page(preprocessing.render, title="Preprocessing", icon=":material/tune:", url_path="preprocessing"),
    st.Page(mining.render, title="Text mining hub", icon=":material/hub:", url_path="text-mining"),
    st.Page(recommend.render, title="Recommendations", icon=":material/manage_search:", url_path="recommendations"),
    st.Page(performance.render, title="Performance analytics", icon=":material/monitoring:", url_path="performance"),
]
nav = st.navigation(pages, position="top")

# the pipeline bar sits above every page but is filled in after the page ran,
# so it shows what the page has just computed
bar = st.container()
nav.run()
with bar:
    common.stepper()
    st.divider()
