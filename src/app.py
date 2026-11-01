"""Interactive results explorer for the ADBench Angle I replication.

Reads every results CSV through the shared loader, so any new algorithm shows
up automatically once its runner writes a CSV to results/.

Run from the repo root:
    streamlit run src/app.py
"""

import streamlit as st

from load_results import load_all_results

st.set_page_config(page_title="ADBench Label Efficiency", layout="wide")


@st.cache_data
def get_results():
    return load_all_results()


st.title("Label efficiency in anomaly detection")
st.caption("CMPE 257 replication of ADBench Angle I: how much do a few labeled anomalies help?")

results = get_results()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Algorithms", results["algorithm"].nunique())
c2.metric("Datasets", results["dataset"].nunique())
c3.metric("Result rows", len(results))
c4.metric("Results files", results["source_file"].nunique())

st.subheader("Algorithms found")
found = (results.groupby(["category", "algorithm", "source_file"]).size()
         .reset_index(name="rows").sort_values(["category", "algorithm"]))
st.dataframe(found, hide_index=True, width="stretch")

with st.expander("All result rows"):
    st.dataframe(results, hide_index=True, width="stretch")

if st.button("Reload results"):
    get_results.clear()
    st.rerun()
