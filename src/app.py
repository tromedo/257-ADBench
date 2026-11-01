"""Interactive results explorer for the ADBench Angle I replication.

Reads every results CSV through the shared loader, so any new algorithm shows
up automatically once its runner writes a CSV to results/.

Run from the repo root:
    streamlit run src/app.py
"""

import altair as alt
import streamlit as st

from load_results import load_all_results, summarize
from plot_results import BASELINE, DATASET_TITLES, METRIC_LABELS, algorithm_colors, anomaly_ratio

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

st.subheader("Performance vs fraction of labeled anomalies")

datasets = [d for d in DATASET_TITLES if d in set(results["dataset"])]
all_algorithms = sorted(results["algorithm"].unique())
f1, f2, f3 = st.columns([1, 1, 2])
dataset = f1.selectbox("Dataset", datasets, format_func=lambda d: DATASET_TITLES[d])
metric = f2.radio("Metric", list(METRIC_LABELS), format_func=METRIC_LABELS.get, horizontal=True)
chosen = f3.multiselect("Algorithms", all_algorithms, default=all_algorithms)

if not chosen:
    st.info("Pick at least one algorithm.")
else:
    colors, order = algorithm_colors(set(all_algorithms))  # colors follow the algorithm, not the selection
    part = results[(results["dataset"] == dataset) & (results["algorithm"].isin(chosen))]
    summary = summarize(part, metric)
    labeled = (part.groupby(["algorithm", "label_fraction"])["n_labeled"].first().reset_index())
    summary = summary.merge(labeled, on=["algorithm", "label_fraction"])
    summary["label"] = summary["label_fraction"].map(lambda f: f"{f * 100:g}%")
    fraction_order = [f"{f * 100:g}%" for f in sorted(results["label_fraction"].unique()) if f > 0]

    color = alt.Color("algorithm:N", title="Algorithm",
                      scale=alt.Scale(domain=order, range=[colors[a] for a in order]))
    tooltip = [alt.Tooltip("algorithm:N", title="Algorithm"),
               alt.Tooltip("category:N", title="Category"),
               alt.Tooltip("label:N", title="Label fraction"),
               alt.Tooltip("n_labeled:Q", title="Labeled anomalies"),
               alt.Tooltip("mean:Q", title=f"Mean {METRIC_LABELS[metric]}", format=".3f"),
               alt.Tooltip("std:Q", title="Std over seeds", format=".3f")]
    y = alt.Y("mean:Q", title=f"{METRIC_LABELS[metric]} (mean of 3 seeds)",
              scale=alt.Scale(domain=[0, 1]))

    labeled_part = summary[summary["category"] != "unsupervised"]
    unsup_part = summary[summary["category"] == "unsupervised"].copy()
    unsup_part["label"] = "All fractions (no labels used)"

    layers = []
    if not labeled_part.empty:
        base = alt.Chart(labeled_part).encode(
            x=alt.X("label:N", sort=fraction_order, title="Labeled anomalies in training",
                    axis=alt.Axis(labelAngle=0)),
            y=y, color=color, tooltip=tooltip)
        layers += [base.mark_line(strokeWidth=2),
                   base.mark_point(filled=True, size=70, strokeWidth=1.5, stroke="white")]
    if not unsup_part.empty:
        layers.append(alt.Chart(unsup_part).mark_rule(strokeWidth=2, strokeDash=[6, 4])
                      .encode(y=y, color=color, tooltip=tooltip))
    if metric == "auc_pr":
        layers.append(alt.Chart({"values": [{"ratio": anomaly_ratio(dataset)}]})
                      .mark_rule(color=BASELINE, strokeDash=[1, 3], strokeWidth=1.5)
                      .encode(y="ratio:Q", tooltip=alt.Tooltip("ratio:Q", title="Random guess", format=".3f")))

    st.altair_chart(alt.layer(*layers).properties(height=420), width="stretch")
    note = "Dashed lines: unsupervised methods, which ignore labels."
    if metric == "auc_pr":
        note += " Dotted gray line: random guess (the dataset's anomaly ratio)."
    st.caption(note + " Hover any point or line for exact values.")

with st.expander("All result rows"):
    st.dataframe(results, hide_index=True, width="stretch")

if st.button("Reload results"):
    get_results.clear()
    st.rerun()
