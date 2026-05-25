"""OfflineRL-Lab Streamlit Dashboard."""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="OfflineRL-Lab Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = ["Dataset Diagnostics", "Training Runs", "Policy Comparison", "Failure Explorer"]
page = st.sidebar.selectbox("Page", pages)

# ─── Dataset Diagnostics ──────────────────────────────────────────────────────

if page == "Dataset Diagnostics":
    st.title("Dataset Diagnostics")
    st.markdown(
        "Upload an HDF5 dataset (`.h5` / `.hdf5`) to run a full diagnostic analysis."
    )

    uploaded_file = st.file_uploader("Upload HDF5 Dataset", type=["h5", "hdf5"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            from offline_rl.datasets.loader import load_dataset
            from offline_rl.datasets.diagnostics import DatasetDiagnostics

            with st.spinner("Loading dataset and running diagnostics..."):
                dataset = load_dataset(tmp_path)
                diag = DatasetDiagnostics(dataset)
                report = diag.run_all()

            # ── Summary metrics ───────────────────────────────────────────────
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Transitions", f"{report.n_transitions:,}")
            col2.metric("Episodes", f"{report.n_episodes:,}")
            col3.metric("Coverage Score", f"{report.coverage_score:.3f}")

            risk_color = {"low": "🟢", "medium": "🟡", "high": "🔴"}
            risk = report.ood_risk["risk_level"]
            col4.metric(
                "OOD Risk",
                f"{risk_color.get(risk, '⚪')} {risk.upper()}",
            )

            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Obs Dim", str(report.obs_dim))
            col6.metric("Act Dim", str(report.act_dim))
            col7.metric(
                "Terminal Rate",
                f"{report.episode_stats['terminal_rate']*100:.1f}%",
            )
            col8.metric("Support Mismatch", f"{report.support_mismatch_risk:.3f}")

            # ── Reward distribution ───────────────────────────────────────────
            st.subheader("Reward Distribution")
            fig = px.histogram(
                x=dataset["rewards"].flatten(),
                nbins=50,
                labels={"x": "Reward"},
                title="Reward Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── PCA scatter ───────────────────────────────────────────────────
            if report.state_pca_points is not None:
                st.subheader("State Space Coverage (PCA)")
                pts = np.array(report.state_pca_points)
                fig = px.scatter(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    opacity=0.3,
                    title=(
                        f"PCA of States "
                        f"(explained var: {report.pca_explained_variance:.2%})"
                    ),
                    labels={"x": "PC1", "y": "PC2"},
                )
                st.plotly_chart(fig, use_container_width=True)

            # ── Action coverage ───────────────────────────────────────────────
            st.subheader("Action Coverage")
            act_cov = report.action_coverage
            dim_keys = [k for k in act_cov.keys() if k.startswith("dim_")]
            if dim_keys:
                cov_df = pd.DataFrame(
                    {
                        "Dimension": dim_keys,
                        "Coverage": [act_cov[k]["coverage"] for k in dim_keys],
                        "Mean": [act_cov[k]["mean"] for k in dim_keys],
                        "Std": [act_cov[k]["std"] for k in dim_keys],
                    }
                )
                fig = px.bar(
                    cov_df,
                    x="Dimension",
                    y="Coverage",
                    title="Action Coverage per Dimension",
                )
                st.plotly_chart(fig, use_container_width=True)

            # ── Warnings ──────────────────────────────────────────────────────
            warnings = []
            if report.ood_risk["risk_level"] == "high":
                frac = report.ood_risk["sparse_state_fraction"]
                warnings.append(
                    f"{frac*100:.1f}% of random actions are far from dataset support"
                )
            if abs(report.reward_stats["skewness"]) > 2.0:
                warnings.append(
                    f"Reward distribution heavily skewed "
                    f"(skewness={report.reward_stats['skewness']:.2f})"
                )
            if report.episode_stats["terminal_rate"] < 0.1:
                warnings.append(
                    "Very few episodes reach terminal state — check done flags"
                )
            if len(report.outlier_indices) > 10:
                warnings.append(
                    f"{len(report.outlier_indices)} outlier transitions detected "
                    f"(IQR-based)"
                )

            if warnings:
                st.subheader("Warnings")
                for w in warnings:
                    st.warning(f"· {w}")
            else:
                st.success("No significant warnings found.")

        finally:
            os.unlink(tmp_path)

# ─── Training Runs ────────────────────────────────────────────────────────────

elif page == "Training Runs":
    st.title("Training Runs")

    runs_dir = Path("runs")
    if not runs_dir.exists():
        st.info("No `runs/` directory found. Train a policy first with `orl train`.")
    else:
        run_dirs = sorted(
            [d for d in runs_dir.iterdir() if d.is_dir() and (d / "metrics.jsonl").exists()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )

        if not run_dirs:
            st.info("No completed runs found. Check that `metrics.jsonl` exists.")
        else:
            selected_runs = st.multiselect(
                "Select runs to compare",
                options=[d.name for d in run_dirs],
                default=[run_dirs[0].name] if run_dirs else [],
            )

            if selected_runs:
                all_metrics: dict = {}
                for run_name in selected_runs:
                    metrics_path = runs_dir / run_name / "metrics.jsonl"
                    records = []
                    with open(metrics_path) as f:
                        for line in f:
                            try:
                                records.append(json.loads(line.strip()))
                            except Exception:
                                pass
                    if records:
                        all_metrics[run_name] = records

                st.subheader("Training Curves")
                metric_keys = [
                    "loss", "q_loss", "policy_loss", "cql_penalty",
                    "value_loss", "bc_loss",
                ]

                for key in metric_keys:
                    dfs = []
                    for run_name, records in all_metrics.items():
                        vals = [
                            (r.get("step", i), r.get(key))
                            for i, r in enumerate(records)
                            if key in r
                        ]
                        if vals:
                            steps, values = zip(*vals)
                            df = pd.DataFrame(
                                {"step": steps, key: values, "run": run_name}
                            )
                            dfs.append(df)

                    if dfs:
                        df_all = pd.concat(dfs, ignore_index=True)
                        fig = px.line(
                            df_all, x="step", y=key, color="run", title=key
                        )
                        st.plotly_chart(fig, use_container_width=True)

                # Config view
                for run_name in selected_runs:
                    for ext in ["yaml", "json"]:
                        config_path = runs_dir / run_name / f"config.{ext}"
                        if config_path.exists():
                            with st.expander(f"Config: {run_name}"):
                                st.code(config_path.read_text(), language=ext)
                            break

# ─── Policy Comparison ────────────────────────────────────────────────────────

elif page == "Policy Comparison":
    st.title("Policy Comparison")

    # Try loading from benchmark artifacts
    artifacts_dir = Path("artifacts") / "benchmark_tables"
    runs_dir = Path("runs")

    if artifacts_dir.exists() and any(artifacts_dir.glob("*.json")):
        st.subheader("Benchmark Table")
        artifact_files = sorted(artifacts_dir.glob("*.json"), reverse=True)
        selected_artifact = st.selectbox(
            "Select benchmark run",
            options=[f.name for f in artifact_files],
        )
        if selected_artifact:
            with open(artifacts_dir / selected_artifact) as f:
                bench_data = json.load(f)
            results = bench_data.get("results", {})
            rows = []
            for algo, seed_results in results.items():
                losses = [
                    r.get("final_loss")
                    for r in seed_results
                    if r.get("final_loss") is not None
                    and not (isinstance(r.get("final_loss"), float) and np.isnan(r.get("final_loss")))
                ]
                rows.append(
                    {
                        "Algorithm": algo.upper(),
                        "N Seeds": len(seed_results),
                        "Mean Loss": f"{np.mean(losses):.4f}" if losses else "N/A",
                        "Std Loss": f"{np.std(losses):.4f}" if losses else "N/A",
                    }
                )
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

    elif runs_dir.exists():
        run_dirs = [
            d for d in runs_dir.iterdir()
            if d.is_dir() and (d / "metrics.jsonl").exists()
        ]
        if run_dirs:
            st.subheader("Run Comparison")
            comparison_data = []
            for run_dir in run_dirs:
                records = []
                with open(run_dir / "metrics.jsonl") as f:
                    for line in f:
                        try:
                            records.append(json.loads(line.strip()))
                        except Exception:
                            pass
                if records:
                    last = records[-1]
                    comparison_data.append(
                        {
                            "run": run_dir.name,
                            "final_loss": last.get("loss", last.get("q_loss")),
                            "mean_return": last.get("mean_return"),
                        }
                    )

            if comparison_data:
                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True)

                if df["mean_return"].notna().any():
                    fig = px.bar(
                        df[df["mean_return"].notna()],
                        x="run",
                        y="mean_return",
                        title="Mean Return by Run",
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No runs found. Train policies first.")
    else:
        st.info("No runs or benchmark data found.")

# ─── Failure Explorer ─────────────────────────────────────────────────────────

elif page == "Failure Explorer":
    st.title("Failure Explorer")
    st.info("Load a checkpoint and dataset to explore policy failures.")

    col1, col2 = st.columns(2)
    checkpoint_path = col1.text_input("Checkpoint path", "runs/latest/checkpoint.pt")
    dataset_path = col2.text_input("Dataset path", "")
    n_episodes = st.slider("Episodes to collect", 10, 100, 20)

    if st.button("Run Failure Analysis") and checkpoint_path:
        try:
            # This is a UI placeholder - full implementation requires checkpoint loading
            st.info("Failure analysis requires a running environment. Feature available via CLI.")
        except Exception as e:
            st.error(f"Error: {e}")
