import streamlit as st
import psycopg2
import psycopg2.extras
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
import torch
import os
import math
from pathlib import Path


def get_db_connection():
    """Create database connection"""
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    return psycopg2.connect(
        dbname=config["database"]["name"],
        user=config["database"]["user"],
        password=config["database"]["password"],
        host=config["database"]["host"],
    )


def get_dataset_info(file_path):
    """Safely get dataset information from PyTorch file"""
    try:
        data = torch.load(file_path)
        if isinstance(data, tuple):
            tensors, labels = data
            shape = tuple(tensors.size())
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            return {"shape": shape, "size_mb": size_mb, "num_samples": len(labels)}
        else:
            shape = tuple(data.size())
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            return {"shape": shape, "size_mb": size_mb, "num_samples": shape[0]}
    except Exception as e:
        return {"error": str(e)}


def load_experiment_data():
    """Load experiment data from database"""
    conn = get_db_connection()
    try:
        # Get experiments
        query = """
            SELECT 
                e.*,
                tm.metrics as training_metrics,
                em.metrics as eval_metrics
            FROM experiments e
            LEFT JOIN LATERAL (
                SELECT 
                    experiment_id,
                    json_object_agg(epoch, metrics) as metrics
                FROM training_metrics
                GROUP BY experiment_id
            ) tm ON tm.experiment_id = e.experiment_id
            LEFT JOIN LATERAL (
                SELECT 
                    experiment_id,
                    json_object_agg(eval_id, metrics) as metrics
                FROM evaluation_metrics
                GROUP BY experiment_id
            ) em ON em.experiment_id = e.experiment_id
        """

        df = pd.read_sql(query, conn)
        return df
    finally:
        conn.close()


def get_selected_runs(experiments_df):
    """Get selected runs from sidebar"""
    st.sidebar.header("Run Selection")

    # Get all run names
    run_names = experiments_df["name"].tolist()

    # Multi-select dropdown with max 16 selections
    selected_runs = st.sidebar.multiselect(
        "Select Runs (max 16)",
        options=run_names,
        default=run_names[:4] if len(run_names) >= 4 else run_names,
        max_selections=16,
    )

    # Filter runs based on selection
    selected_runs_df = experiments_df[experiments_df["name"].isin(selected_runs)]
    return selected_runs_df


def display_config_comparison(runs):
    """Display comparison of experiment configurations"""
    st.header("Experiment Configurations")

    # Extract configurations
    configs_df = pd.DataFrame([run["config"] for _, run in runs.iterrows()])
    configs_df["run_name"] = runs["name"]

    # Reorder columns to put run_name first
    cols = ["run_name"] + [col for col in configs_df.columns if col != "run_name"]
    configs_df = configs_df[cols]

    st.dataframe(configs_df)


def plot_training_curves(runs):
    """Plot training curves for each run"""
    st.header("Training Curves")

    n_runs = len(runs)
    if n_runs == 0:
        st.warning("No runs selected")
        return

    # Calculate grid dimensions
    cols = math.ceil(math.sqrt(n_runs))
    rows = math.ceil(n_runs / cols)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=runs["name"].tolist(),
        specs=[[{"secondary_y": True}] * cols for _ in range(rows)],
    )

    for idx, (_, run) in enumerate(runs.iterrows()):
        row = (idx // cols) + 1
        col = (idx % cols) + 1

        # Extract metrics for each epoch
        training_data = pd.DataFrame(run["training_metrics"]).T

        # Plot training and validation metrics
        metrics = {
            "train_loss": {"color": "blue", "dash": "solid"},
            "val_loss": {"color": "red", "dash": "solid"},
            "train_accuracy": {"color": "green", "dash": "dash"},
            "val_accuracy": {"color": "orange", "dash": "dash"},
        }

        for metric, style in metrics.items():
            if metric in training_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=training_data.index,
                        y=training_data[metric],
                        name=metric.replace("_", " ").title(),
                        line=dict(color=style["color"], dash=style["dash"]),
                        showlegend=(idx == 0),
                    ),
                    row=row,
                    col=col,
                    secondary_y="accuracy" in metric.lower(),
                )

        # Update axes labels
        fig.update_xaxes(title_text="Epochs", row=row, col=col)
        fig.update_yaxes(title_text="Loss", secondary_y=False, row=row, col=col)
        fig.update_yaxes(title_text="Accuracy", secondary_y=True, row=row, col=col)

    height = max(400, rows * 400)
    fig.update_layout(
        height=height,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)


def display_evaluation_metrics(runs):
    """Display evaluation metrics for each run"""
    st.header("Evaluation Metrics")

    eval_data = []
    for _, run in runs.iterrows():
        metrics = run["eval_metrics"]
        if metrics:
            # Get the latest evaluation metrics
            latest_eval = list(metrics.values())[-1]
            latest_eval["run_name"] = run["name"]
            eval_data.append(latest_eval)

    if not eval_data:
        st.warning("No evaluation metrics available")
        return

    eval_df = pd.DataFrame(eval_data)

    # Create visualization
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Test Accuracy", "Test Loss"))

    # Original test metrics
    fig.add_trace(
        go.Bar(
            name="Original",
            x=eval_df["run_name"],
            y=eval_df["test_accuracy"],
            text=eval_df["test_accuracy"].round(3),
            textposition="auto",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            name="Original",
            x=eval_df["run_name"],
            y=eval_df["test_loss"],
            text=eval_df["test_loss"].round(3),
            textposition="auto",
        ),
        row=1,
        col=2,
    )

    # Blurred test metrics
    if "test_blurred_accuracy" in eval_df.columns:
        fig.add_trace(
            go.Bar(
                name="Blurred",
                x=eval_df["run_name"],
                y=eval_df["test_blurred_accuracy"],
                text=eval_df["test_blurred_accuracy"].round(3),
                textposition="auto",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                name="Blurred",
                x=eval_df["run_name"],
                y=eval_df["test_blurred_loss"],
                text=eval_df["test_blurred_loss"].round(3),
                textposition="auto",
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)


def display_artifacts(runs):
    """Display and allow downloading of artifacts"""
    st.header("Experiment Artifacts")

    conn = get_db_connection()
    try:
        for _, run in runs.iterrows():
            with st.expander(f"Artifacts - {run['name']}", expanded=True):
                # Get artifacts for this run
                cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cur.execute(
                    "SELECT * FROM artifacts WHERE experiment_id = %s",
                    (run["experiment_id"],),
                )
                artifacts = cur.fetchall()

                if not artifacts:
                    st.write("No artifacts found for this run")
                    continue

                # Group artifacts by type
                for artifact in artifacts:
                    st.subheader(f"📦 {artifact['name']}")

                    file_path = Path(artifact["file_path"])
                    if file_path.exists():
                        with open(file_path, "rb") as f:
                            file_bytes = f.read()

                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.download_button(
                                label=f"📥 Download {artifact['name']}",
                                data=file_bytes,
                                file_name=f"{artifact['name']}_{run['name']}{file_path.suffix}",
                                mime="application/octet-stream",
                            )

                        with col2:
                            if artifact["metadata"]:
                                for key, value in artifact["metadata"].items():
                                    st.write(f"{key}: {value}")

                            info = get_dataset_info(file_path)
                            if "error" not in info:
                                st.write(f"Size: {info['size_mb']:.2f} MB")
                                if "shape" in info:
                                    st.write(f"Shape: {info['shape']}")
                                if "num_samples" in info:
                                    st.write(f"Samples: {info['num_samples']}")
                    else:
                        st.warning(f"File not found: {file_path}")

                cur.close()
    finally:
        conn.close()


def main():
    st.set_page_config(page_title="MNIST Experiment Tracker", layout="wide")
    st.title("MNIST Experiment Tracker")

    # Load experiment data
    experiments_df = load_experiment_data()
    if experiments_df.empty:
        st.error("No experiments found in the database")
        return

    # Get selected runs
    selected_runs = get_selected_runs(experiments_df)

    if len(selected_runs) == 0:
        st.warning("Please select at least one run from the sidebar")
        return

    # Display different sections
    display_config_comparison(selected_runs)
    plot_training_curves(selected_runs)
    display_evaluation_metrics(selected_runs)
    display_artifacts(selected_runs)


if __name__ == "__main__":
    main()
