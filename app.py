import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import torch
import os
import math


# Set MLflow tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")


def get_dataset_info(file_path):
    """
    Safely get dataset information from PyTorch file
    """
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
    """Load experiment data from MLflow"""
    experiment = mlflow.get_experiment_by_name("MNIST_Experiments")
    if experiment is None:
        st.error("No experiment found with name 'MNIST_Experiments'")
        return None

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    return runs


def get_selected_runs(runs):
    """Get selected runs from sidebar"""
    st.sidebar.header("Run Selection")

    # Get all run names
    run_names = runs["tags.mlflow.runName"].tolist()

    # Multi-select dropdown with max 16 selections
    selected_runs = st.sidebar.multiselect(
        "Select Runs (max 16)",
        options=run_names,
        default=run_names[:4],  # Default to first 4 runs
        max_selections=16,
    )

    # Filter runs based on selection
    selected_runs_df = runs[runs["tags.mlflow.runName"].isin(selected_runs)]
    return selected_runs_df


def display_config_comparison(runs):
    """Display comparison of experiment configurations"""
    st.header("Experiment Configurations")

    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    params_df = runs[[col for col in runs.columns if col.startswith("params.")]].copy()
    params_df.columns = [col.replace("params.", "") for col in params_df.columns]

    # Add run names
    params_df.loc[:, "run_name"] = runs["tags.mlflow.runName"]

    # Filter for relevant parameters
    important_params = [
        "name",
        "hidden_size",
        "batch_size",
        "learning_rate",
        "max_epochs",
        "max_samples",
    ]
    available_params = [col for col in important_params if col in params_df.columns]
    params_df = params_df[["run_name"] + available_params]

    st.dataframe(params_df)


def plot_training_curves(runs):
    """Plot training curves with dynamically sized grid based on number of runs"""
    st.header("Training Curves")

    # Calculate required grid dimensions
    n_runs = len(runs)
    if n_runs == 0:
        st.warning("No runs selected")
        return

    # Calculate grid dimensions - aim for a roughly square grid
    cols = math.ceil(math.sqrt(n_runs))
    rows = math.ceil(n_runs / cols)

    # Create subplot grid with secondary y-axes
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=runs["tags.mlflow.runName"].tolist(),
        specs=[[{"secondary_y": True}] * cols for _ in range(rows)],
    )

    colors = {
        "train_loss": "blue",
        "val_loss": "red",
        "train_accuracy": "green",
        "val_accuracy": "orange",
    }

    # Plot each model in its own subplot
    for idx, run_id in enumerate(runs["run_id"]):
        row = (idx // cols) + 1
        col = (idx % cols) + 1

        # Get metric history for loss
        metric_history = mlflow.MlflowClient().get_metric_history(run_id, "train_loss")
        epochs = [m.step for m in metric_history]

        # Training and validation loss (primary y-axis)
        for metric in ["train_loss", "val_loss"]:
            try:
                metric_data = mlflow.MlflowClient().get_metric_history(run_id, metric)
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=[m.value for m in metric_data],
                        name=metric.replace("_", " ").title(),
                        line=dict(color=colors[metric]),
                        showlegend=(idx == 0),
                    ),
                    row=row,
                    col=col,
                    secondary_y=False,
                )
            except Exception as e:
                st.warning(f"Could not plot {metric} for run {run_id}: {str(e)}")

        # Training and validation accuracy (secondary y-axis)
        for metric in ["train_accuracy", "val_accuracy"]:
            try:
                metric_data = mlflow.MlflowClient().get_metric_history(run_id, metric)
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=[m.value for m in metric_data],
                        name=metric.replace("_", " ").title(),
                        line=dict(color=colors[metric], dash="dash"),
                        showlegend=(idx == 0),
                    ),
                    row=row,
                    col=col,
                    secondary_y=True,
                )
            except Exception as e:
                st.warning(f"Could not plot {metric} for run {run_id}: {str(e)}")

        # Update axes labels
        fig.update_xaxes(title_text="Epochs", row=row, col=col)
        fig.update_yaxes(title_text="Loss", secondary_y=False, row=row, col=col)
        fig.update_yaxes(title_text="Accuracy", secondary_y=True, row=row, col=col)

    # Adjust height based on number of rows
    height = max(400, rows * 400)  # Minimum height of 400px, then scales with rows

    fig.update_layout(
        height=height,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def create_metrics_figure(metrics_df, title_prefix):
    """Helper function to create metrics visualization"""
    if len(metrics_df.columns) < 3:  # run_name + at least 2 metrics
        st.warning(f"No {title_prefix.lower()} metrics available")
        return None

    # Find accuracy and loss columns
    accuracy_col = [col for col in metrics_df.columns if "accuracy" in col.lower()][0]
    loss_col = [col for col in metrics_df.columns if "loss" in col.lower()][0]

    # Create figure with subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(f"{title_prefix} Accuracy", f"{title_prefix} Loss"),
    )

    # Add accuracy bars
    fig.add_trace(
        go.Bar(
            name="Accuracy",
            x=metrics_df["run_name"],
            y=metrics_df[accuracy_col],
            text=metrics_df[accuracy_col].round(3),
            textposition="auto",
        ),
        row=1,
        col=1,
    )

    # Add loss bars
    fig.add_trace(
        go.Bar(
            name="Loss",
            x=metrics_df["run_name"],
            y=metrics_df[loss_col],
            text=metrics_df[loss_col].round(3),
            textposition="auto",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(height=400, showlegend=False)
    return fig


def display_metrics(runs, metric_prefix, title):
    """Generic function to display metrics"""
    st.header(title)

    metrics = runs[
        [col for col in runs.columns if col.startswith(f"metrics.{metric_prefix}")]
    ].copy()
    if not metrics.empty:
        metrics.columns = [
            col.replace(f"metrics.{metric_prefix}_", "") for col in metrics.columns
        ]
        metrics.loc[:, "run_name"] = runs["tags.mlflow.runName"]

        fig = create_metrics_figure(metrics, title)
        if fig:
            st.plotly_chart(fig, use_container_width=True)


def display_test_results(runs):
    """Display both regular and blurred test results"""
    st.header("Test Results")

    # Regular test metrics
    original_metrics = runs[
        [
            col
            for col in runs.columns
            if col.startswith("metrics.test_") and "blurred" not in col.lower()
        ]
    ].copy()
    original_metrics.columns = [
        col.replace("metrics.test_", "") for col in original_metrics.columns
    ]
    original_metrics.loc[:, "run_name"] = runs["tags.mlflow.runName"]

    # Blurred test metrics
    blurred_metrics = runs[
        [col for col in runs.columns if "blurred" in col.lower()]
    ].copy()
    blurred_metrics.columns = [
        col.replace("metrics.test_blurred_", "") for col in blurred_metrics.columns
    ]
    blurred_metrics.loc[:, "run_name"] = runs["tags.mlflow.runName"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Test Data")
        fig1 = create_metrics_figure(original_metrics, "Original Test")
        if fig1:
            st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Blurred Test Data")
        fig2 = create_metrics_figure(blurred_metrics, "Blurred Test")
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)


def display_dataset_metadata(runs):
    """Display metadata about the datasets used in the experiment"""
    st.header("Dataset Metadata")

    try:
        client = MlflowClient()
        run_id = runs.iloc[0]["run_id"]  # Get metadata from first run

        # Get the artifact path
        artifact_path = client.download_artifacts(run_id, "dataset_metadata.json")

        with open(artifact_path, "r") as f:
            metadata = json.load(f)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Original Dataset")
            st.write(f"Train Size: {metadata['original']['train_size']}")
            st.write(f"Test Size: {metadata['original']['test_size']}")
            st.write(
                f"Image Shape: {' × '.join(map(str, metadata['original']['image_shape']))}"
            )
            st.write(f"Number of Classes: {metadata['original']['num_classes']}")
            if "file_size_mb" in metadata["original"]:
                st.write(f"File Size: {metadata['original']['file_size_mb']:.2f} MB")

            st.write("Class Distribution:")
            dist_df = pd.DataFrame.from_dict(
                metadata["original"]["class_distribution"],
                orient="index",
                columns=["Count"],
            )
            st.dataframe(dist_df)

        with col2:
            st.subheader("Blurred Dataset")
            st.write(f"Train Size: {metadata['blurred']['train_size']}")
            st.write(f"Test Size: {metadata['blurred']['test_size']}")
            st.write(
                f"Image Shape: {' × '.join(map(str, metadata['blurred']['image_shape']))}"
            )
            st.write(f"Number of Classes: {metadata['original']['num_classes']}")
            st.write(f"Blur Sigma: {metadata['blurred']['blur_sigma']}")
            if "file_size_mb" in metadata["blurred"]:
                st.write(f"File Size: {metadata['blurred']['file_size_mb']:.2f} MB")

            st.write("Class Distribution:")
            blur_dist_df = pd.DataFrame.from_dict(
                metadata["blurred"]["class_distribution"],
                orient="index",
                columns=["Count"],
            )
            st.dataframe(blur_dist_df)

        with col3:
            if "limited" in metadata:
                st.subheader("Limited Dataset")
                st.write(f"Train Size: {metadata['limited']['train_size']}")
                st.write(f"Test Size: {metadata['limited'].get('test_size', 'N/A')}")
                st.write(
                    f"Image Shape: {' × '.join(map(str, metadata['limited']['image_shape']))}"
                )
                st.write(f"Number of Classes: {metadata['limited']['num_classes']}")
                if "file_size_mb" in metadata["limited"]:
                    st.write(f"File Size: {metadata['limited']['file_size_mb']:.2f} MB")

                st.write("Class Distribution:")
                limited_dist_df = pd.DataFrame.from_dict(
                    metadata["limited"]["class_distribution"],
                    orient="index",
                    columns=["Count"],
                )
                st.dataframe(limited_dist_df)

    except Exception as e:
        st.warning(f"Could not load dataset metadata: {str(e)}")


def display_artifact_downloads(runs):
    """Display and allow downloading of model checkpoints and dataset files"""
    st.header("Experiment Artifacts")

    client = MlflowClient()

    for _, run in runs.iterrows():
        run_id = run["run_id"]
        run_name = run["tags.mlflow.runName"]

        try:
            # Create an expander for each run
            with st.expander(f"Artifacts - {run_name}", expanded=True):
                # Model Checkpoint Section
                st.subheader("📦 Model Checkpoint")
                col1, col2 = st.columns([2, 1])

                with col1:
                    # Model weights download
                    model_path = "model/data/model.pth"
                    temp_path = client.download_artifacts(run_id, model_path)

                    with open(temp_path, "rb") as f:
                        model_bytes = f.read()

                    st.download_button(
                        label="📥 Download model weights",
                        data=model_bytes,
                        file_name=f"model_{run_name}.pth",
                        mime="application/octet-stream",
                    )

                    size_mb = len(model_bytes) / (1024 * 1024)
                    st.write(f"📊 Size: {size_mb:.2f} MB")

                with col2:
                    # Model metadata
                    st.write("Model Configuration:")
                    st.write(
                        f"- Hidden Size: {runs[runs['run_id'] == run_id]['params.hidden_size'].iloc[0]}"
                    )
                    st.write(
                        f"- Test Accuracy: {runs[runs['run_id'] == run_id]['metrics.test_accuracy'].iloc[0]:.4f}"
                    )

                # Dataset Section
                st.subheader("🗃️ Datasets")

                # List of dataset files to download
                if "small_dataset" in run_name:
                    dataset_files = [
                        ("datasets/original/mnist_train.pt", "training_data"),
                        ("datasets/original/mnist_test.pt", "test_data"),
                        (
                            "datasets/blurred/mnist_train_blurred.pt",
                            "blurred_training_data",
                        ),
                        ("datasets/blurred/mnist_test_blurred.pt", "blurred_test_data"),
                        ("datasets/limited/mnist_train.pt", "limited_training_data"),
                    ]
                else:
                    dataset_files = [
                        ("datasets/original/mnist_train.pt", "training_data"),
                        ("datasets/original/mnist_test.pt", "test_data"),
                        (
                            "datasets/blurred/mnist_train_blurred.pt",
                            "blurred_training_data",
                        ),
                        ("datasets/blurred/mnist_test_blurred.pt", "blurred_test_data"),
                    ]

                for artifact_path, name in dataset_files:
                    try:
                        # Download and create button for each dataset file
                        temp_path = client.download_artifacts(run_id, artifact_path)
                        dataset_info = get_dataset_info(temp_path)

                        with open(temp_path, "rb") as f:
                            file_bytes = f.read()

                        col1, col2 = st.columns([2, 1])

                        with col1:
                            st.download_button(
                                label=f"📥 Download {name}",
                                data=file_bytes,
                                file_name=f"{name}_{run_name}.pt",
                                mime="application/octet-stream",
                                key=f"{run_id}_{name}",
                            )

                        with col2:
                            if "error" in dataset_info:
                                st.write(f"⚠️ Error: {dataset_info['error']}")
                            else:
                                st.write(f"📊 Shape: {dataset_info['shape']}")
                                st.write(f"📈 Samples: {dataset_info['num_samples']}")
                                st.write(f"💾 Size: {dataset_info['size_mb']:.2f} MB")

                    except Exception as e:
                        st.warning(
                            f"Could not load {name} for run {run_name}: {str(e)}"
                        )

        except Exception as e:
            st.warning(f"Could not load artifacts for run {run_name}: {str(e)}")


def main():
    st.set_page_config(page_title="MNIST Experiment Tracker", layout="wide")
    st.title("MNIST Experiment Tracker")

    # Load experiment data
    runs = load_experiment_data()
    if runs is None or runs.empty:
        return

    # Get selected runs from sidebar
    selected_runs = get_selected_runs(runs)

    if len(selected_runs) == 0:
        st.warning("Please select at least one run from the sidebar.")
        return

    # Display different sections with selected runs
    display_config_comparison(selected_runs)
    plot_training_curves(selected_runs)
    display_metrics(selected_runs, "train", "Training Metrics")
    display_metrics(selected_runs, "val", "Validation Metrics")
    display_test_results(selected_runs)
    display_dataset_metadata(selected_runs)
    display_artifact_downloads(selected_runs)


if __name__ == "__main__":
    main()
