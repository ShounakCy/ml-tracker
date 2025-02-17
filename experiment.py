import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import datasets, transforms
from skimage.filters import gaussian
import numpy as np
import os
from pathlib import Path
import mlflow
import yaml
import uuid
from datetime import datetime
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec


def load_dataset(data_path, max_samples=None):
    """Load dataset with optional sample limiting"""
    tensors, labels = torch.load(str(data_path))
    if max_samples:
        tensors = tensors[:max_samples]
        labels = labels[:max_samples]
    return TensorDataset(tensors, labels)


def save_dataset(path, dataset):
    """Save dataset tensors"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    images, labels = dataset.tensors
    torch.save((images, labels), str(path))


def create_blurred_dataset(input_path, output_path, sigma=2):
    """Create and save blurred version of dataset"""
    dataset = load_dataset(input_path)
    images, labels = dataset.tensors
    blurred_images = np.empty_like(images)

    for i in range(images.shape[0]):
        blurred_images[i, 0] = gaussian(images[i, 0], sigma=sigma, mode="reflect")

    blurred_dataset = TensorDataset(torch.Tensor(blurred_images), labels)
    save_dataset(output_path, blurred_dataset)
    return blurred_dataset


class SimpleNN(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.softmax(self.fc2(self.relu(self.fc1(x))))


class ExperimentRunner:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Setup MLflow
        mlflow.set_tracking_uri(self.config["mlflow"]["backend_store_uri"])
        experiment = mlflow.get_experiment_by_name(
            self.config["mlflow"]["experiment_name"]
        )
        if experiment is None:
            mlflow.create_experiment(
                self.config["mlflow"]["experiment_name"],
                artifact_location=self.config["mlflow"]["artifact_location"],
            )
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])

        # Create paths
        self.root_path = Path(self.config["paths"]["root"])
        self.root_path.mkdir(exist_ok=True)

    def prepare_mnist_datasets(self):
        """Download and prepare MNIST datasets"""
        train_path = self.root_path / self.config["paths"]["train_data"]
        test_path = self.root_path / self.config["paths"]["test_data"]

        if not (train_path.exists() and test_path.exists()):
            train_dataset = datasets.MNIST(
                root=self.root_path,
                train=True,
                transform=transforms.ToTensor(),
                download=True,
            )
            test_dataset = datasets.MNIST(
                root=self.root_path,
                train=False,
                transform=transforms.ToTensor(),
                download=True,
            )

            # Create and save training dataset
            train_tensors = torch.stack([img[0] for img in train_dataset])
            train_labels = torch.tensor([label for _, label in train_dataset])
            train_dataset = TensorDataset(train_tensors, train_labels)
            save_dataset(train_path, train_dataset)

            # Create and save test dataset
            test_tensors = torch.stack([img[0] for img in test_dataset])
            test_labels = torch.tensor([label for _, label in test_dataset])
            test_dataset = TensorDataset(test_tensors, test_labels)
            save_dataset(test_path, test_dataset)

        return train_path, test_path

    def train_epoch(self, model, dataloader, criterion, optimizer):
        """Train the model for one epoch"""
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        return {"loss": running_loss / total, "accuracy": correct / total}

    def evaluate(self, model, dataloader, criterion):
        """Evaluate the model"""
        model.eval()
        running_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return {"loss": running_loss / total, "accuracy": correct / total}

    def get_dataset_metadata(self, dataset, file_path):
        """Get metadata for a dataset"""
        images, labels = dataset.tensors
        return {
            "size": len(dataset),
            "image_shape": list(images.shape[1:]),
            "num_classes": len(torch.unique(labels)),
            "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
            "class_distribution": {
                int(label): int((labels == label).sum())
                for label in torch.unique(labels)
            },
        }

    def run_experiment(self, model_config_name):
        """Run complete experiment: data prep, training, and testing"""
        exp_id = str(uuid.uuid4())[:8]
        model_config = self.config["model_configs"][model_config_name]
        run_name = f"{model_config['name']}_{exp_id}"

        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_params(model_config)
            mlflow.log_param("timestamp", datetime.now().isoformat())

            print(f"Step 1: Preparing datasets for {run_name}...")

            # Prepare original datasets
            train_path, test_path = self.prepare_mnist_datasets()

            # Load datasets
            train_dataset = load_dataset(train_path, model_config.get("max_samples"))
            test_dataset = load_dataset(test_path)

            # Create and save limited dataset if max_samples specified
            if model_config.get("max_samples"):
                limited_train_path = (
                    self.root_path
                    / "datasets"
                    / "limited"
                    / self.config["paths"]["train_data"]
                )
                limited_train_path.parent.mkdir(parents=True, exist_ok=True)
                save_dataset(limited_train_path, train_dataset)
                mlflow.log_artifact(limited_train_path, "datasets/limited")

            # Create blurred datasets
            print("Creating blurred datasets...")
            train_blurred_path = (
                self.root_path / self.config["paths"]["train_data_blurred"]
            )
            test_blurred_path = (
                self.root_path / self.config["paths"]["test_data_blurred"]
            )

            if not (train_blurred_path.exists() and test_blurred_path.exists()):
                create_blurred_dataset(
                    train_path,
                    train_blurred_path,
                    sigma=self.config["data_prep"]["blur_sigma"],
                )
                create_blurred_dataset(
                    test_path,
                    test_blurred_path,
                    sigma=self.config["data_prep"]["blur_sigma"],
                )

            # Load blurred datasets
            blurred_test_dataset = load_dataset(test_blurred_path)

            # Log artifacts
            mlflow.log_artifact(train_path, "datasets/original")
            mlflow.log_artifact(test_path, "datasets/original")
            mlflow.log_artifact(train_blurred_path, "datasets/blurred")
            mlflow.log_artifact(test_blurred_path, "datasets/blurred")

            # Prepare dataset metadata
            dataset_metadata = {
                "original": self.get_dataset_metadata(
                    load_dataset(train_path), train_path
                ),
                "blurred": self.get_dataset_metadata(
                    load_dataset(train_blurred_path), train_blurred_path
                ),
            }

            if model_config.get("max_samples"):
                dataset_metadata["limited"] = self.get_dataset_metadata(
                    train_dataset, limited_train_path
                )

            mlflow.log_dict(dataset_metadata, "dataset_metadata.json")

            # Split training data
            train_size = int(model_config["data_split_ratio"] * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_subset, val_subset = random_split(
                train_dataset, [train_size, val_size]
            )

            # Create data loaders
            train_loader = DataLoader(
                train_subset, batch_size=model_config["batch_size"], shuffle=True
            )
            val_loader = DataLoader(
                val_subset, batch_size=model_config["batch_size"], shuffle=False
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config["testing"]["batch_size"],
                shuffle=False,
            )
            blurred_test_loader = DataLoader(
                blurred_test_dataset,
                batch_size=self.config["testing"]["batch_size"],
                shuffle=False,
            )

            print("\nStep 2: Training model...")
            model = SimpleNN(hidden_size=model_config["hidden_size"])
            criterion = nn.NLLLoss()
            optimizer = optim.SGD(model.parameters(), lr=model_config["learning_rate"])

            # Training loop
            for epoch in range(model_config["max_epochs"]):
                train_metrics = self.train_epoch(
                    model, train_loader, criterion, optimizer
                )
                val_metrics = self.evaluate(model, val_loader, criterion)

                mlflow.log_metrics(
                    {
                        "train_loss": train_metrics["loss"],
                        "train_accuracy": train_metrics["accuracy"],
                        "val_loss": val_metrics["loss"],
                        "val_accuracy": val_metrics["accuracy"],
                    },
                    step=epoch,
                )

                print(
                    f"Epoch [{epoch+1}/{model_config['max_epochs']}], "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.4f}"
                )

            print("\nStep 3: Testing model...")
            # Test on original data
            test_metrics = self.evaluate(model, test_loader, criterion)
            mlflow.log_metrics(
                {
                    "test_loss": test_metrics["loss"],
                    "test_accuracy": test_metrics["accuracy"],
                }
            )

            print("Original Test Data Results:")
            print(f"Test Loss: {test_metrics['loss']:.4f}")
            print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")

            # Test on blurred data
            blurred_test_metrics = self.evaluate(model, blurred_test_loader, criterion)
            mlflow.log_metrics(
                {
                    "test_blurred_loss": blurred_test_metrics["loss"],
                    "test_blurred_accuracy": blurred_test_metrics["accuracy"],
                }
            )

            print("\nBlurred Test Data Results:")
            print(f"Test Loss: {blurred_test_metrics['loss']:.4f}")
            print(f"Test Accuracy: {blurred_test_metrics['accuracy']:.4f}")

            # Log model
            input_schema = Schema(
                [TensorSpec(np.dtype(np.float32), (-1, 28, 28), name="images")]
            )
            output_schema = Schema(
                [TensorSpec(np.dtype(np.float32), (-1, 10), name="output")]
            )
            signature = ModelSignature(inputs=input_schema, outputs=output_schema)
            mlflow.pytorch.log_model(model, "model", signature=signature)


def main():
    runner = ExperimentRunner("config.yaml")

    # Run each model configuration
    for model_config_name in runner.config["model_configs"]:
        print(f"\nRunning experiment for {model_config_name} configuration")
        print("=" * 50)
        runner.run_experiment(model_config_name)


if __name__ == "__main__":
    main()
