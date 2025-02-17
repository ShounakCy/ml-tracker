import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import datasets, transforms
from skimage.filters import gaussian
import numpy as np
import os
from pathlib import Path
import yaml
import uuid
from datetime import datetime
import json
import psycopg2
from psycopg2.extras import Json

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

class ExperimentRunner:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Connect to database
        self.conn = psycopg2.connect(
            dbname=self.config['database']['name'],
            user=self.config['database']['user'],
            password=self.config['database']['password'],
            host=self.config['database']['host']
        )
        
        # Create root path
        self.root_path = Path(self.config["paths"]["root"])
        self.root_path.mkdir(exist_ok=True, parents=True)
        
        # Validate and create necessary directories
        self.validate_directories()
        
    def validate_directories(self):
        """Validate that necessary directories exist and create them if not."""
        paths_to_check = [
            self.root_path / self.config["paths"]["train_data"],
            self.root_path / self.config["paths"]["test_data"],
            self.root_path / self.config["paths"]["train_data_blurred"],
            self.root_path / self.config["paths"]["test_data_blurred"],
            self.root_path / "models"
        ]
        for path in paths_to_check:
            path.parent.mkdir(parents=True, exist_ok=True)

    def log_experiment(self, name, config, description=None, tags=None):
        """Log a new experiment to the database"""
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO experiments (name, config, description, tags)
            VALUES (%s, %s, %s, %s)
            RETURNING experiment_id
        """, (name, Json(config), description, Json(tags) if tags else None))
        experiment_id = cur.fetchone()[0]
        self.conn.commit()
        cur.close()
        return experiment_id

    def log_metrics(self, experiment_id, metrics, epoch=None):
        """Log metrics to the database"""
        cur = self.conn.cursor()
        if epoch is not None:
            # Training metrics
            cur.execute("""
                INSERT INTO training_metrics (experiment_id, epoch, metrics)
                VALUES (%s, %s, %s)
                ON CONFLICT (experiment_id, epoch) 
                DO UPDATE SET metrics = EXCLUDED.metrics
            """, (experiment_id, epoch, Json(metrics)))
        else:
            # Evaluation metrics
            cur.execute("""
                INSERT INTO evaluation_metrics (experiment_id, metrics)
                VALUES (%s, %s)
            """, (experiment_id, Json(metrics)))
        self.conn.commit()
        cur.close()

    def log_artifact(self, experiment_id, name, artifact_type, file_path, metadata=None):
        """Log artifact information to the database"""
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO artifacts (experiment_id, name, type, file_path, metadata)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (experiment_id, name) 
            DO UPDATE SET file_path = EXCLUDED.file_path, metadata = EXCLUDED.metadata
        """, (experiment_id, name, artifact_type, str(file_path), Json(metadata) if metadata else None))
        self.conn.commit()
        cur.close()

    def update_experiment_status(self, experiment_id, status, duration_seconds=None):
        """Update experiment status and duration"""
        cur = self.conn.cursor()
        cur.execute("""
            UPDATE experiments 
            SET status = %s, 
                duration_seconds = %s,
                completed_at = CASE WHEN %s = 'completed' THEN CURRENT_TIMESTAMP ELSE completed_at END
            WHERE experiment_id = %s
        """, (status, duration_seconds, status, experiment_id))
        self.conn.commit()
        cur.close()

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
        start_time = datetime.now()
        exp_id = str(uuid.uuid4())[:8]
        model_config = self.config["model_configs"][model_config_name]
        run_name = f"{model_config['name']}_{exp_id}"

        # Log experiment start
        experiment_id = self.log_experiment(
            name=run_name,
            config=model_config,
            description=f"MNIST experiment with {model_config_name} configuration",
            tags={"model_type": "SimpleNN", "dataset": "MNIST"}
        )

        try:
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
                self.log_artifact(
                    experiment_id,
                    "limited_dataset",
                    "dataset",
                    limited_train_path,
                    self.get_dataset_metadata(train_dataset, limited_train_path)
                )
            
            # Create blurred datasets
            print("Creating datasets...")
            train_blurred_path = self.root_path / self.config["paths"]["train_data_blurred"]
            test_blurred_path = self.root_path / self.config["paths"]["test_data_blurred"]

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

            # Log dataset artifacts
            for path, name in [
                (train_path, "original_train"),
                (test_path, "original_test"),
                (train_blurred_path, "blurred_train"),
                (test_blurred_path, "blurred_test")
            ]:
                self.log_artifact(
                    experiment_id,
                    name,
                    "dataset",
                    path,
                    self.get_dataset_metadata(load_dataset(path), path)
                )

            # Split training data
            train_size = int(model_config["data_split_ratio"] * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

            # Create data loaders
            train_loader = DataLoader(
                train_subset, batch_size=model_config["batch_size"], shuffle=True
            )
            val_loader = DataLoader(
                val_subset, batch_size=model_config["batch_size"], shuffle=False
            )
            test_loader = DataLoader(
                test_dataset, batch_size=self.config["testing"]["batch_size"], shuffle=False
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
                train_metrics = self.train_epoch(model, train_loader, criterion, optimizer)
                val_metrics = self.evaluate(model, val_loader, criterion)

                # Log training metrics
                self.log_metrics(
                    experiment_id,
                    {
                        "train_loss": train_metrics["loss"],
                        "train_accuracy": train_metrics["accuracy"],
                        "val_loss": val_metrics["loss"],
                        "val_accuracy": val_metrics["accuracy"]
                    },
                    epoch=epoch
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
            blurred_test_metrics = self.evaluate(model, blurred_test_loader, criterion)

            # Log evaluation metrics
            self.log_metrics(
                experiment_id,
                {
                    "test_loss": test_metrics["loss"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_blurred_loss": blurred_test_metrics["loss"],
                    "test_blurred_accuracy": blurred_test_metrics["accuracy"]
                }
            )

            # Save model
            model_path = self.root_path / "models" / f"{run_name}.pth"
            model_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(model.state_dict(), model_path)
            self.log_artifact(
                experiment_id,
                "model_checkpoint",
                "model",
                model_path,
                {"hidden_size": model_config["hidden_size"]}
            )

            # Update experiment status
            duration = (datetime.now() - start_time).total_seconds()
            self.update_experiment_status(experiment_id, "completed", duration)

            print("\nExperiment completed successfully!")
            print(f"Duration: {duration:.2f} seconds")
            print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"Blurred Test Accuracy: {blurred_test_metrics['accuracy']:.4f}")

        except Exception as e:
            print(f"Error during experiment: {str(e)}")
            self.update_experiment_status(experiment_id, "failed")
            raise e

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

def main():
    runner = ExperimentRunner("config.yaml")
    try:
        # Run each model configuration
        for model_config_name in runner.config["model_configs"]:
            print(f"\nRunning experiment for {model_config_name} configuration")
            print("=" * 50)
            runner.run_experiment(model_config_name)
    finally:
        runner.close()

if __name__ == "__main__":
    main()
