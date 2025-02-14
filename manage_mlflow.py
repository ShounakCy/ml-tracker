# manage_mlflow.py
import subprocess
import socket
import yaml


def is_port_available(port):
    """Check if a port is available"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0


def start_mlflow(port=5000):
    """Start MLflow server"""

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    backed_store_uri = config["mlflow"]["backend_store_uri"]
    default_artifact_root = config["mlflow"]["artifact_location"]
    # Find available port
    while not is_port_available(port):
        print(f"Port {port} in use, trying next port...")
        port += 1

    print(f"Starting MLflow server on port {port}")

    # Start server
    cmd = [
        "mlflow",
        "ui",
        "--backend-store-uri",
        backed_store_uri,
        "--default-artifact-root",
        default_artifact_root,
        "--host",
        "localhost",
        "--port",
        str(port),
    ]

    return subprocess.Popen(cmd)


if __name__ == "__main__":
    try:
        # Start server
        process = start_mlflow()
        print("MLflow server is running. Press Ctrl+C to stop.")

        # Keep script running
        process.wait()

    except KeyboardInterrupt:
        # Handle Ctrl+C
        print("\nStopping MLflow server...")
        process.terminate()
        process.wait()
