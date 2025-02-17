# MNIST Experiment Tracker

A machine learning experiment system for MNIST digit classification with MLflow tracking and Streamlit visualization.



## Folder Structure

```bash
ml-tracker/
├── app.py                # Streamlit dashboard application
├── experiment.py         # Script to run MNIST training experiments
├── manage_mlflow.py      # Script to start the MLflow tracking server
├── config.yaml           # Configuration settings
├── mlflow.db             # SQLite database for MLflow tracking
├── mlruns/               # Directory where MLflow stores run data ( artifacts, models, dataset)
├── README.md             # Documentation
├── requirements.txt      # Project dependencies
└── data/                 # Directory to store the datasets
    ├── MNIST/
    │   ├── raw/
    │   │   ├── images
    │   │   ├── labels
    ├── mnist_train.pt
    ├── mnist_test.pt
    ├── mnist_train_blurred.pt
    └── mnist_test_blurred.pt
```
    
## Quick Start Guide

```bash
Python 3.9.16 used.
```

```bash
git clone https://github.com/ShounakCy/ml-tracker.git
cd ml-traker
```
### 0. Create virtual env

```bash
python3 -m venv mlenv
source mlenv/bin/activate
```
 or 
 
```bash
conda create --name mlenv python=3.9.16
conda activate mlenv
```
### 1. Install Dependencies

First, install all required packages:
```bash
pip install -r requirements.txt
```

### 2. Running the Project

Follow these steps in order:

#### Step 1: Start MLflow Server
```bash
python manage_mlflow.py
```
- This starts the MLflow tracking server (check the server link in the terminal)
- Keep this terminal window open
- Press Ctrl+C when you want to stop the server

#### Step 2: Run the Experiments
Open a new terminal and run:
```bash
python experiment.py
```
This will:
- Download the MNIST dataset (first run only)
- Create blurred versions of the dataset
- Train four different model configurations:
  - default (128 hidden units)
  - small_hidden (2 hidden units)
  - small_dataset (limited samples)
  - blurred_dataset (trained on blurred images)

#### Step 3: Launch the Dashboard
Open a new terminal and run:
```bash
streamlit run app.py
```
- The dashboard will open in your default browser (check the web app link in the terminal)
- If it doesn't open automatically, check the terminal for the correct URL


### Configuration

Modify `config.yaml` to adjust:

- Model architecture parameters
- Training hyperparameters
- Dataset options

### Dataset Information

The project supports both standard MNIST and blurred variants:

- mnist_train.pt: Standard MNIST training set
- mnist_test.pt: Standard MNIST test set
- mnist_train_blurred.pt: Blurred MNIST training set
- mnist_test_blurred.pt: Blurred MNIST test set

### MLflow Integration

MLflow provides a more comprehensive, specialized, and user-friendly solution that is designed for managing the entire machine learning lifecycle. Experiments are tracked using MLflow, storing:

- Training metrics (accuracy, loss)
- Model parameters
- Datasets
- Model artifacts
- Training configurations

#### Underlying mlflow db schema

```bash
CREATE TABLE experiments (
            experiment_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            artifact_location TEXT,
            lifecycle_stage TEXT NOT NULL
        )
CREATE TABLE runs (
            run_uuid TEXT PRIMARY KEY,
            experiment_id INTEGER,
            name TEXT,
            source_type INTEGER,
            source_name TEXT,
            entry_point_name TEXT,
            user_id TEXT,
            status INTEGER,
            start_time BIGINT,
            end_time BIGINT,
            source_version TEXT,
            lifecycle_stage TEXT,
            artifact_uri TEXT,
            CONSTRAINT experiment_id_fk FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
        )
CREATE TABLE metrics (
            key TEXT NOT NULL,
            value REAL NOT NULL,
            timestamp INTEGER NOT NULL,
            run_uuid TEXT NOT NULL,
            step INTEGER NOT NULL,
            FOREIGN KEY (run_uuid) REFERENCES runs (run_uuid),
            PRIMARY KEY (key, timestamp, run_uuid)
        )
CREATE TABLE params (
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            run_uuid TEXT NOT NULL,
            FOREIGN KEY (run_uuid) REFERENCES runs (run_uuid),
            PRIMARY KEY (key, run_uuid)
        )
CREATE TABLE tags (
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            run_uuid TEXT NOT NULL,
            FOREIGN KEY (run_uuid) REFERENCES runs (run_uuid),
            PRIMARY KEY (key, run_uuid)
        )
CREATE TABLE alembic_version (
            version_num VARCHAR(32) NOT NULL, 
            CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
)
```

### Dashboard Usage

1. Select experiments to compare in the sidebar
2. View:
   - Training curves
   - Model configurations
   - Performance metrics
   - Dataset information
3. Download model checkpoints and datasets as needed

## Troubleshooting

Common issues and solutions:

1. **MLflow server port in use**
   - The script will automatically try the next available port
   - Check the terminal output for the actual port number

2. **Database errors**
   - Delete `mlflow.db` and `mlruns` folder to start fresh
   - Restart the MLflow server

## Implementation without MLFlow

```bash
ml-tracker/without-mlflow/
```


## Future Work

- Provide hyperparameter optimization option on the dashboard
- Automated Model Deployment
- Real-time monitoring on the dashboard
- Data storage in cloud service
- Display best and work evaluation result images in the dashboard
