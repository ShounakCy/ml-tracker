# MNIST Experiment Tracker

A machine learning experiment system for MNIST digit classification with MLflow tracking and Streamlit visualization.

## Folder Structure

```bash
ml-tracker/without-mlflow
├── setup.py                # Streamlit dashboard application
├── app.py                # Streamlit dashboard application
├── experiment.py         # Script to run MNIST training experiments
├── config.yaml           # Configuration settings
├── README.md             # Documentation
├── requirements.txt      # Project dependencies
└── data/                 # Directory to store the datasets
```
    
## Quick Start Guide

```bash
Python 3.9.16 used.
```

```bash
git clone https://github.com/ShounakCy/ml-tracker.git
cd ml-traker/without-mlflow
```
### 0. Create virtual env

```bash
python3 -m venv venv
source venv/bin/activate
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

#### Step 1: Setup DB
```bash
python setup.py

psql <database_name> 

experiment_tracking_new=# \dt
                    List of relations
 Schema |        Name        | Type  |       Owner        
--------+--------------------+-------+--------------------
 public | artifacts          | table | shounakchakraborty
 public | evaluation_metrics | table | shounakchakraborty
 public | experiments        | table | shounakchakraborty
 public | training_metrics   | table | shounakchakraborty
(4 rows)

experiment_tracking_new=# SELECT experiment_id, description, tags FROM experiments ORDER BY completed_at DESC limit 10;
 experiment_id |                     description                     |                      tags                      
---------------+-----------------------------------------------------+------------------------------------------------
             4 | MNIST experiment with blurred_dataset configuration | {"dataset": "MNIST", "model_type": "SimpleNN"}
             3 | MNIST experiment with small_dataset configuration   | {"dataset": "MNIST", "model_type": "SimpleNN"}
             2 | MNIST experiment with small_hidden configuration    | {"dataset": "MNIST", "model_type": "SimpleNN"}
             1 | MNIST experiment with default configuration         | {"dataset": "MNIST", "model_type": "SimpleNN"}
(4 rows)

experiment_tracking_new=# SELECT * FROM training_metrics ORDER BY created_at DESC limit 10;
 metric_id | experiment_id | epoch |                                                                    metrics                                                                     |         created_at         
-----------+---------------+-------+------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------
        40 |             4 |     9 | {"val_loss": 0.28288472910722096, "train_loss": 0.2773536302447319, "val_accuracy": 0.9195, "train_accuracy": 0.9215833333333333}              | 2025-02-17 18:52:07.336941
        39 |             4 |     8 | {"val_loss": 0.29213269712527595, "train_loss": 0.28764003205299377, "val_accuracy": 0.9166666666666666, "train_accuracy": 0.9191666666666667} | 2025-02-17 18:52:06.837821
        38 |             4 |     7 | {"val_loss": 0.3032727531790733, "train_loss": 0.298986409385999, "val_accuracy": 0.9135833333333333, "train_accuracy": 0.9148125}             | 2025-02-17 18:52:06.341209
        37 |             4 |     6 | {"val_loss": 0.31346589225530624, "train_loss": 0.3122797297636668, "val_accuracy": 0.9108333333333334, "train_accuracy": 0.910625}            | 2025-02-17 18:52:05.837831
        36 |             4 |     5 | {"val_loss": 0.3262250948746999, "train_loss": 0.327757839034001, "val_accuracy": 0.9075833333333333, "train_accuracy": 0.90725}               | 2025-02-17 18:52:05.341816
        35 |             4 |     4 | {"val_loss": 0.34223429425557456, "train_loss": 0.3473828511536121, "val_accuracy": 0.9036666666666666, "train_accuracy": 0.9023541666666667}  | 2025-02-17 18:52:04.850613
        34 |             4 |     3 | {"val_loss": 0.3650995638370514, "train_loss": 0.37598000425100325, "val_accuracy": 0.89825, "train_accuracy": 0.8958958333333333}             | 2025-02-17 18:52:04.35154
        33 |             4 |     2 | {"val_loss": 0.3982979506254196, "train_loss": 0.4268189262946447, "val_accuracy": 0.8916666666666667, "train_accuracy": 0.8860416666666666}   | 2025-02-17 18:52:03.80757
        32 |             4 |     1 | {"val_loss": 0.4702480628490448, "train_loss": 0.5601456625064214, "val_accuracy": 0.8810833333333333, "train_accuracy": 0.8640833333333333}   | 2025-02-17 18:52:03.279578
        31 |             4 |     0 | {"val_loss": 0.717241516828537, "train_loss": 1.379305993159612, "val_accuracy": 0.8425833333333334, "train_accuracy": 0.709875}               | 2025-02-17 18:52:02.783493
```


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


### Dashboard Usage

1. Select experiments to compare in the sidebar
2. View:
   - Training curves
   - Model configurations
   - Performance metrics
   - Dataset information
3. Download model checkpoints and datasets as needed