# FedFit: Federated Learning Framework for Personalized Fitness Activity Recognition

**A CNN-LSTM-based human activity recognition system with simulated federated learning using the WISDM dataset.**

## Overview

FedFit is a machine learning project designed to recognize human physical activities using smartphone or wearable sensor data. It combines **Convolutional Neural Networks (CNNs)** and **Long Short-Term Memory (LSTM)** networks to learn spatial and temporal patterns in sensor signals.

The project also implements a simulated federated learning workflow using **Federated Averaging (FedAvg)**. This demonstrates how multiple client models can contribute to a shared global model without directly combining their local training datasets.

FedFit includes an interactive Streamlit dashboard for exploring the project, demonstrating activity predictions, visualizing model analytics, and understanding the federated learning process.

## Key Features

- **Human Activity Recognition:** Predicts physical activities from time-series sensor data.
- **CNN-LSTM Architecture:** Combines feature extraction with temporal sequence learning.
- **Federated Learning:** Simulates multiple clients participating in collaborative model training.
- **FedAvg Aggregation:** Aggregates client model parameters into a global model.
- **Interactive Dashboard:** Provides project overview, activity prediction, model analytics, and federated learning visualizations.
- **Prediction Reports:** Supports viewing prediction results and downloading supported reports as CSV files.
- **Model Evaluation:** Uses accuracy, precision, recall, F1-score, and confusion-matrix analysis.
- **Local Data Processing:** Supports preprocessing, sequence preparation, model training, and evaluation workflows.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning and model training |
| CNN-LSTM | Time-series activity classification |
| Federated Averaging (FedAvg) | Global model aggregation |
| Pandas and NumPy | Data processing and numerical computation |
| Streamlit | Interactive web application |
| Matplotlib / visualization libraries | Results visualization, where used |
| Git and Git LFS | Version control and large-file management |
| WISDM Dataset | Sensor-based human activity recognition |

## Project Structure

```text
FedFit/
├── app.py
├── app_before_ui_upgrade.py
├── preprocess.py
├── prepare_data.py
├── train_model.py
├── federated_train.py
├── federated_train_backup.py
├── federated_train_before_validation.py
├── diagnose_federated.py
├── requirements.txt
├── sample_activity.csv
├── .gitattributes
├── .gitignore
│
├── data/
│   ├── WISDM_ar_v1.1/
│   ├── WISDM_ar_latest.tar.gz
│   ├── wisdm_clean.csv
│   └── processed/
│       ├── X.npy
│       ├── y.npy
│       ├── users.npy
│       └── label_mapping.csv
│
└── models/
    ├── fedfit_cnn_lstm.pt
    ├── fedfit_federated_cnn_lstm.pt
    ├── fedfit_federated_gentle_69_50.pt
    ├── fedfit_federated_weighted_68_50.pt
    ├── metrics.json
    ├── federated_metrics.json
    ├── federated_metrics_gentle_69_50.json
    └── federated_metrics_weighted_68_50.json
```

*Note: The structure above reflects the main files in the current project. Backup files are retained for development and comparison.*

## Dataset

FedFit uses the **WISDM (Wireless Sensor Data Mining) dataset**, which contains smartphone or wearable accelerometer measurements associated with physical activities.

The project processes sensor readings into numerical sequences suitable for CNN-LSTM training.

### Data workflow

1. Load the raw sensor dataset.
2. Clean and preprocess the sensor readings.
3. Encode activity labels.
4. Prepare numerical arrays and user identifiers.
5. Train a centralized CNN-LSTM baseline.
6. Simulate federated client training and aggregate model parameters.
7. Evaluate the models and display results in the dashboard.

The repository includes the dataset files and processed arrays. Large files are tracked using Git LFS.

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version supported by your installed dependencies
- Git
- Git LFS
- Visual Studio Code or another Python IDE

### 1. Clone the repository

```bash
git clone https://github.com/Samhith-Alavala/ALT_2420030250_TEAM8.git
cd ALT_2420030250_TEAM8
```

### 2. Download Git LFS files

Install Git LFS if necessary, then run:

```bash
git lfs install
git lfs pull
```

This retrieves the large dataset and model files tracked through Git LFS.

### 3. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell activation is restricted, use Git Bash:

```bash
source .venv/Scripts/activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ensure that `requirements.txt` lists all dependencies required by the application and training scripts. Install a compatible PyTorch build if it is not included.

## Running the Application

Launch the Streamlit dashboard from the project root:

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL, usually:

```text
http://localhost:8501
```

Open the URL in your browser.

### Dashboard sections

- **Overview:** Introduction to the FedFit framework.
- **Demo Mode:** Demonstration of activity recognition.
- **Activity Predictor:** Explore supported activity prediction workflows.
- **Model Analytics:** Examine available model evaluation results.
- **Federated Learning:** Explore the simulated federated training workflow.
- **About Project:** Review project details and methodology.

The availability of individual features depends on the data and model files present in the repository.

## Model Training

Run the following scripts from the project root, using the virtual environment.

### Data preprocessing

```bash
python preprocess.py
```

### Prepare model inputs

```bash
python prepare_data.py
```

### Train the centralized model

```bash
python train_model.py
```

### Run federated training

```bash
python federated_train.py
```

### Diagnose federated training

```bash
python diagnose_federated.py
```

**Important:** Some scripts depend on previously generated data files and model checkpoints. Check each script's configuration and expected input/output paths before rerunning it, because training may overwrite existing results.

## Experimental Results

The current development experiments produced the following results:

| Experiment | Metric | Result |
|---|---|---:|
| Centralized CNN-LSTM baseline | Held-out-user test accuracy | 81.15% |
| Federated CNN-LSTM experiment | Development accuracy | 69.50% |
| Federated CNN-LSTM experiment | Weighted F1-score | 0.632 |
| Federated CNN-LSTM experiment | Macro F1-score | 0.547 |

### Interpretation

- The centralized baseline achieved 81.15% accuracy on its held-out-user test.
- The federated experiment achieved a development accuracy of 69.50%.
- The difference indicates that the evaluated federated configuration did not match the centralized baseline's reported accuracy.
- The macro F1-score indicates that performance across activity classes requires further improvement.
- The federated experiment showed very low recall for the Upstairs activity, identifying a key area for model improvement.

**Evaluation note:** The centralized test and federated development scores are not a strictly equivalent comparison. Held-out users were used during federated model-selection experiments, so the 69.50% result should not be presented as an unbiased final test score.

Results may change when preprocessing, model parameters, client configurations, or training settings are modified.

## Federated Learning Methodology

FedFit demonstrates a simulated Federated Averaging workflow:

1. Initialize a global CNN-LSTM model.
2. Distribute the global model parameters to simulated clients.
3. Train the model locally on each client's assigned data.
4. Collect the client model parameters.
5. Aggregate the parameters using FedAvg.
6. Update the global model.
7. Repeat the process across communication rounds.
8. Evaluate the resulting model using the project's evaluation workflow.

This implementation is an educational simulation. It does not currently establish a production deployment with real distributed devices, secure aggregation, or differential privacy.

## Limitations

- Federated training is simulated on a single machine rather than across independent physical devices.
- The reported federated development score is not an unbiased final test result.
- Performance varies across activities, with particularly low recall for Upstairs in the reported experiment.
- Sensor data quality, activity imbalance, and differences between users can affect generalization.
- Secure aggregation, differential privacy, and production-scale communication have not been established in the current implementation.
- Results depend on compatible dependencies, model checkpoints, and correctly prepared input data.

## Future Enhancements

- Improve minority-class recognition and per-activity recall.
- Explore class weighting, resampling, and additional model architectures.
- Evaluate federated models using a strictly independent test set.
- Test different client counts, local epochs, and aggregation strategies.
- Add secure aggregation and differential privacy.
- Extend the simulation to multiple devices or distributed environments.
- Improve reproducibility through configuration files and experiment tracking.
- Expand dashboard visualizations and model-comparison reports.
- Investigate deployment on wearable or mobile devices.

## Reproducibility

For consistent experiments:

- Use a documented Python environment.
- Record package versions and random seeds.
- Keep training, validation, and test users separated.
- Preserve evaluation metrics for every experiment.
- Avoid selecting models using the final test set.
- Verify that the required Git LFS files have been downloaded.

## Author and Repository

**Project:** FedFit — Federated Learning Framework for Personalized Fitness Activity Recognition

**GitHub Repository:**  
https://github.com/Samhith-Alavala/ALT_2420030250_TEAM8

## License

No license has been specified yet. Add an appropriate `LICENSE` file before distributing the project under an open-source license.

## Acknowledgements

This project uses the WISDM dataset for sensor-based activity recognition and explores deep learning and federated learning techniques for personalized fitness activity classification.
