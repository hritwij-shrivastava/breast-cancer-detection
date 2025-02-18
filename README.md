# Breast Cancer Detection

This project demonstrates how machine learning can assist in detecting breast cancer using various ML models. The dataset is sourced from the UCI repository and preprocessed to enable effective analysis and modeling.

## Overview

1. **Data Preprocessing**:
   - Imported data from the UCI repository.
   - Renamed columns (features) and organized them into a pandas DataFrame.
   - Removed the ID column for better analysis.
   - Used the `describe()` function to explore statistical features such as mean, max, min, and quartiles.
   - Created histograms to understand feature distributions.
   - Plotted a scatterplot matrix to identify potential linear relationships between variables.

2. **Models**:
   - Implemented machine learning models for breast cancer detection, including K-Nearest Neighbors (KNN) and Support Vector Machines (SVM).

## Prerequisites

- Python 3.x
- `virtualenv`
- `mlflow`

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd breast-cancer-detection
   ```

2. Create and activate a virtual environment:
   ```bash
   pip3 install virtualenv
   virtualenv env
   ```

3. Install required packages:
   ```bash
   pip install --upgrade mlflow
   ```

## Running the Application

1. Start the MLflow server in one terminal:
   ```bash
   mlflow server --host 127.0.0.1 --port 8080
   ```

2. Activate the virtual environment in another terminal:
   - For Windows PowerShell:
     ```bash
     .\env\Scripts\activate.ps1
     ```
   - For other terminals:
     ```bash
     source env/bin/activate
     ```

3. Install necessary libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the scripts:
   ```bash
   pip install -r requirements.txt
   python .\datasets.py
   python .\breast_cancer_detection_with_knn.py
   python .\breast_cancer_detection_with_svm.py
   ```

## Features

- Data exploration and visualization for better understanding of features.
- KNN and SVM implementations for cancer detection.
- MLflow integration for tracking experiments.

## Dataset

The dataset is sourced from the UCI repository and includes features relevant to breast cancer detection.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributions

Contributions are welcome! Feel free to submit issues or pull requests.

## Contact

For any questions or suggestions, please reach out to [Your Name/Email].