# -*- coding: utf-8 -*-
"""Breast Cancer Detection with KNN"""

# Import necessary libraries
import sys
import scipy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error
from pandas.plotting import scatter_matrix
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:8080")
mlflow.set_experiment("breast_cancer_detection")

# Start MLflow run
with mlflow.start_run():
    # Search for runs with the parameter "dataset_name" set to "breast_cancer_data_cleaned"
    runs = mlflow.search_runs(
        filter_string="params.dataset_name = 'breast_cancer_data_cleaned'",  # Search by parameter
        order_by=["start_time DESC"],  # Get the latest run
        max_results=1
    )

    if runs.empty:
        raise ValueError("No runs found with the parameter 'dataset_name = breast_cancer_data_cleaned'.")

    # Get the run ID of the matching run
    run_id = runs.iloc[0]["run_id"]

    # Define the artifact path (relative to the logged run)
    artifact_relative_path = "datasets/breast_cancer_data_cleaned.csv"

    # Download the artifact locally
    artifact_local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=artifact_relative_path
    )

    # Load the dataset into a pandas DataFrame
    df = pd.read_csv(artifact_local_path)

    # Create X and Y datasets for training
    X = np.array(df.drop(['class'], axis=1))
    y = np.array(df['class'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SVC()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mlflow.log_metrics({"accuracy":accuracy_score(y_test, y_pred), "mse": mean_squared_error(y_test, y_pred)})

    # Log the sklearn model and register as version 1
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="sklearn-model",
        registered_model_name="SVM",
    )
