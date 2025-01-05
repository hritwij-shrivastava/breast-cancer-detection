import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

mlflow.set_tracking_uri("http://localhost:8080")
mlflow.set_experiment("breast_cancer_detection")

# Load original dataset
dataset_source_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
names = ['id', 'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
         'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
         'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']

raw_data = pd.read_csv(dataset_source_url, names=names)

# Log the original dataset
with mlflow.start_run() as original_run:
    original_run_id = original_run.info.run_id  # Save original run ID for lineage tracking
    mlflow.log_param("dataset_name", "breast_cancer_data")
    mlflow.log_param("source_url", dataset_source_url)
    raw_data.to_csv("original_breast_cancer_data.csv", index=False)
    mlflow.log_artifact("original_breast_cancer_data.csv", artifact_path="datasets")

# Preprocess the data
df = raw_data.copy()
df.replace('?', -99999, inplace=True)  # Replace missing values
df.drop(['id'], axis=1, inplace=True)  # Drop ID column

# Save the cleaned dataset locally
cleaned_dataset_path = "breast_cancer_data_cleaned.csv"
df.to_csv(cleaned_dataset_path, index=False)

# Log the cleaned dataset and add lineage metadata
with mlflow.start_run() as cleaned_run:
    mlflow.log_param("dataset_name", "breast_cancer_data_cleaned")
    mlflow.log_param("source_run_id", original_run_id)  # Reference original dataset run ID
    mlflow.log_artifact(cleaned_dataset_path, artifact_path="datasets")

    # Log dataset statistics
    mlflow.log_metric("row_count", df.shape[0])
    mlflow.log_metric("column_count", df.shape[1])

    # Visualize dataset
    df.hist(figsize=(10, 10))
    plt.savefig("histograms.png")
    mlflow.log_artifact("histograms.png", artifact_path="visualizations")

    scatter_matrix(df, figsize=(18, 18))
    plt.savefig("scatter_matrix.png")
    mlflow.log_artifact("scatter_matrix.png", artifact_path="visualizations")

print("Cleaned dataset logged with lineage reference to the original dataset.")
