# Single Instance Training for Iris

import datetime
import os
import subprocess
import sys
import pandas as pd
import xgboost as xgb

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', dest='model_dir',
                    default=os.getenv('AIP_MODEL_DIR'), type=str, help='Model dir.')
args = parser.parse_args()

# Download data
iris_data_filename = 'iris_data.csv'
iris_target_filename = 'iris_target.csv'
data_dir = 'gs://cloud-samples-data/ai-platform/iris'

# gsutil outputs everything to stderr so we need to divert it to stdout.
subprocess.check_call(['gsutil', 'cp', os.path.join(data_dir,
                                                    iris_data_filename),
                       iris_data_filename], stderr=sys.stdout)
subprocess.check_call(['gsutil', 'cp', os.path.join(data_dir,
                                                    iris_target_filename),
                       iris_target_filename], stderr=sys.stdout)


# Load data into pandas, then use `.values` to get NumPy arrays
iris_data = pd.read_csv(iris_data_filename).values
iris_target = pd.read_csv(iris_target_filename).values

# Convert one-column 2D array into 1D array for use with XGBoost
iris_target = iris_target.reshape((iris_target.size,))


# Load data into DMatrix object
dtrain = xgb.DMatrix(iris_data, label=iris_target)


# Train XGBoost model
bst = xgb.train({}, dtrain, 20)

# Export the classifier to a file
model_filename = 'model.bst'
bst.save_model(model_filename)

# Upload the saved model file to Cloud Storage
gcs_model_path = os.path.join(args.model_dir, model_filename)
subprocess.check_call(['gsutil', 'cp', model_filename, gcs_model_path],
    stderr=sys.stdout)
