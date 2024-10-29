import os
os.system('clear')
import argparse
import torch
import torch.nn as nn
import time
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import IntegerType

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims):
        super(MLPClassifier, self).__init__()
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, num_classes))  # Output layer with `num_classes` units
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Compute logits
        logits = self.model(x)
        # Get predicted class using argmax
        predicted_classes = torch.argmax(logits, dim=1)
        return predicted_classes
    

# Define a pandas UDF for parallel classification
@pandas_udf(IntegerType())
def MLPClassifier_udf(*batch_inputs):
    # The * symbol before batch_inputs is a way to collect any number of positional arguments into a single tuple.

    raise NotImplementedError("This function is not implemented yet.")


if __name__=="__main__":
    # Set up argument parsing
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Edit Distance with PySpark")
    parser.add_argument('--n_input', type=int, default=10000, help="Number of sentences")
    parser.add_argument('--hidden_dim', type=int, default=1024, help="hidden_dim")
    parser.add_argument('--hidden_layer', type=int, default=50, help="hidden_layer")
    args = parser.parse_args()

    # Configuration
    input_dim = 128  # Input dimension
    num_classes = 10  # Number of classes
    hidden_dims = [args.hidden_dim * args.hidden_layer]  # Hidden layer sizes
    # Model and input setup
    mlp_model = MLPClassifier(input_dim, num_classes, hidden_dims)
    x = torch.randn(args.n_input, input_dim)  # A random input vector of dimension n

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Spark version

    # Initialize Spark session

    # Convert Pandas DataFrame to Spark DataFrame

    # Apply the UDF to perform distributed classification
    start_time = time.time()
    
    
    end_time = time.time()

    print(f"Time taken for distributed classification: {end_time - start_time:.6f} seconds")

    # Stop Spark session


    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Non-spark version

    # Timing the forward pass
    start_time = time.time()
    output = mlp_model(x)
    end_time = time.time()

    # Output and timing results
    print(f"Output: {output.shape}")
    print(f"Time taken for forward pass: {end_time - start_time:.6f} seconds")
