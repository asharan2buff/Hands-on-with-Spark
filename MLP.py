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
import warnings
warnings.filterwarnings("ignore")

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
        logits = self.model(x)
        predicted_classes = torch.argmax(logits, dim=1)
        return predicted_classes

# Define a pandas UDF for parallel classification
def create_mlp_udf(input_dim, num_classes, hidden_dims):
    model = MLPClassifier(input_dim=input_dim, num_classes=num_classes, hidden_dims=hidden_dims)
    model.eval()

    @pandas_udf(IntegerType())
    def MLPClassifier_udf(*batch_inputs):
        # Convert inputs to a tensor
        batch_tensor = torch.stack([torch.tensor(col, dtype=torch.float32) for col in batch_inputs], dim=1)
        with torch.no_grad():
            model_output = model(batch_tensor)
        return pd.Series(model_output.numpy().tolist())

    return MLPClassifier_udf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLP Classification with PySpark")
    parser.add_argument('--n_input', type=int, default=10000, help="Number of inputs")
    parser.add_argument('--hidden_dim', type=int, default=1024, help="Hidden dimension size")
    parser.add_argument('--hidden_layer', type=int, default=50, help="Number of hidden layers")
    args = parser.parse_args()

    input_dim = 128  # Input dimension
    num_classes = 10  # Number of classes
    hidden_dims = [args.hidden_dim] * args.hidden_layer  # Hidden layer sizes

    # Model and input setup
    mlp_model = MLPClassifier(input_dim, num_classes, hidden_dims)

    # Non-spark version: Timing the forward pass
    x = torch.randn(args.n_input, input_dim)
    start_time_non_spark = time.time()
    output_non_spark = mlp_model(x)
    end_time_non_spark = time.time()
    time_non_spark = end_time_non_spark - start_time_non_spark

    # Spark version: Initialize Spark session and perform distributed classification
    spark = SparkSession.builder.appName("MLPClassifier").getOrCreate()

    # Convert input data to Pandas DataFrame and then to Spark DataFrame
    df_pandas = pd.DataFrame(x.numpy())
    df_spark = spark.createDataFrame(df_pandas)

    # Create the UDF with the model's configuration
    MLPClassifier_udf = create_mlp_udf(input_dim=input_dim, num_classes=num_classes, hidden_dims=hidden_dims)

    start_time_spark = time.time()
    df_result = df_spark.withColumn("prediction", MLPClassifier_udf(*[df_spark[col] for col in df_spark.columns]))
    #df_result.show()  # Optionally show some results
    end_time_spark = time.time()

    time_spark = end_time_spark - start_time_spark
    print(f"Time cost for spark and non-spark version: [{time_spark:.3f}, {time_non_spark:.3f}] seconds")

    spark.stop()
