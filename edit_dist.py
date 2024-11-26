import os
from itertools import combinations
import pandas as pd
import multiprocessing
import time
from tqdm import tqdm
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import udf
import warnings
warnings.filterwarnings("ignore")


def edit_distance(pair):
    str1, str2 = pair
    m, n = len(str1), len(str2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])

    return dp[m][n]

# Function to compute edit distances using multiple processes
def compute_edit_distance_multiprocess(pairs, num_workers):
    with multiprocessing.Pool(processes=num_workers) as pool:
        distances = list(tqdm(pool.imap(edit_distance, pairs), total=len(pairs), desc="Multiprocessing Pairs"))
    return distances


if __name__=="__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Edit Distance with PySpark")
    parser.add_argument('--csv_dir', type=str, required=True, help="Directory of csv file")
    parser.add_argument('--num_sentences', type=int, required=True, help="Number of sentences")
    args = parser.parse_args()

    # Number of processes to use (set to the number of CPU cores)
    num_workers = multiprocessing.cpu_count()
    print(f'Number of available CPU cores: {num_workers}')

    # Load the sentences from the CSV file
    text_data = pd.read_csv(args.csv_dir)['sentence']
    text_data = text_data[:args.num_sentences]
    pair_data = list(combinations(text_data, 2))

    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("EditDistancePy") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "100") \
        .getOrCreate()

    # Define schema for DataFrame
    schema = StructType([
        StructField("str1", StringType(), True),
        StructField("str2", StringType(), True)
    ])

    # Register edit distance function as a Spark UDF
    edit_distance_udf = udf(lambda str1, str2: edit_distance((str1, str2)), IntegerType())

    # Create DataFrame
    spark_pair_data = spark.createDataFrame(pair_data, schema=schema).repartition(100)

    # Measure time for Spark version
    start_time_spark = time.time()
    result_df = spark_pair_data.withColumn("edit_distance", edit_distance_udf("str1", "str2"))
    result_df.count()  # Trigger computation
    end_time_spark = time.time()
    time_spark = end_time_spark - start_time_spark
    spark.stop()

    # Measure time for multiprocessing version
    start_time_multiprocess = time.time()
    edit_distances = compute_edit_distance_multiprocess(pair_data, num_workers)
    end_time_multiprocess = time.time()
    time_multiprocess = end_time_multiprocess - start_time_multiprocess

    # Measure time for vanilla for-loop version
    start_time_forloop = time.time()
    distances = []
    for pair in tqdm(pair_data, desc="For-loop Pairs", ncols=100):
        distances.append(edit_distance(pair))
    end_time_forloop = time.time()
    time_forloop = end_time_forloop - start_time_forloop

    # Print results in the specified format
    print(f"Time cost (Spark, multi-process, for-loop): [{time_spark:.3f}, {time_multiprocess:.3f}, {time_forloop:.3f}]")
