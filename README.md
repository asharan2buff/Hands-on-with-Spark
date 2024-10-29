# HomeWork 2

This assignment provides hands-on experience with writing and executing Spark in Python. You’ll start by installing PySpark, then implement functions to calculate edit distance between text strings and create efficient inference code for an MLP model. At last, you’ll modify a bird flock simulation code to utilize Spark for enhanced performance.

---

## 1. Task
### 1.1 Edit Distance
A script for loading text data and computing the edit distance is provided in [`edit_dist.py`](https://github.com/UB-CSE587/homework_2/blob/main/edit_dist.py), which includes a basic implementation using a `for` loop. Revise this code to include:
- A Spark version
- A multi-process version

Record the execution time for each version when computing pairwise edit distances for 1,000 sentences.

### 1.2 MLP Inference
Inference code for an MLP classifier is available in [`MLP.py`](https://github.com/UB-CSE587/homework_2/blob/main/MLP.py). Update this code to include a Spark-based implementation for more efficient inference.

### 1.3 Flock Move Simulation using Spark
In this task, you will work with a bird flock simulation, where each bird’s position is represented by a point in 3D space. Each bird follows movement rules based on flock dynamics:

1. **Alignment**: Birds attempt to stay close to the leader bird, which follows a determined path with uniform velocity.
2. **Separation**: Birds maintain distance from nearby neighbors. If a bird gets too close to a neighbor (within a threshold), it moves away.
3. **Cohesion**: Birds strive to stay with the flock. If a bird is too far from its nearest neighbor (beyond a threshold), it moves closer.
4. **Velocity Constraints**: Flying speed is restricted to a certain range—too slow and the bird risks falling; too fast, and it exceeds physical limits.

A demonstration of the bird flock movement is shown below:
![bird](bird_simulation.gif)

Please update the provided simulation code ([`bird.py`](https://github.com/UB-CSE587/homework_2/blob/main/bird.py)) to utilize Spark for parallel processing of position updates for all birds. All necessary hyperparameters (thresholds, max/min values) are included in the script, along with a basic implementation using a for-loop to update bird positions. Revise this to add a Spark-based implementation to handle position updates more efficiently.

---

## 2. Evaluation

The evaluation will be conducted on the same machine to ensure consistency across all implementations.

- **Task 1.1**, you are required to report the time costs for each version (for-loop, multi-process, and Spark) for varying numbers of sentences: `[10, 50, 100, 250, 500, 1000]`. Include these results in your report. 
Submitted code will be tested by a command like:
```
python edit_dist.py --csv_dir /path/to/csv --num_sentences n
```
And at the end of your code, it should print the time cost in this format:
```
print(f"Time cost (Spark, multi-process, for-loop): [{time_1:.3f}, {time_2:.3f}, {time_3:.3f}]")
```

- **Task 1.2**, test and report the time costs with values for `n_input` set to `[1000, 5000, 10000, 50000, 100000]`, keeping `hidden_dim` and `hidden_layer` as default. Submitted code will be tested by a command like:
```
python MLP.py --n_input n --hidden_dim d --hidden_layer l
```
And at the end of your code, it should print the time cost in this format:
```
print(f"Time cost for spark and non-spark version: [{time_1:.3f},  {time_2:.3f}] seconds")
```

- **Task 2**, you are required to run the simulation using both the Spark and non-Spark implementations with `[200, 1,000, 5,000, 10,000]` birds for `200` frames. Record the time cost per frame for both implementations and include these results in your report, along with a discussion of your observations.

Additional results with different configurations are encouraged to enhance your report. Present all results in your report, either as tables or plots for clarity, and write discussion on any trends observed.

---

## 3. Scoring

The score will be assigned as follows:

- **Task 1.1**: Successful code execution – 10 points (must print reasonable time cost in required format).
- **Task 1.2**: Successful code execution – 10 points (must print reasonable time cost in required format).
- **Task 2**: Successful code execution – 20 points (must print reasonable time cost in required format and generate a meaningful GIF image).
- **Report Quality**: 60 points (must include all numerical results, with clear descriptions and discussion).

