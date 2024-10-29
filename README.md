# homework_2

This assignment provides hands-on experience with writing and executing Spark in Python. You’ll start by installing PySpark, then implement functions to calculate edit distance between text strings and create efficient inference code for an MLP model. In the second half, you’ll modify a bird flock simulation code to utilize Spark for enhanced performance.

## Task 1
### 1.1 Edit Distance
A script for loading text data and computing the edit distance is provided in [`edit_dist.py`](https://github.com/UB-CSE587/homework_2/blob/main/edit_dist.py), which includes a basic implementation using a `for` loop. Revise this code to include:
- A Spark version
- A multi-process version

Record the execution time for each version when computing pairwise edit distances for 1,000 sentences.

### 1.2 MLP Inference
Inference code for an MLP classifier is available in [`MLP.py`](https://github.com/UB-CSE587/homework_2/blob/main/MLP.py). Update this code to include a Spark-based implementation for more efficient inference.

---

## Task 2
### Flock Simulation using Spark
In this task, you will work with a bird flock simulation, where each bird’s position is represented by a point in 3D space. Each bird follows movement rules based on flock dynamics:

1. **Alignment**: Birds attempt to stay close to the leader bird, which follows a determined path with uniform velocity.
2. **Separation**: Birds maintain distance from nearby neighbors. If a bird gets too close to a neighbor (within a threshold), it moves away.
3. **Cohesion**: Birds strive to stay with the flock. If a bird is too far from its nearest neighbor (beyond a threshold), it moves closer.
4. **Velocity Constraints**: Flying speed is restricted to a certain range—too slow and the bird risks falling; too fast, and it exceeds physical limits.


A demonstration of the bird flock movement is shown below:
![bird](bird_simulation.gif)

Please update the provided simulation code (`birds.py`) to utilize Spark for parallel processing of position updates for all birds. All necessary hyperparameters (thresholds, max/min values) are included in the script, along with a basic implementation using a for-loop to update bird positions. Revise this to add a Spark-based implementation to handle position updates more efficiently.
