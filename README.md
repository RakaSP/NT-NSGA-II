# NT-NSGA-II

NT-NSGA-II is a Python-based Vehicle Routing Problem (VRP) experimentation project. It compares several metaheuristic optimization algorithms, including GA, PSO, ACO, NSGA-II, and an RL-assisted NSGA-II variant called NT-NSGA-II.

The project supports routing experiments on CSV-based VRP/GIB cases and TSPLIB-style TSP cases. It can split a full VRP instance into clusters, solve each cluster separately, and save per-cluster as well as aggregated experiment results.

---

<h1 id="table-of-contents" style="border-bottom: none;">Table of Contents</h1>

* [Project Structure](#project-structure)
* [Installation](#installation)
* [Running the Project](#running-the-project)
* [Features](#features)
* [Configuration & Modifying Features](#configuration-modifying-features)
* [Output Files](#output-files)
* [Visualization and Analysis](#visualization-and-analysis)
* [Reproducibility](#reproducibility)
* [License](#license)

---

<h1 id="project-structure" style="border-bottom: none;">Project Structure</h1>

```text
NT-NSGA-II/
├── Algorithm/              # Optimization algorithms and RL environment
│   ├── ACO.py
│   ├── GA.py
│   ├── NSGA2.py
│   ├── PSO.py
│   ├── BaseAlgorithm.py
│   ├── BaseBugReplicated.py
│   ├── Gym/
│   └── NN/
├── Experiments/            # Experiment runner wrapper
│   └── Experiments.py
├── Problems/               # Problem datasets and vehicle files
│   ├── GIB.csv
│   ├── GIB_vehicles.csv
│   ├── azam_cluster.csv
│   ├── vrp_nodes.csv
│   ├── vrp_vehicles.csv
│   └── TSP/
├── Utils/                  # Logging and helper utilities
├── vrp_core/               # Core VRP loading, scoring, decoding, and metrics
├── Core.py                 # Main solver engine
├── Runner.py               # Main experiment entry point
├── solver_config.yaml      # Experiment configuration
├── viz_server.py           # Local route visualization server
├── visualize.ipynb         # Analysis notebook
└── README.md
```

---

<h1 id="installation" style="border-bottom: none;">Installation</h1>

Clone the repository:

```bash
git clone https://github.com/RakaSP/NT-NSGA-II.git
cd NT-NSGA-II
```

Create a virtual environment:

```bash
python -m venv pyenv
```

Activate the virtual environment.

On Windows:

```bash
pyenv\Scripts\activate
```

On macOS/Linux:

```bash
source pyenv/bin/activate
```

Install the required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

<h1 id="running-the-project" style="border-bottom: none;">Running the Project</h1>

This section explains the normal workflow after installation.

### Step 1: Check the configuration file

Open:

```text
solver_config.yaml
```

Make sure the dataset paths and output directory are correct.

For the GIB case:

```yaml
tsp: false
gib: true

gib_nodes: "Problems/GIB.csv"
gib_depot_id: 0
gib_vehicles: "Problems/GIB_vehicles.csv"

output_dir: results
```

For a TSP case:

```yaml
tsp: true
gib: false

tsp_nodes: "Problems/TSP/eil76.tsp"
tsp_depot_id: 1
tsp_vehicles: "Problems/vrp_vehicles.csv"

output_dir: results
```

All result paths in this README use `<output_dir>` to mean the folder set in `solver_config.yaml`.

### Step 2: Choose which algorithms to run

Open:

```text
Runner.py
```

Find the algorithm list:

```python
ALGOS = ["aco", "pso", "ga", "nsga2", "ntnsga2"]
```

You can run all algorithms or only selected ones.

Example:

```python
ALGOS = ["nsga2", "ntnsga2"]
```

### Step 3: Set the number of runs

Still inside `Runner.py`, set:

```python
RUNS_PER_ALGO = 1
```

For repeated experiments:

```python
RUNS_PER_ALGO = 10
```

### Step 4: Choose clustering mode

For automatic clustering:

```python
PRE_CLUSTER_PATH = None
```

For predefined clusters:

```python
PRE_CLUSTER_PATH = "results/pre_cluster.csv"
```

Expected predefined cluster format:

```csv
jalur,route
1,"5,6,8,58,62,63,64,75"
2,"32,43,46,51,54,55,56,69,73,74"
```

Each row represents one cluster. The `route` column contains the node IDs assigned to that cluster.

### Step 5: Run the experiment

Run:

```bash
python Runner.py
```

The script will run the selected algorithms and save results into the configured output directory.

Example output folders:

```text
<output_dir>/aco_run1/
<output_dir>/pso_run1/
<output_dir>/ga_run1/
<output_dir>/nsga2_run1/
<output_dir>/ntnsga2_run1/
```

### Step 6: Check the results

After the run finishes, check:

```text
<output_dir>/experiment_summary.csv
<output_dir>/experiment_summary.json
```

The CSV summary contains important metrics such as:

* algorithm
* run index
* seed
* total distance
* total cost
* route time
* solving time
* number of clusters

You can also inspect per-cluster results inside each algorithm run folder:

```text
<output_dir>/nsga2_run1/
├── cluster_0/
│   ├── cluster_summary.json
│   ├── metadata.json
│   └── metrics.csv
├── cluster_1/
│   ├── cluster_summary.json
│   ├── metadata.json
│   └── metrics.csv
└── ...
```

The main file for each cluster is:

```text
cluster_summary.json
```

It contains the final distance, cost, route time, solving time, and route.

### Step 7: Visualize the routes

Run the local route visualization server:

```bash
python viz_server.py --config solver_config.yaml --root <output_dir>
```

Example:

```bash
python viz_server.py --config solver_config.yaml --root results
```

Then open the URL printed in the terminal, usually:

```text
http://127.0.0.1:8000
```

### Step 8: Analyze the experiment

Open:

```text
visualize.ipynb
```

Use this notebook to analyze experiment results, such as:

* convergence curves
* distance comparison
* route-time comparison
* solving-time comparison
* Pareto front analysis
* Wilcoxon statistical test
* algorithm performance comparison

Make sure the notebook points to the same folder as `output_dir`.

---

<h1 id="features" style="border-bottom: none;">Features</h1>

* Vehicle Routing Problem optimization
* Support for multiple algorithms:

  * Genetic Algorithm
  * Particle Swarm Optimization
  * Ant Colony Optimization
  * NSGA-II
  * NT-NSGA-II
* Cluster-based VRP solving
* Optional fixed or predefined clustering
* Optional RL-assisted NSGA-II parameter control
* Per-cluster result export
* Aggregated experiment summary export
* Route visualization server
* Analysis notebook for experiment comparison
* Support for GIB CSV cases and TSP/TSPLIB cases

---

<h1 id="configuration-modifying-features" style="border-bottom: none;">Configuration & Modifying Features</h1>

### Main config file

Experiments are controlled from:

```text
solver_config.yaml
```

Important configuration fields include:

```yaml
# Problem mode
tsp: false
gib: true

# GIB problem files
gib_nodes: "Problems/GIB.csv"
gib_depot_id: 0
gib_vehicles: "Problems/GIB_vehicles.csv"

# TSP problem files
tsp_nodes: "Problems/TSP/eil76.tsp"
tsp_depot_id: 1
tsp_vehicles: "Problems/vrp_vehicles.csv"

# Output folder
output_dir: results

# Runtime limit
time_limit: 2.8
```

Algorithm parameters are also configured in the same file:

```yaml
algo_params:
  nsga2:
    population_size: 50
    crossover_rate: 0.7
    mutation_rate: 0.1

  ga:
    population_size: 50
    crossover_rate: 0.7
    mutation_rate: 0.1

  aco:
    number_of_ants: 50
    pheromone_exponent: 1.0
    heuristic_exponent: 5.0
    evaporation_rate: 0.2

  pso:
    population_size: 200
    inertia_weight: 0.5
    cognitive_coefficient: 1.6
    social_coefficient: 1.6
```

### Changing the runtime limit

The experiment time limit is controlled by:

```yaml
time_limit: 2.8
```

Increase it for longer optimization:

```yaml
time_limit: 10
```

### Changing algorithm parameters

Each algorithm has its own parameter section in `solver_config.yaml`.

Example for NSGA-II:

```yaml
algo_params:
  nsga2:
    population_size: 50
    crossover_rate: 0.7
    mutation_rate: 0.1
```

Example modification:

```yaml
algo_params:
  nsga2:
    population_size: 100
    crossover_rate: 0.8
    mutation_rate: 0.05
```

### Using predefined clusters

In `Runner.py`, set:

```python
PRE_CLUSTER_PATH = "results/pre_cluster.csv"
```

instead of:

```python
PRE_CLUSTER_PATH = None
```

The expected CSV format is:

```csv
jalur,route
1,"5,6,8,58,62,63,64,75"
2,"32,43,46,51,54,55,56,69,73,74"
```

Each row represents one cluster. The `route` column contains the node IDs assigned to that cluster.

### Static vs random clustering

Clustering behavior is controlled in:

```text
Experiments/Experiments.py
```

Find:

```python
STATIC_CLUSTER = True
STATIC_CLUSTER_SEED = 187031
```

When `STATIC_CLUSTER = True`, the same clustering seed is reused across all runs.

When `STATIC_CLUSTER = False`, each run receives a new random seed, and clustering may differ between runs.

If `PRE_CLUSTER_PATH` is provided in `Runner.py`, the predefined cluster file determines the clusters.

### Replicating Ahmad's Bug

The repository includes two base algorithm classes:

```python
from Algorithm.BaseAlgorithm import BaseAlgorithm
from Algorithm.BaseBugReplicated import BaseBugReplicated
```

`BaseAlgorithm` uses the corrected base behavior.

`BaseBugReplicated` is used when you want to replicate the previous buggy behavior for comparison or validation.

To replicate Ahmad's bug for an algorithm, change the parent class from `BaseAlgorithm` to `BaseBugReplicated`.

Example for `Algorithm/NSGA2.py`.

Before:

```python
class NSGA2(BaseAlgorithm):
    ...
```

After:

```python
class NSGA2(BaseBugReplicated):
    ...
```

Do the same for other algorithm files if needed:

```text
Algorithm/ACO.py
Algorithm/GA.py
Algorithm/PSO.py
Algorithm/NSGA2.py
```

For example:

```python
class ACO(BaseBugReplicated):
    ...
```

```python
class GA(BaseBugReplicated):
    ...
```

```python
class PSO(BaseBugReplicated):
    ...
```

```python
class NSGA2(BaseBugReplicated):
    ...
```

To return to the corrected behavior, change the parent class back to `BaseAlgorithm`.

Example:

```python
class NSGA2(BaseAlgorithm):
    ...
```

When comparing corrected and bug-replicated results, keep the same input data, clustering seed, runtime, and algorithm parameters.

---

<h1 id="output-files" style="border-bottom: none;">Output Files</h1>

After running an experiment, the project writes per-cluster and aggregated results.

Experiment-level outputs:

```text
<output_dir>/experiment_summary.csv
<output_dir>/experiment_summary.json
```

Per-cluster outputs:

```text
<output_dir>/<algorithm>_run<run_number>/cluster_0/
├── cluster_summary.json
├── metadata.json
└── metrics.csv
```

The CSV summary contains key metrics such as:

* algorithm
* run index
* seed
* total distance
* total cost
* route time
* solving time
* number of clusters

---

<h1 id="visualization-and-analysis" style="border-bottom: none;">Visualization and Analysis</h1>

The repository includes two visualization and analysis tools.

### 1. Route visualization server

Run:

```bash
python viz_server.py --config solver_config.yaml --root <output_dir>
```

Example:

```bash
python viz_server.py --config solver_config.yaml --root results
```

Then open the URL printed in the terminal, usually:

```text
http://127.0.0.1:8000
```

The visualizer can display:

* routes
* clusters
* route metrics
* distance comparisons
* cluster summaries

### 2. Analysis notebook

Open:

```text
visualize.ipynb
```

Use this notebook for deeper experiment analysis, including:

* convergence visualization
* distance comparison
* route-time comparison
* solving-time comparison
* Pareto front visualization
* Wilcoxon statistical testing
* algorithm comparison plots

Before running the notebook, make sure the result path inside the notebook matches your experiment output folder.

---

<h1 id="reproducibility" style="border-bottom: none;">Reproducibility</h1>

To make clustering reproducible, set:

```python
STATIC_CLUSTER = True
STATIC_CLUSTER_SEED = 187031
```

To compare algorithms fairly, keep the same:

* dataset
* vehicle file
* clustering mode
* clustering seed
* algorithm parameters
* runtime limit
* number of runs

If using predefined clusters, keep the same `PRE_CLUSTER_PATH` across all experiments.

---

<h1 id="license" style="border-bottom: none;">License</h1>

This project is licensed under the MIT License.

Copyright (c) 2026 Raka Satya Prasasta

See the [LICENSE](LICENSE) file for the full license text.
