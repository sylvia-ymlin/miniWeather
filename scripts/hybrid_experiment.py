
import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
# Increase problem size to stress memory bandwidth
# 400x100 is large enough to cause contention on a laptop
NX = 400
NZ = 200
TIME = 10.0 

def run_case(label, mpi_ranks, omp_threads):
    print(f"Running Case: {label} (MPI={mpi_ranks}, OMP={omp_threads})...")
    
    # Set Environment Variable for OpenMP
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp_threads)
    
    cmd = ["mpirun", "-n", str(mpi_ranks), "./miniWeather_mpi", 
           "--nx", str(NX), "--nz", str(NZ), "--time", str(TIME)]
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None

    # Parse CPU Time
    match = re.search(r"CPU Time:\s+([0-9.]+)\s+sec", result.stdout)
    if match:
        time = float(match.group(1))
        print(f"  -> Time: {time:.4f} s")
        return time
    else:
        print("  -> Failed to parse time")
        return None

# Defin Experiments (Total Parallelism = 4)
experiments = [
    {"label": "Pure MPI\n(4 MPI, 1 Thread)", "ranks": 4, "threads": 1},
    {"label": "Balanced\n(2 MPI, 2 Threads)", "ranks": 2, "threads": 2},
    {"label": "Hybrid\n(1 MPI, 4 Threads)", "ranks": 1, "threads": 4}
]

results = []
labels = []

print(f"--- Hybrid Scaling Experiment (Grid: {NX}x{NZ}) ---")
for exp in experiments:
    t = run_case(exp["label"], exp["ranks"], exp["threads"])
    if t:
        results.append(t)
        labels.append(exp["label"])

# Plotting
plt.figure(figsize=(10, 6))
bars = plt.bar(labels, results, color=['#d62728', '#ff7f0e', '#2ca02c'])

plt.ylabel('Execution Time (seconds) - Lower is Better', fontsize=12)
plt.title(f'Performance Comparison: MPI vs Hybrid OpenMP\n(Grid: {NX}x{NZ}, Total Cores: 4)', fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}s',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('hybrid_performance.png', dpi=300)
print("\nPlot saved to hybrid_performance.png")
