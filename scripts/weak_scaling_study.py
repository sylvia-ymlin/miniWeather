import subprocess
import re
import sys
import os

def clean_build():
    # Helper to clean/configure cmake with new params
    pass # We will do this inside the loop

def run_weak_scaling_step(ranks):
    # Calculate global NX based on ranks to keep work-per-rank constant
    # Baseline: 100x50 per rank
    # Rank 1: NX=100
    # Rank 2: NX=200
    # ...
    local_nx = 100
    global_nx = local_nx * ranks
    global_nz = 50
    
    print(f"--- Configuration: {ranks} Ranks, Global Grid {global_nx}x{global_nz} ---", file=sys.stderr)
    
    # Reconfigure CMake
    cmake_cmd = [
        "cmake", 
        f"-DNX={global_nx}", 
        f"-DNZ={global_nz}", 
        ".."
    ]
    subprocess.run(cmake_cmd, cwd="./build", check=True, stdout=subprocess.DEVNULL)
    
    # Rebuild
    subprocess.run(["make", "-j"], cwd="./build", check=True, stdout=subprocess.DEVNULL)
    
    # Run
    run_cmd = ["mpiexec", "-n", str(ranks), "./miniWeather_mpi"]
    result = subprocess.run(run_cmd, capture_output=True, text=True, cwd="./build")
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
        return None

    # Parse Time
    match = re.search(r"CPU Time:\s+([0-9.]+)\s+sec", result.stdout)
    if match:
        return float(match.group(1))
    return None

def main():
    print("| Ranks | Grid Size | Time (s) | Weak Efficiency |")
    print("|-------|-----------|----------|-----------------|")
    
    baseline_time = None
    
    for n in [1, 2, 3, 4]:
        time = run_weak_scaling_step(n)
        if time is None:
            continue
            
        if n == 1:
            baseline_time = time
            efficiency = 100.0
        else:
            # Weak Scaling Efficiency = T_1 / T_n * 100%
            # Ideal: Time stays constant as we add resources + work
            efficiency = (baseline_time / time) * 100.0
            
        grid = f"{100*n}x50"
        print(f"| {n:5d} | {grid:9s} | {time:8.4f} | {efficiency:14.1f}% |")

if __name__ == "__main__":
    main()
