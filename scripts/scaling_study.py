import subprocess
import re
import sys

def run_experiment(ranks):
    print(f"Running with {ranks} MPI ranks...", file=sys.stderr)
    cmd = ["mpiexec", "-n", str(ranks), "./miniWeather_mpi"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
    
    if result.returncode != 0:
        print(f"Error running with {ranks} ranks: {result.stderr}", file=sys.stderr)
        return None

    # Parse CPU Time
    match = re.search(r"CPU Time:\s+([0-9.]+)\s+sec", result.stdout)
    if match:
        return float(match.group(1))
    else:
        print(f"Could not parse time from output: {result.stdout[:100]}...", file=sys.stderr)
        return None

def main():
    print("| Ranks | Time (s) | Speedup | Efficiency |")
    print("|-------|----------|---------|------------|")
    
    base_time = None
    
    for n in [1, 2, 3, 4]:
        time = run_experiment(n)
        if time is None:
            continue
            
        if n == 1:
            base_time = time
            speedup = 1.0
            efficiency = 100.0
        else:
            speedup = base_time / time
            efficiency = (speedup / n) * 100.0
            
        print(f"| {n:5d} | {time:8.4f} | {speedup:7.2f} | {efficiency:9.1f}% |")

if __name__ == "__main__":
    main()
