
import subprocess
import re
import sys
import argparse

def run_simulation(exe_path, nx, nz, time):
    """Running simulation ensuring that physics is correct"""
    cmd = [exe_path, "--nx", str(nx), "--nz", str(nz), "--time", str(time)]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: Simulation failed with return code {result.returncode}")
        print(result.stderr)
        return False, 0.0, 0.0

    # Capture d_mass and d_te from output
    # d_mass: -2.265785e-14
    # d_te:   -2.069896e-06
    mass_match = re.search(r"d_mass:\s+([-0-9.eE+]+)", result.stdout)
    te_match = re.search(r"d_te:\s+([-0-9.eE+]+)", result.stdout)

    if not mass_match or not te_match:
        print("Error: Could not parse d_mass or d_te from output")
        return False, 0.0, 0.0

    d_mass = float(mass_match.group(1))
    d_te = float(te_match.group(1))
    
    return True, d_mass, d_te

def validate(d_mass, d_te, mass_tol=1e-13, te_tol=1e-4):
    """Validation thresholds"""
    valid = True
    print(f"\nValidation Results:")
    
    # Check Mass Conservation (should be near machine precision)
    if abs(d_mass) > mass_tol:
        print(f"❌ FAIL: Mass conservation violation. d_mass ({d_mass}) > tolerance ({mass_tol})")
        valid = False
    else:
        print(f"✅ PASS: Mass conserved. d_mass ({d_mass}) <= tolerance ({mass_tol})")

    # Check Energy Conservation (allow small drift due to hyper-viscosity)
    if abs(d_te) > te_tol:
        print(f"❌ FAIL: Energy drift too large. d_te ({d_te}) > tolerance ({te_tol})")
        valid = False
    else:
        print(f"✅ PASS: Energy drift acceptable. d_te ({d_te}) <= tolerance ({te_tol})")

    return valid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate miniWeather physics correctness")
    parser.add_argument("--exe", default="./miniWeather_serial", help="Path to executable")
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--nz", type=int, default=50)
    parser.add_argument("--time", type=float, default=10.0)
    args = parser.parse_args()

    success, d_mass, d_te = run_simulation(args.exe, args.nx, args.nz, args.time)
    
    if success:
        if validate(d_mass, d_te):
            print("\nResult: SUCCESS (Physics Verified)")
            sys.exit(0)
        else:
            print("\nResult: FAILURE (Physics Violation)")
            sys.exit(1)
    else:
        sys.exit(1)
