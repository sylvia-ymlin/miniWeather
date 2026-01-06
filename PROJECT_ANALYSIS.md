# Project Deep Analysis: miniWeather (HPC Mini-App)

## 1. Surface Level
**"What does this project do? How does it work?"**

*   **Core Task**: Simulates the dynamics of a stratified, compressible, dry atmosphere using the Euler equations. It mimics the computational kernel of large-scale weather prediction models (like WRF or CAM).
*   **Key Features**:
    *   **Physics**: Solves for Density ($\rho$), Momentum ($\rho u, \rho w$), and Potential Temperature ($\rho \theta$).
    *   **Parallelism**: Implements **Hybrid MPI + OpenMP** parallelism. MPI handles domain decomposition (inter-node), while OpenMP handles loop-level parallelism (intra-node).
    *   **Numerical Method**: Uses a **Finite Volume Method (FVM)** with 4th-order spatial reconstruction and 3rd-order Runge-Kutta time integration.
*   **Input/Output**:
    *   **Input**: Simulation parameters (Grid size `nx, nz`, Time `dt`, Data Scenario like "Rising Thermal").
    *   **Output**: Console logs tracking **Mass Conservation** and **Total Energy**; optional parallel NetCDF (`.nc`) files for visualization.

## 2. Middle Level
**"Why this architecture? How is correctness verified?"**

*   **Algorithmic Choices**:
    *   **Dimensional Splitting (Strang Splitting)**: The solver updates the X-direction and Z-direction sequentially (`x -> z`, then `z -> x`). This simplifies the implementation of high-order stencils and allows re-using the `semi_discrete_step` logic for both dimensions.
    *   **1D Domain Decomposition**: The domain is sliced into vertical "slabs" (along X). This minimizes the surface area of ghost layers (halo regions) each process needs to exchange.
*   **Verification Mechanisms**:
    *   **Conservation Laws**: The simulation explicitly monitors Global Mass and Total Energy. In a purely explicitly scheme, Mass should be conserved to machine precision ($\approx 10^{-15}$), while Energy may drift slightly due to dissipation (Hyper-viscosity).
    *   **Hyper-viscosity**: Explicit diffusion (`hv_beta`) is added to stabilize the numerical scheme against high-frequency noise (Gibbs phenomenon), which is critical for non-linear shock capture.

## 3. Deep Level
**"Technical Decisions & Engineering Trade-offs"**

*   **Memory Layout (AoS vs SoA)**:
    *   The `state` array is allocated as a single contiguous block but accessed as `[variable][vertical][horizontal]` (conceptually).
    *   *Critical Observation*: The code uses a "Strided" layout where variables are the outer dimension. This effectively creates **Structure-of-Arrays (SoA)** behavior for inner loops (looping over `k` then `i`), which is generally friendly for SIMD vectorization on CPUs.
*   **Halo Exchange Optimizations**:
    *   **Explicit Packing**: The `set_halo_values_x` function manually packs non-contiguous boundary data into specific send buffers (`sendbuf_l/r`). This is necessary because MPI Derived Datatypes can sometimes be slower than manual packing for simple strided patterns, and this gives the developer full control over memory access.
    *   **Non-Blocking Communication**: Uses `MPI_Irecv` (post receive) -> `Pack` -> `MPI_Isend`, allowing potential overlap of packing work with communication latency.
*   **I/O Bottleneck Mitigation**:
    *   **Parallel NetCDF (PNetCDF)**: The original code leverages PNetCDF to allow all MPI ranks to write to a single file simultaneously. This avoids the bottleneck of gathering all data to Rank 0 (Serial I/O), which would crash memory on Petascale systems.
    *   **Engineering Improvement (My Contribution)**: Recognizing that PNetCDF is a "heavy" dependency for local development, I implemented `ifdef` guards to make I/O optional, significantly lowering the barrier to entry for development and testing.

## 4. Build System Modernization
*   **Why CMake?**: The original project used manual `Makefile`s dependent on specific HPC modules (Cray/PGI).
*   **Migration**: I authored a modern `CMakeLists.txt` that:
    1.  Automatically detects the MPI implementation (`find_package(MPI)`).
    2.  Sets appropriate C++ standards (C++11).
    3.  Manages configuration variables (`NX`, `NZ`) via accessible CMake cache options.
    4.  Cleanly separates `miniWeather_mpi` from `serial` targets.

## 5. Performance Analysis
**Experimental validation of the parallel implementation.**

### 5.1 Strong Scaling Study
**Methodology**: Fixed total problem size ($100 \times 50$ grid). Increased MPI ranks from 1 to 4 to measure Speedup ($T_1/T_N$) and Parallel Efficiency.

| Ranks | Time (s) | Speedup | Efficiency | Analysis |
|:---:|:---:|:---:|:---:|:---|
|  1  | 4.95 | 1.00 | 100.0% | Baseline |
|  2  | 2.52 | 1.96 |  98.1% | Excellent Scaling |
|  3  | 1.92 | 2.59 |  86.2% | Good Scaling |
|  4  | 1.76 | 2.81 |  70.2% | Efficiency Drop (Small Problem) |

**Conclusion**: The system exhibits strong scaling limitations at $N=4$ for this small problem size. With only $25 \times 50$ cells per rank, the surface-to-volume ratio increases, causing halo exchange latency (`MPI_Isend/Irecv`) to dominate the computation time. This confirms the need for larger problem sizes (Weak Scaling) or 2D decomposition to utilize higher core counts effectively.

### 5.2 Weak Scaling Study
**Methodology**: Variable problem size ($100 \times 50$ cells per rank). Increased ranks from 1 to 4 while keeping local work constant. Ideal scaling implies constant Time.

| Ranks | Grid Size | Time (s) | Weak Efficiency | Analysis |
|:---:|:---:|:---:|:---:|:---|
| 1 | 100x50 | 12.70 | 100.0% | Baseline |
| 2 | 200x50 | 25.03 | 50.8% | Memory Bandwidth Saturation |
| 3 | 300x50 | 23.56 | 53.9% | Memory Bandwidth Saturation |
| 4 | 400x50 | 38.31 | 33.2% | **Severe Contention** |

**Conclusion**: The poor weak scaling efficienty (dropping to 33%) on the local test bench (Apple M-series Unified Memory) indicates **Memory Bandwidth Saturation**. While the CPU core count increases, the total available system bandwidth is shared. As the global problem size grows, the concurrent memory requests from 4 MPI processes saturate the memory controller, a common phenomenon in "fat node" shared-memory architectures compared to distributed clusters.

## 6. Code Refactoring & Modernization
**Improving Maintainability, Safety, and Extensibility**

In addition to performance analysis, I undertook a significant refactoring effort to modernize the legacy C-style codebase into idiomatic C++.

*   **Object-Oriented Encapsulation**:
    *   **Problem**: The original code relied on global variables (`double *state`, `int nx`) and free functions, polluting the global namespace and making unit testing impossible.
    *   **Solution**: Encapsulated the entire simulation state and logic into a `MiniWeatherSimulation` class. This provides a clear interface (`init`, `Run`, `Finalize`) and allows for multiple simulation instances (e.g., for ensemble support in the future).

*   **RAII & Memory Safety**:
    *   **Problem**: Manual `malloc` and `free` calls were scattered throughout `init`, `output`, and `finalize`, leading to potential double-free errors (observed during development) and memory leaks if early returns occurred.
    *   **Solution**: Replaced all raw pointer arrays with `std::vector<double>`. This leverages **RAII (Resource Acquisition Is Initialization)** to ensure automatic memory cleanup when the class instance goes out of scope, eliminating memory leaks and dampening usage errors.

*   **Decoupling Serial & MPI**:
    *   **Problem**: The "serial" version contained a "Dummy MPI" section and included `mpi.h` with dummy functions, creating a hard dependency on MPI libraries even for the serial build.
    *   **Solution**: Completely stripped MPI dependencies from `miniWeather_serial.cpp`. It is now a standalone C++ application that builds without an MPI compiler, serving as a true "Gold Standard" baseline for correctness verification.

*   **Dependency Management**:
    *   **Problem**: Hard dependency on Parallel NetCDF (PNetCDF).
    *   **Solution**: Implemented strict preprocessor guards (`#ifdef _PNETCDF`) around all I/O calls. This allows the code to build and run (with output disabled) on systems lacking the heavy PNetCDF library, significantly improving portability.
