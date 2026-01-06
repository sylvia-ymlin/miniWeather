# Interview Preparation: miniWeather (HPC & Parallel Scaling)

## 1. Elevator Pitch
"I optimized and modernized a High-Performance Computing (HPC) mini-app called `miniWeather`, which simulates atmospheric dynamics. My work focused on unblocking local development by refactoring the I/O layer to be modular (removing hard dependencies on Parallel NetCDF) and migrating the legacy build system to modern CMake. This allowed me to rigorously verify the 4th-order Finite Volume solver's scalability and correctness (conservation laws) on standard hardware before deployment to clusters."

## 2. STAR Stories

### Situation: Dependency Blocker for Local Dev
*   **Situation**: The project required `PNetCDF` (Parallel NetCDF) to compile, which is standard on Cray supercomputers but rare/complex to install on local macOS/Linux workstations. This blocked rapid prototyping and debugging.
*   **Task**: Make the I/O layer optional without breaking the MPI communication structure.
*   **Action**: I utilized C++ preprocessor guards (`#ifdef _PNETCDF`) to isolate all NetCDF calls. I refactored the `output()` and error-handling routines to degrade gracefully (printing "Output disabled") when the library is missing. I also updated the build system to conditionally link the library.
*   **Result**: Reduced the "Time to Hello World" for a new developer from hours (library installation) to minutes. The code now compiles out-of-the-box on any machine with a C++ compiler and MPI.

### Situation: Modernizing Legacy Build Systems
*   **Situation**: The project relied on brittle, machine-specific Makefiles (`Makefile.bw`, `Makefile.xl`, etc.) that hardcoded compiler paths and flag combinations.
*   **Task**: Create a portable, "write once, run anywhere" build configuration.
*   **Action**: I wrote a `CMakeLists.txt` using modern CMake (Target-based architecture). I used `find_package(MPI REQUIRED)` to abstract away compiler wrappers (`mpicxx`) and library paths, and exposed simulation parameters (`NX`, `NZ`) as build-time configuration options.
*   **Result**: Achieved cross-platform compatibility (macOS/Linux) and enabled integration with IDEs (like VS Code/CLion) that rely on CMake compilation databases.

## 3. Technical Deep Dive Questions

### Q: Why use 1D domain decomposition instead of 2D?
**A**: For this scale (typically $10000 \times 100$ grid), the vertical dimension $Z$ is much smaller than $X$.
1.  **Memory Contiguity**: Keeping $Z$ contiguous in memory allows for better CPU cache locality during vertical flux calculations (which are expensive).
2.  **Physics Dependency**: Many atmospheric processes (like radiation or column physics) are strictly vertical. Splitting $Z$ would require frequent synchronization for these physics packages.
3.  **Trade-off**: At extreme scales (millions of cores), 1D decomposition limits scalability (message size $\propto NZ$). 2D decomposition is eventually needed but adds code complexity (Strided memory types for halos).

### Q: Explain the Halo Exchange in this code.
**A**:
1.  **Pack**: Before communication, we manually copy the non-contiguous boundary stripes (ghost cells) from the main 3D `state` array into a contiguous `sendbuf`.
2.  **Isend/Irecv**: We use non-blocking MPI calls (`MPI_Isend`, `MPI_Irecv`). This allows us to potentially overlapping computation of the *interior* domain while the boundary data is in flight (Latency Hiding).
3.  **Wait**: We `MPI_Waitall` before processing the boundaries.
4.  **Unpack**: Received data is copied back into the `state` array's halo regions.

### Q: How did you verify correctness?
**A**:
I relied on **Conservation Laws**. The Euler equations conserve Mass, Momentum, and Energy.
*   I checked `d_mass` (change in mass) at the end of the simulation. It was $\approx 10^{-15}$, confirming the finite volume transport was conservative.
*   I monitored `d_te` (Total Energy). The slight drift ($10^{-4}$) is expected due to the explicit Hyper-viscosity term added for numerical stability, which dissipates energy to dampen oscillations.

## 4. Code Walkthrough Checklist
*   [ ] Show `CMakeLists.txt`: Explain `find_package(MPI)`.
*   [ ] Show `miniWeather_mpi.cpp` -> `perform_timestep`: Explain the Strang Splitting (`x` then `z`, then `z` then `x`).
*   [ ] Show `set_halo_values_x`: Explain the Manual Packing vs MPI Types trade-off.
*   [ ] Show `output`: Explain the `#ifdef` engineering fix.

---

## 5. Mock Interview Transcript (HPC Engineer Role)

**Interviewer:** "Let's talk about your `miniWeather` project. Can you explain the challenge with the original I/O implementation?"

**You:** "Certainly. The original project tightly coupled the `output()` function with the `Parallel-NetCDF` library. While this is standard on Cray supercomputers like Blue Waters, it's a huge pain point for local development on laptops or CI/CD environments where PNetCDF isn't available. I couldn't even compile the code to verify my MPI logic without spending hours installing dependencies."

**Interviewer:** "So, how did you resolve this?"

**You:** "I implemented a modular compilation strategy. I wrapped all NetCDF-specific calls in C++ preprocessor guards (`#ifdef _PNETCDF`). In the `CMakeLists.txt` file I created, I made the PNetCDF linking conditional. Now, if the library is missing, the code still compiles and runs, simply printing 'Output disabled' instead of crashing. This reduced the setup time for a new developer from hours to basically zero."

**Interviewer:** "Good. Now, on the physics side, you mentioned 'Strang Splitting'. Why use that?"

**You:** "Great question. We want 2nd-order accuracy in time. A naive implementation would need a complex multi-dimensional Riemann solver. Dimensional splitting simplifies this by treating the X and Z directions sequentially.
Ideally, to advance from time $t$ to $t+\Delta t$, we do an X-step for $\Delta t$ and then a Z-step for $\Delta t$. But that's only 1st-order accurate.
Strang Splitting alternates the order: Step $n$ is $X(\Delta t/2) \to Z(\Delta t) \to X(\Delta t/2)$. Or more commonly in this code, we alternate the full sequence: one step is $X \to Z$, the next is $Z \to X$. This cancels out the splitting error over time, achieving 2nd-order accuracy without the complexity of a fully coupled solver."

**Interviewer:** "I see you used manual packing for MPI. Why not use MPI Derived Datatypes (like `MPI_Type_vector`)?"

**You:** "That's a classic trade-off. While `MPI_Type_vector` is cleaner code-wise, implementations can sometimes suffer from performance overhead depending on the MPI library (OpenMPI vs MPICH vs Cray-MPI) and how it handles non-contiguous memory access.
In `miniWeather`, the data layout is such that vertical columns are contiguous, but we need to exchange vertical strips for the X-direction halos. Since we are already memory-bound, I chose to write an explicit packing kernel. This ensures the CPU prefetcher works efficiently to stream data into a contiguous buffer before handing it off to the network card. It gives me predictable performance across different platforms."

**Interviewer:** "How do you know your refactoring didn't break the physics?"

**You:** "I relied on the conservation properties of the finite volume method. After every simulation, the code effectively integrates mass and energy across the entire domain using `MPI_Allreduce`.
I verified that `d_mass` (change in mass) remained at machine precision ($\approx 10^{-15}$), which proves that my MPI ghost cell exchanges were correct—if I had lost even a single floating-point number at the boundary, mass would leak.
For energy, I observed a small drift of $10^{-4}$, which is the expected behavior of the hyper-viscosity diffusion term I kept enabled to dampen shockwaves."
