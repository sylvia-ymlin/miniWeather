# miniWeather Parallel Programming Study

Learning project analyzing parallel programming strategies for atmospheric flow simulation, based on ORNL's miniWeather mini-app.

## 🎯 Learning Focus

- **Numerical methods**: 4th-order finite-volume, 3rd-order Runge-Kutta time integration
- **Parallel strategies**: Serial → MPI → MPI+OpenMP → GPU (OpenACC/OpenMP4.5)
- **Performance analysis**: Scaling, communication overhead, conservation properties

## 📊 Performance Results

Tested on Mac M-series with OpenMPI:
- **Problem size**: 100×50 grid, 1000s simulation
- **MPI scaling**: 3.21× speedup with 4 processes (80.2% parallel efficiency)
- **Communication overhead**: ~20% for this problem size
- **Conservation**: Mass conserved to machine precision (10^-14), energy error < 10^-4

See [性能测试结果.md](性能测试结果.md) for detailed results and analysis.

## 📚 Key Learning Outcomes

### 1. Domain Decomposition (MPI Parallelization)
- Understood 1D domain splitting along X-direction
- Each MPI rank processes local subdomain with periodic boundaries
- Analyzed load balancing for uniform grids

### 2. Halo Exchange Communication
- Studied non-blocking communication patterns (`MPI_Isend`/`MPI_Irecv`)
- 2-cell halo regions required for 4th-order spatial stencil
- Measured communication overhead (~20% for 4 processes, 100×50 grid)

### 3. Global Reductions
- Analyzed collective operations for conservation checks
- `MPI_Allreduce` for global mass and energy computation
- Verified conservation properties: d_mass ~ 10^-14 (machine precision)

### 4. Numerical Stability and Accuracy
- **Hyper-viscosity**: Understood stabilization technique (hv_beta=0.05)
- **Strang splitting**: Why alternating X-Z sweeps gives second-order accuracy
- **CFL condition**: Analyzed stability constraint (CFL=1.5, max_speed=450 m/s)

### 5. Performance Trade-offs
- Identified compute vs communication balance
- Strong scaling: 80.2% efficiency indicates low communication overhead
- Problem-size dependency: Communication becomes less significant for larger grids

## 🔬 Technical Details

### Numerical Scheme
- **Spatial discretization**: 4th-order finite-volume with Gauss-Legendre quadrature (3-point)
- **Time integration**: 3rd-order Runge-Kutta with Strang splitting
- **Stabilization**: Hyper-viscosity to suppress numerical oscillations
- **Boundary conditions**: Periodic with 2-cell halo regions

### Parallel Implementations Analyzed
1. **Serial**: Baseline implementation (~500 lines)
2. **MPI**: Domain decomposition with halo exchange
3. **MPI+OpenMP**: Hybrid parallelism for NUMA architectures
4. **MPI+OpenMP4.5**: GPU offloading with target directives
5. **MPI+OpenACC**: Alternative GPU acceleration approach

### Code Structure
- **State variables**: Density, momentum (u,w), potential temperature
- **Time stepping**: `perform_timestep()` → alternating X/Z sweeps
- **Conservative fluxes**: `compute_tendencies_x/z()` with flux calculations
- **I/O**: Parallel NetCDF for scalable output (disabled in performance tests)

## 📖 Sources and Attribution

**Original Project**: [miniWeather by Matthew Norman (ORNL)](https://github.com/mrnorman/miniWeather)

**License**: BSD 2-Clause (see [LICENSE](LICENSE))

**Learning Materials**:
- miniWeather README and documentation
- MPI and OpenMP programming guides
- Performance analysis tutorials

---

## 📝 Documentation in This Repository

- [性能测试结果.md](性能测试结果.md) - Performance benchmarks and scaling analysis
- [C串行版本代码详解.md](C串行版本代码详解.md) - Detailed code walkthrough
- [学习指南.md](学习指南.md) - Study notes and learning path

---

**Important Note**: This is a learning project focused on understanding parallel programming techniques. The original implementation is by Matthew Norman at Oak Ridge National Laboratory. My work involves:
- Running performance tests on different configurations
- Analyzing parallel efficiency and communication patterns
- Documenting numerical methods and algorithm design
- Creating Chinese language study materials for better understanding

All core algorithm implementation credit goes to the original author.

