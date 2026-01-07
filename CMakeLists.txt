cmake_minimum_required(VERSION 3.10)
project(miniWeather CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP (optional)
find_package(OpenMP)

# Default Parameters
if(NOT DEFINED NX)
    set(NX 100)
endif()
if(NOT DEFINED NZ)
    set(NZ 50)
endif()
if(NOT DEFINED SIM_TIME)
    set(SIM_TIME 1000)
endif()
if(NOT DEFINED OUT_FREQ)
    set(OUT_FREQ 10)
endif()
if(NOT DEFINED DATA_SPEC)
    set(DATA_SPEC 2) # DATA_SPEC_THERMAL = 2
endif()

# Compiler Definitions
add_definitions(-D_NX=${NX})
add_definitions(-D_NZ=${NZ})
add_definitions(-D_SIM_TIME=${SIM_TIME})
add_definitions(-D_OUT_FREQ=${OUT_FREQ})
add_definitions(-D_DATA_SPEC=${DATA_SPEC})

# ============================================================================
# Target 1: Serial Baseline (No parallelism)
# ============================================================================
add_executable(miniWeather_serial miniWeather_serial.cpp)

# ============================================================================
# Target 2: MPI + OpenMP (Hybrid CPU Parallelism)
# ============================================================================
add_executable(miniWeather_mpi miniWeather_mpi.cpp)
target_link_libraries(miniWeather_mpi PUBLIC MPI::MPI_CXX)
if(OpenMP_CXX_FOUND)
    target_link_libraries(miniWeather_mpi PUBLIC OpenMP::OpenMP_CXX)
endif()

# ============================================================================
# Target 3: MPI + OpenACC (GPU Offloading via OpenACC)
# Requires NVIDIA HPC SDK (nvc++) or PGI compiler
# Build: cmake -DCMAKE_CXX_COMPILER=nvc++ -DENABLE_OPENACC=ON ..
# ============================================================================
option(ENABLE_OPENACC "Build OpenACC GPU version" OFF)
if(ENABLE_OPENACC)
    add_executable(miniWeather_openacc miniWeather_mpi_openacc.cpp)
    target_link_libraries(miniWeather_openacc PUBLIC MPI::MPI_CXX)
    target_compile_options(miniWeather_openacc PRIVATE -acc -gpu=managed -Minfo=accel)
    message(STATUS "OpenACC target enabled. Requires nvc++ compiler.")
endif()

# ============================================================================
# Target 4: MPI + OpenMP 4.5 Target (GPU Offloading via OpenMP)
# Requires Clang/LLVM with offloading support or NVIDIA HPC SDK
# Build: cmake -DCMAKE_CXX_COMPILER=clang++ -DENABLE_OMP_TARGET=ON ..
# ============================================================================
option(ENABLE_OMP_TARGET "Build OpenMP Target GPU version" OFF)
if(ENABLE_OMP_TARGET)
    add_executable(miniWeather_omp_target miniWeather_mpi_openmp45.cpp)
    target_link_libraries(miniWeather_omp_target PUBLIC MPI::MPI_CXX)
    # Adjust flags based on your compiler (example for nvc++)
    target_compile_options(miniWeather_omp_target PRIVATE -mp=gpu -Minfo=mp)
    message(STATUS "OpenMP Target offloading enabled. Requires compatible compiler.")
endif()

# ============================================================================
# Testing
# ============================================================================
enable_testing()
add_test(NAME ValidationTest 
         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/validate.py 
                 --exe $<TARGET_FILE:miniWeather_serial> --nx 100 --nz 50 --time 5)
add_test(NAME MPI_Test COMMAND mpiexec -n 2 ./miniWeather_mpi)

