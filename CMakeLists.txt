cmake_minimum_required(VERSION 3.10)
project(miniWeather CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find MPI
find_package(MPI REQUIRED)

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

# MPI Executable
add_executable(miniWeather_mpi miniWeather_mpi.cpp)
if(OpenMP_CXX_FOUND)
    target_link_libraries(miniWeather_mpi PUBLIC OpenMP::OpenMP_CXX)
endif()
target_link_libraries(miniWeather_mpi PUBLIC MPI::MPI_CXX)

# Optional: Serial Executable (only if I fix the code later, currently disabled to avoid build errors)
add_executable(miniWeather_serial miniWeather_serial.cpp)

enable_testing()
add_test(NAME ValidationTest COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/validate.py --exe $<TARGET_FILE:miniWeather_serial> --nx 100 --nz 50 --time 5)
add_test(NAME MPI_Test COMMAND mpiexec -n 2 ./miniWeather_mpi)
