cmake_minimum_required(VERSION 3.16)

project(ParallelStochasticFiltering LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# packages:
find_package(OpenMP REQUIRED)

# sim the state space:
add_executable(stochastic_filter src/simulate_state_space.cc)

# filters:
add_executable(kalman_filter src/kalman_filter.cc)
add_executable(particle_filter_serial src/particle_filter_serial.cc)
add_executable(particle_filter_omp src/particle_filter_openmp.cc)

# link libs:
target_link_libraries(particle_filter_omp PRIVATE OpenMP::OpenMP_CXX)