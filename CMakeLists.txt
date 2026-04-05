cmake_minimum_required(VERSION 3.18)
project(hog_detector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type defaults to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Profiling flag (cmake -DENABLE_PROFILING=ON ..)
option(ENABLE_PROFILING "Compile with gprof instrumentation" OFF)
if(ENABLE_PROFILING)
    add_compile_options(-pg)
    add_link_options(-pg)
endif()

# Release flags
add_compile_options(
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-march=native>
    $<$<CONFIG:RelWithDebInfo>:-O2>
    $<$<CONFIG:RelWithDebInfo>:-g>
)

# OpenCV — image I/O and video capture only
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)

# OpenMP — CPU parallelism for Phase 2A
find_package(OpenMP REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

include_directories(project/include)

# ── Main detector executable ─────────────────────────────────────────────────
set(DETECTOR_SOURCES
    project/src/hog.cpp
    project/src/svm.cpp
    project/src/nms.cpp
    project/src/sliding_window.cpp
    project/src/sliding_window_omp.cpp
    project/src/main.cpp
)

add_executable(hog_detector ${DETECTOR_SOURCES})
target_link_libraries(hog_detector PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

# ── HOG correctness test ──────────────────────────────────────────────────────
add_executable(test_hog project/test/test_hog.cpp project/src/hog.cpp project/src/svm.cpp)
target_link_libraries(test_hog PRIVATE ${OpenCV_LIBS})

# Print build summary
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Profiling:  ${ENABLE_PROFILING}")
