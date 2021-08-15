cmake_minimum_required(VERSION 3.8)

project(YoloV4Plugin LANGUAGES CXX CUDA)

set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_30;code=sm_30)

include_directories(${PROJECT_SOURCE_DIR}/include)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda-10.2/include)
    link_directories(/usr/local/cuda/cuda-10.2/lib64)
else()
    message("embed_platform off")
    include_directories(/usr/local/cuda-11.3/include)
    link_directories(/usr/local/cuda-11.3/lib64)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

# Sets variable to a value if variable is unset.
macro(set_ifndef var val)
    if (NOT ${var})
        set(${var} ${val})
    endif()
    message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

# -------- CONFIGURATION --------
set_ifndef(TRT_LIB /usr/lib/aarch64-linux-gnu)
set_ifndef(TRT_INCLUDE /usr/include/aarch64-linux-gnu)


# Find dependencies:
message("\nThe following variables are derived from the values of the previous variables unless provided explicitly:\n")

# TensorRT's nvinfer lib
find_library(_NVINFER_LIB nvinfer HINTS ${TRT_LIB} PATH_SUFFIXES lib lib64)
set_ifndef(NVINFER_LIB ${_NVINFER_LIB})

# -------- BUILDING --------

# Add include directories
include_directories(${TRT_INCLUDE} ${CMAKE_SOURCE_DIR})


# Define clip plugin library target
add_library(yolov4plugin MODULE
  ${CMAKE_SOURCE_DIR}/yololayer.cu
  ${CMAKE_SOURCE_DIR}/yololayer.h
  ${CMAKE_SOURCE_DIR}/yolov4layer.cpp
)

# Use C++11
target_compile_features(yolov4plugin PUBLIC cxx_std_11)

# Link TensorRT's nvinfer lib
target_link_libraries(yolov4plugin PRIVATE ${NVINFER_LIB})

# We need to explicitly state that we need all CUDA files
# to be built with -dc as the member functions will be called by
# other libraries and executables (in our case, Python inference scripts)
set_target_properties(yolov4plugin PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)



