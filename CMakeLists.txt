cmake_minimum_required(VERSION 3.20)

# Select a supported CUDA host compiler before project() enables CUDA.
# With GCC 16 or newer, prefer g++-15; otherwise enable the compiler override
# and use C++20 for CUDA, since newer libstdc++ headers require it.
find_program(_cxx_compiler NAMES $ENV{CXX} c++ g++)
if(_cxx_compiler)
    execute_process(COMMAND ${_cxx_compiler} -dumpversion
                    OUTPUT_VARIABLE _gcc_version OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
endif()
if(_gcc_version VERSION_GREATER_EQUAL 16)
    find_program(_gcc15 NAMES g++-15)
    if(_gcc15)
        set(CMAKE_CUDA_HOST_COMPILER "${_gcc15}")
    else()
        set(CMAKE_CUDA_FLAGS_INIT "-allow-unsupported-compiler -std=c++20")
    endif()
endif()

project(qn_dmrg_contraction LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_STANDARD 17)
else()
    set(CMAKE_CUDA_STANDARD 20)
endif()
set(CMAKE_CUDA_ARCHITECTURES native)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

enable_testing()

add_library(qnd INTERFACE)
target_include_directories(qnd INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(test_unit tests/test_unit.cpp)
target_link_libraries(test_unit PRIVATE qnd)
target_compile_options(test_unit PRIVATE -Wall -Wextra)
add_test(NAME unit COMMAND test_unit)

add_executable(qnd_ref src/qnd_ref.cu)
target_link_libraries(qnd_ref PRIVATE qnd CUDA::cublas)
target_compile_options(qnd_ref PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)

add_executable(qnd_gate1 src/qnd_gate1.cu)
target_link_libraries(qnd_gate1 PRIVATE qnd CUDA::cublas)
target_compile_options(qnd_gate1 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)

# Gate 0 across the preset profiles at small-to-moderate chi; larger chi
# runs are a manual driver invocation.
foreach(case "tj-L64:32" "tj-L64:64" "tj-L64:256" "tj-L256:64"
             "hubbard-hf-L64:64" "cylinder-w4:64")
  string(REPLACE ":" ";" parts ${case})
  list(GET parts 0 prof)
  list(GET parts 1 chi)
  add_test(NAME gate0_${prof}_${chi} COMMAND qnd_ref ${prof} ${chi})
endforeach()
