cmake_minimum_required(VERSION 3.27)

project(_ext LANGUAGES CXX)

# ----------------------------- Setup -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

# ----------------------------- Dependencies -----------------------------
find_package(
  Python
  COMPONENTS Interpreter Development.Module
  REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)

# ----------------------------- Extensions -----------------------------

# Auto-discover kernel sources so new files don't require CMake edits.
file(
  GLOB KERNEL_CPP_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/csrc/*.cpp")
list(REMOVE_ITEM KERNEL_CPP_SOURCES "${CMAKE_CURRENT_LIST_DIR}/csrc/bindings.cpp")

# Add library
add_library(mlx_serve_kernel)

# Add sources
target_sources(mlx_serve_kernel PUBLIC ${KERNEL_CPP_SOURCES})

# Add include headers
target_include_directories(mlx_serve_kernel PUBLIC ${CMAKE_CURRENT_LIST_DIR}/csrc)

# Link to mlx
target_link_libraries(mlx_serve_kernel PUBLIC mlx)

# ----------------------------- Metal -----------------------------

file(
  GLOB KERNEL_METAL_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/csrc/*.metal")

# Build metallib
if(MLX_BUILD_METAL)
  mlx_build_metallib(
    TARGET
    mlx_ext_metallib
    TITLE
    mlx_serve_kernel
    SOURCES
    ${KERNEL_METAL_SOURCES}
    INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}
    ${MLX_INCLUDE_DIRS}
    OUTPUT_DIRECTORY
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

  add_dependencies(mlx_serve_kernel mlx_ext_metallib)

endif()

# ----------------------------- Python Bindings -----------------------------
nanobind_add_module(
  _ext
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN
  mlx
  ${CMAKE_CURRENT_LIST_DIR}/csrc/bindings.cpp)
target_link_libraries(_ext PRIVATE mlx_serve_kernel)

if(BUILD_SHARED_LIBS)
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
endif()