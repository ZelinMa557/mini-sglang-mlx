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
add_library(mini_sglang_mlx_kernel)

# Add sources
target_sources(mini_sglang_mlx_kernel PUBLIC ${KERNEL_CPP_SOURCES})

# Add include headers
target_include_directories(mini_sglang_mlx_kernel PUBLIC ${CMAKE_CURRENT_LIST_DIR}/csrc)

# Link to mlx
target_link_libraries(mini_sglang_mlx_kernel PUBLIC mlx)

# ----------------------------- Metal -----------------------------

file(
  GLOB KERNEL_METAL_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/csrc/*.metal")

if(MLX_BUILD_METAL)
  # Build metallib for kernels without MPP cooperative tensors.
  mlx_build_metallib(
    TARGET
    mlx_ext_metallib
    TITLE
    mini_sglang_mlx_kernel
    SOURCES
    ${KERNEL_METAL_SOURCES}
    INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}
    ${MLX_INCLUDE_DIRS}
    OUTPUT_DIRECTORY
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

  add_dependencies(mini_sglang_mlx_kernel mlx_ext_metallib)

  # The MPP cooperative tensor layout depends on the Metal language version
  # used at compile time. Offline `xcrun metal` compilation produces a
  # layout that does not match the MPP runtime on this OS/Xcode, which
  # corrupts the MMA operand placement. Compiling the kernel sources at
  # runtime via MTL::newLibrary with MTL::LanguageVersion4_0 (the same
  # path MLX uses for its own kernels) produces the correct layout.
  # Generate an embedded copy of the sources for runtime JIT compilation.
  set(JIT_SOURCE_CPP "${CMAKE_CURRENT_BINARY_DIR}/jit_sources.cpp")
  add_custom_command(
    OUTPUT ${JIT_SOURCE_CPP}
    COMMAND
      ${Python_EXECUTABLE} "${CMAKE_CURRENT_LIST_DIR}/tools/gen_jit_source.py"
      ${JIT_SOURCE_CPP}
      "${CMAKE_CURRENT_LIST_DIR}/csrc/mlx_metal_utils_min.h"
      ${KERNEL_METAL_SOURCES}
    DEPENDS
      ${CMAKE_CURRENT_LIST_DIR}/tools/gen_jit_source.py
      ${CMAKE_CURRENT_LIST_DIR}/csrc/mlx_metal_utils_min.h
      ${KERNEL_METAL_SOURCES}
    COMMAND_EXPAND_LISTS
    VERBATIM
    COMMENT "Generating embedded metal sources for runtime JIT")
  add_custom_target(mini_sglang_mlx_kernel_jit_sources DEPENDS ${JIT_SOURCE_CPP})
  add_dependencies(mini_sglang_mlx_kernel mini_sglang_mlx_kernel_jit_sources)
  target_sources(mini_sglang_mlx_kernel PRIVATE ${JIT_SOURCE_CPP})
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
target_link_libraries(_ext PRIVATE mini_sglang_mlx_kernel)

if(BUILD_SHARED_LIBS)
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
endif()