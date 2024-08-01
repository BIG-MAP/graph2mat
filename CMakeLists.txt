cmake_minimum_required(VERSION 3.21)
# We will use cmake_path for file-name manipulation

project(${SKBUILD_PROJECT_NAME} LANGUAGES C)

find_package(
  Python
  COMPONENTS Interpreter Development.Module
  NumPy
  REQUIRED)

find_program(CYTHON "cython")

add_subdirectory("src")
