cmake_minimum_required(VERSION 3.20)

project(BuildAll LANGUAGES CXX)

# needed to generate test target
enable_testing()

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../tests ${CMAKE_BINARY_DIR}/tests)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../examples/spiral ${CMAKE_BINARY_DIR}/examples/spiral)
