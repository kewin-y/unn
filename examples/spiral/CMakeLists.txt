cmake_minimum_required(VERSION 3.20)

project(unnSpiralExample LANGUAGES CXX)

include(../../cmake/CPM.cmake)

CPMAddPackage(NAME unn SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../..)

set(sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

add_executable(${PROJECT_NAME} ${sources})

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)

target_link_libraries(${PROJECT_NAME} unn::unn)
