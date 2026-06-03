cmake_minimum_required(VERSION 3.20)

project(unn VERSION 0.1.0 LANGUAGES CXX)

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
  )
endif()

include(cmake/CPM.cmake)

CPMAddPackage("gh:TheLartians/PackageProject.cmake@1.13.0")
CPMAddPackage(
  NAME Eigen
  VERSION 3.4.1
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.1/eigen-3.4.1.tar.gz
  DOWNLOAD_ONLY YES
)

if(Eigen_ADDED)
  add_library(Eigen INTERFACE IMPORTED)
  target_include_directories(Eigen INTERFACE ${Eigen_SOURCE_DIR})
endif()

set(headers
  ${CMAKE_CURRENT_SOURCE_DIR}/include/unn/Layer.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/unn/Layer_Dense.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/unn/Loss_CCE.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/unn/ReLU.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/unn/Softmax.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/unn/Softmax_And_Loss_CCE.hpp
)

set(sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Layer_Dense.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ReLU.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Softmax.cpp
)

add_library(${PROJECT_NAME} ${headers} ${sources})
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)

# being a cross-platform target, we enforce standards conformance on MSVC
target_compile_options(${PROJECT_NAME} PUBLIC "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/permissive->")

target_link_libraries(${PROJECT_NAME} PRIVATE Eigen)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${PROJECT_NAME}-${PROJECT_VERSION}>
)

string(TOLOWER ${PROJECT_NAME}/version.hpp VERSION_HEADER_LOCATION)

packageProject(
  NAME ${PROJECT_NAME}
  VERSION ${PROJECT_VERSION}
  NAMESPACE ${PROJECT_NAME}
  BINARY_DIR ${PROJECT_BINARY_DIR}
  INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include
  INCLUDE_DESTINATION include/${PROJECT_NAME}-${PROJECT_VERSION}
  VERSION_HEADER "${VERSION_HEADER_LOCATION}"
  COMPATIBILITY SameMajorVersion
  DEPENDENCIES "eigen 3.4.1"
)
