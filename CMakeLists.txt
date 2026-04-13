cmake_minimum_required(VERSION 3.20)

project(unn VERSION 0.1.0 LANGUAGES CXX)

include(CMakePackageConfigHelpers)
include(FetchContent)
include(GNUInstallDirs)

option(UNN_BUILD_EXAMPLES "Build unn example programs" ON)
option(UNN_FETCH_EIGEN "Fetch Eigen when it is not already installed" ON)

find_package(Eigen3 CONFIG QUIET)

if(NOT Eigen3_FOUND)
  if(NOT UNN_FETCH_EIGEN)
    message(FATAL_ERROR
      "Eigen3 was not found. Install Eigen3 or configure with -DUNN_FETCH_EIGEN=ON."
    )
  endif()

  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
  )
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(eigen)
endif()

add_library(unn
  src/Layer_Dense.cpp
  src/ReLU.cpp
  src/Softmax.cpp
  src/Loss_CCE.cpp
)

add_library(unn::unn ALIAS unn)

target_compile_features(unn PUBLIC cxx_std_17)

target_include_directories(unn
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(unn PUBLIC Eigen3::Eigen)

set_target_properties(unn PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  EXPORT_NAME unn
)

if(UNN_BUILD_EXAMPLES)
  add_executable(unn_demo examples/main.cpp)
  target_link_libraries(unn_demo PRIVATE unn::unn)
endif()

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/unnConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/unnConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/unnConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/unn"
)

install(TARGETS unn
  EXPORT unnTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/unnConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/unnConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/unn"
)

install(EXPORT unnTargets
  FILE unnTargets.cmake
  NAMESPACE unn::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/unn"
)
