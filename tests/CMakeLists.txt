cmake_minimum_required(VERSION 3.14...3.22)

project(unnTests LANGUAGES CXX)


option(ENABLE_TEST_COVERAGE "Enable test coverage" OFF)
option(TEST_INSTALLED_VERSION "Test the version found by find_package" OFF)

include(../cmake/CPM.cmake)

CPMAddPackage("gh:doctest/doctest@2.5.2")
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

if(TEST_INSTALLED_VERSION)
  find_package(unn REQUIRED)
else()
  CPMAddPackage(NAME unn SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/..)
endif()

# file(GLOB sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
set(sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/smoke.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/layer_dense.cpp
)
add_executable(${PROJECT_NAME} ${sources})
target_link_libraries(${PROJECT_NAME} doctest::doctest unn::unn Eigen)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)

# enable compiler warnings
if(NOT TEST_INSTALLED_VERSION)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(unn PUBLIC -Wall -Wpedantic -Wextra -Werror)
  elseif(MSVC)
    target_compile_options(unn PUBLIC /W4 /WX)
    target_compile_definitions(${PROJECT_NAME} PUBLIC DOCTEST_CONFIG_USE_STD_HEADERS)
  endif()
endif()

enable_testing()

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests(${PROJECT_NAME})

if(ENABLE_TEST_COVERAGE)
  target_compile_options(unn PUBLIC -O0 -g -fprofile-arcs -ftest-coverage)
  target_link_options(unn PUBLIC -fprofile-arcs -ftest-coverage)
endif()
