cmake_minimum_required(VERSION 3.31...4.0)

include (FetchContent)

# Project
project(poops LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen
  GIT_TAG 5.0.0
)

FetchContent_MakeAvailable(eigen)

set(SOURCE_FILES
  src/Layer_Dense.cpp
  src/main.cpp
  src/ReLU.cpp
  src/Softmax.cpp
  src/Loss_CCE.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries (${PROJECT_NAME} PRIVATE Eigen3::Eigen)
