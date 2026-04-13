cmake_minimum_required(VERSION 3.20)

project(unn_package_smoke LANGUAGES CXX)

find_package(unn CONFIG REQUIRED)

add_executable(unn_package_smoke main.cpp)
target_link_libraries(unn_package_smoke PRIVATE unn::unn)
