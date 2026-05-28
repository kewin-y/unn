# unn

Neural networks experiment, or notes from [NNFS](https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3).

## Build

```sh
mkdir -p build
cd build
cmake ..
cmake --build .
```

If Eigen is already installed as a CMake package, `unn` will use it. If not, the default `UNN_FETCH_EIGEN=ON` fallback downloads Eigen automatically.

## Install

```sh
mkdir -p build
cd build
cmake ..
cmake --build .
cmake --install . --prefix /your/install/prefix
```

This installs:

- headers under `include/unn`
- the `unn` library
- CMake package files under `lib/cmake/unn`

## Consume From CMake

In a downstream project:

```cmake
find_package(unn CONFIG REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE unn::unn)
```

When configuring the downstream project, point CMake at the install prefix if needed:

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=/your/install/prefix
```

## Options

- `UNN_BUILD_EXAMPLES`: build the example executable (`ON` by default)
- `UNN_FETCH_EIGEN`: fetch Eigen if it is not already installed (`ON` by default)
