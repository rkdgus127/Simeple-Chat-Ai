cmake_minimum_required(VERSION 4.4)

project(Ai LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

add_executable(Ai
        src/main.cpp
        src/core/Tensor.cpp
        src/core/Tensor.h
        src/core/Autograd.cpp
        src/core/Autograd.h
)