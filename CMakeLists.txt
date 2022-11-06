cmake_minimum_required(VERSION 3.10)

project(NumericOptim CXX)

set(CMAKE_C_STANDARD 11)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/src/main.cpp)

file(GLOB UNCONSTRAINED_INCLUDE ${PROJECT_SOURCE_DIR}/include/**/*.h)

target_include_directories(${PROJECT_NAME} PUBLIC UNCONSTRAINED_INCLUDE)
