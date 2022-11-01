cmake_minimum_required(VERSION 3.10)
project(NumericOptim CXX)

set(CMAKE_C_STANDARD 11)

find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(${PROJECT_NAME} src/main.cpp)

file(GLOB unconstrained_include ${PROJECT_SOURCE_DIR}/include/**/*.h)

target_include_directories(${PROJECT_NAME} PUBLIC unconstrained_include)
