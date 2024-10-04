cmake_minimum_required(VERSION 3.30)

project(micro_torch)

file(GLOB SRC_FILES src/*.cpp)

add_executable(${PROJECT_NAME} ${SRC_FILES})
target_include_directories(${SRC_FILES} PRIVATE include)