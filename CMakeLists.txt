cmake_minimum_required(VERSION 3.30)

project(micro_torch)

file(GLOB SRC_FILES src/*.cpp)

add_subdirectory(libs/glog)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
add_executable(${PROJECT_NAME} ${SRC_FILES})
target_include_directories(${PROJECT_NAME} PRIVATE include libs/glog/src)
target_link_libraries(${PROJECT_NAME} PRIVATE glog)