cmake_minimum_required(VERSION 3.30)

project(micro_torch)

set(BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${BINARY_DIR})

add_subdirectory(libs/glog)
add_subdirectory(libs/gtest)

set(EXECUTABLE_OUTPUT_PATH ${BINARY_DIR})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

file(GLOB SRC_FILES src/[!main]*.cpp) # list all files except for main.cpp
add_library(${PROJECT_NAME} ${SRC_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC include libs/glog/src)
target_link_libraries(${PROJECT_NAME} PUBLIC glog)

install(TARGETS ${PROJECT_NAME} DESTINATION lib)

add_subdirectory(tests)

set(EXECUTABLE_NAME "micro_torch_main")
add_executable(${EXECUTABLE_NAME} src/main.cpp)
target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${PROJECT_NAME})