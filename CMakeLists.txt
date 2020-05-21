cmake_minimum_required(VERSION 3.15)
project(study)

set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)


add_executable(study main.cpp prediction.cpp prediction.h)
target_link_libraries(study "${TORCH_LIBRARIES}")

set_property(TARGET study PROPERTY CXX_STANDARD 14)