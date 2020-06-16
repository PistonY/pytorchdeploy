cmake_minimum_required(VERSION 3.15)
project(pytorchdeploy)

set(CMAKE_CXX_STANDARD 14)
set(Torch_DIR /media/piston/data/cuda/libtorch-cxx11-abi-shared-with-deps-1.5.0/libtorch/share/cmake/Torch)
#set(Torch_DIR /libtorch/share/cmake/Torch)
find_package(OpenCV 4 REQUIRED)
find_package(Torch REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(pytorchdeploy main.cpp prediction.cpp prediction.h transform.h)

target_link_libraries(pytorchdeploy "${TORCH_LIBRARIES}")
target_link_libraries(pytorchdeploy "${OpenCV_LIBS}")

set_property(TARGET pytorchdeploy PROPERTY CXX_STANDARD 14)