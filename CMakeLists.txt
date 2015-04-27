project(age_progression CXX)

cmake_minimum_required(VERSION 2.8)

set(ASM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/asmlib-opencv-read-only/src/lib")
set(WARP_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/warp/include")

add_subdirectory(${ASM_SOURCE_DIR})
add_subdirectory(warp)
add_subdirectory(apps)


