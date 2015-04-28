project(age_progression CXX)

cmake_minimum_required(VERSION 2.8)

set(ASM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/asmlib-opencv-read-only/src/lib")
set(WARP_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/warp/include")

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:wchar_t-")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Werror -Wall -Wextra -Wno-error=unused-parameter -pedantic")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "-std=c++11 -stdlib=libc++")
endif()

add_subdirectory(${ASM_SOURCE_DIR})
add_subdirectory(warp)
add_subdirectory(apps)


