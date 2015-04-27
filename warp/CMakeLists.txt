project(warp)

find_package(OpenCV REQUIRED)

set(WARP_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(WARP_INCLUDE_DIR "${WARP_DIR}/include")
set(WARP_SOURCE_DIR  "${WARP_DIR}/src")

set(WARP_HEADERS ${WARP_INCLUDE_DIR}/warp.hpp)
set(WARP_SOURCES ${WARP_SOURCE_DIR}/warp.cpp)

include_directories(${OpenCV_INCLUDE_DIRS} ${WARP_INCLUDE_DIR} ${ASM_SOURCE_DIR})

add_library(warp ${WARP_SOURCES} ${WARP_HEADERS} )

target_link_libraries(warp asm ${OpenCV_LIBRARIES})

