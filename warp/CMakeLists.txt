project(warp)

find_package(OpenCV REQUIRED)

set(WARP_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(WARP_INCLUDE_DIR "${WARP_DIR}/include")
set(WARP_SOURCE_DIR  "${WARP_DIR}/src")

set(WARP_HEADERS ${WARP_INCLUDE_DIR}/warp.hpp ${WARP_INCLUDE_DIR}/fold.hpp ${WARP_INCLUDE_DIR}/util.hpp)
set(WARP_SOURCES ${WARP_SOURCE_DIR}/warp.cpp ${WARP_SOURCE_DIR}/fold.cpp ${WARP_SOURCE_DIR}/util.cpp)

include_directories(${OpenCV_INCLUDE_DIRS} ${WARP_INCLUDE_DIR} ${ASM_SOURCE_DIR} ${FACE_SDK_SRC_DIR})

add_library(warp ${WARP_SOURCES} ${WARP_HEADERS} )

target_link_libraries(warp asm clmTracker utilities avatarAnim ${OpenCV_LIBRARIES})

