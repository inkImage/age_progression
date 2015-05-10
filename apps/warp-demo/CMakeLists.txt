project(warp-demo)

find_package(OpenCV REQUIRED)

set(WARP_DEMO_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

set(HEADERS fold.hpp )
set(SOURCES warp-demo.cpp fold.cpp )

include_directories(${WARP_DEMO_DIR} ${WARP_INCLUDE_DIR} "${ASM_SOURCE_DIR}" ${FACE_SDK_SRC_DIR} "${OpenCV_INCLUDE_DIRS}")

add_executable( warp-demo ${SOURCES} ${HEADERS})
target_link_libraries( warp-demo warp asm clmTracker utilities avatarAnim ${OpenCV_LIBRARIES})
