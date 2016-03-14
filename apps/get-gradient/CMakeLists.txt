project(get-gradient)

find_package(OpenCV REQUIRED)

set(GET_GRADIENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

set(HEADERS )
set(SOURCES get-gradient.cpp )

include_directories(${GET_GRADIENT_DIR} ${WARP_INCLUDE_DIR} "${ASM_SOURCE_DIR}" ${FACE_SDK_SRC_DIR} "${OpenCV_INCLUDE_DIRS}")

add_executable( get-gradient ${SOURCES} ${HEADERS})
target_link_libraries( get-gradient warp asm clmTracker utilities avatarAnim ${OpenCV_LIBRARIES})
