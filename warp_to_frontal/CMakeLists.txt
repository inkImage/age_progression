project(warp_to_fronal)

find_package(OpenCV REQUIRED)

set(WARPFRONT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(WARPFRONT_INCLUDE_DIR "${WARP_DIR}/include")
set(WARPFRONT_SOURCE_DIR  "${WARP_DIR}/src")

file(GLOB WARPFRONT_HEADERS "${WARPFRONT_INCLUDE_DIR}/*.h" "${WARPFRONT_INCLUDE_DIR}/*.hpp")
file(GLOB WARPFRONT_SOURCES "${WARPFRONT_SOURCE_DIR}/*.cpp")

add_library(warpfront)
add_executable( warpfront_test ${sources} ${headers})
target_link_libraries(warpfront ${OpenCV_LIBRARIES})
target_link_libraries( warp_test warpfront ${OpenCV_LIBRARIES})