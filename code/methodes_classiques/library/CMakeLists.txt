set(SOURCES lib.cpp include/lib.h)

find_package(OpenCV REQUIRED)
add_library(lib_denoising STATIC ${SOURCES})
target_link_libraries(lib_denoising PUBLIC ${OpenCV_LIBS})
target_include_directories(lib_denoising PUBLIC include)

message(STATUS ${OpenCV_LIBS})
