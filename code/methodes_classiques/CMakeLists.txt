cmake_minimum_required(VERSION 3.16)
project(methodes_classiques)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(library)

add_executable(methodes_classiques main.cpp)
target_link_libraries(methodes_classiques PUBLIC lib_denoising)
add_custom_command(TARGET methodes_classiques PRE_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_directory
	${CMAKE_SOURCE_DIR}/images/ $<TARGET_FILE_DIR:methodes_classiques>/images
)
