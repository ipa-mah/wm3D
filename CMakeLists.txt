cmake_minimum_required (VERSION 3.1)
project(wm3D VERSION 0.1.0 LANGUAGES CXX)

# We need a CMAKE_DIR with some code to find external dependencies
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
set(CMAKE_CXX_STANDARD 14)
#find_package(OpenCV REQUIRED NO_MODULE PATHS /usr/local NO_DEFAULT_PATH)
find_package(OpenCV REQUIRED)
MESSAGE(STATUS "OpenCV included at ${OpenCV_INCLUDE_DIRS}")
find_package(OpenMP)
if (OPENMP_FOUND)
    MESSAGE("OpenMP DIR : ${OpenMP_DIR}")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
find_package(Boost REQUIRED)
find_package(CUDA 9.0 REQUIRED)
CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
list(APPEND CUDA_NVCC_FLAGS "--compiler-options -fPIC")
message("ARCH_FLAGS = ${ARCH_FLAGS}")

find_package(GLEW REQUIRED)
find_package(PCL REQUIRED)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
find_package(OpenGL REQUIRED)
find_package(GLFW3 REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(realsense2 REQUIRED )
#Open3D

find_package(Open3D HINTS ${CMAKE_INSTALL_PREFIX}/lib/CMake)
list(APPEND Open3D_LIBRARIES dl)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${Open3D_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Open3D_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${Open3D_EXE_LINKER_FLAGS}")

message(STATUS "Found Open3D ${Open3D_VERSION}")
# link_directories must be before add_executable
link_directories(${Open3D_LIBRARY_DIRS})

# tinyobjectloader
include(ExternalProject)
add_subdirectory(elibs)
include_directories(${CMAKE_SOURCE_DIR}/elibs)

##########

include_directories(
    include
    ${Boost_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS} 
    ${Open3D_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIRS}
    ${GLUT_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${realsense_INCLUDE_DIR}
    )
file(GLOB sources
    
    "${CMAKE_SOURCE_DIR}/src/cuda/*.cu"
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/utility/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/visualization/*.cpp"
    )
cuda_add_library(${PROJECT_NAME} ${sources} )

target_link_libraries(
    ${PROJECT_NAME}
    ${Boost_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    ${Open3D_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${PCL_LIBRARIES}
    ${GLFW_LIBRARIES}
    ${GLUT_LIBRARY}
    ${GLEW_LIBRARIES}
    tinyobjloader
    ${realsense2_LIBRARIES}
    )

add_executable(mesh_rgbd_rendering test/mesh_rgbd_rendering.cpp)
target_link_libraries(mesh_rgbd_rendering ${PROJECT_NAME})


#add_executable(opengl_test test/opengl_test.cpp)
#target_link_libraries(opengl_test ${PROJECT_NAME})
#
add_executable(function_test test/function_test.cpp)
target_link_libraries(function_test ${PROJECT_NAME})


# Install project files

# copy directory of header files to /usr/local/include/
install(DIRECTORY include/${PROJECT_NAME}
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
    PATTERN "*.svn" EXCLUDE)
#copy file package.xml to /usr/local/share
install (FILES package.xml DESTINATION share/$(PROJECT_NAME))
# install CMake targets and config files



# uninstall target
if (NOT TARGET uninstall)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY)

    add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P
        ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif ()

