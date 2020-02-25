cmake_minimum_required (VERSION 3.1)
project(perception_utils VERSION 0.1.0 LANGUAGES CXX)

# We need a CMAKE_DIR with some code to find external dependencies
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV REQUIRED NO_MODULE PATHS /usr/local NO_DEFAULT_PATH)
MESSAGE(STATUS "OpenCV included at ${OpenCV_INCLUDE_DIRS}")
find_package(OpenMP)
if (OPENMP_FOUND)
    MESSAGE("OpenMP DIR : ${OpenMP_DIR}")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
find_package(Boost REQUIRED)
find_package(GLEW REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLFW3 REQUIRED)
find_package(CUDA 9.0 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp)


#Open3D
find_package(Open3D REQUIRED)
link_directories(${Open3D_LIBRARY_DIRS})
include_directories(${Open3D_INCLUDE_DIRS} )
#MESSAGE(STATUS "Open3D included at ${Open3D_INCLUDE_DIRS}")

# tinyobjectloader
include(ExternalProject)
add_subdirectory(elibs)
include_directories(${CMAKE_SOURCE_DIR}/elibs)

##########

include_directories(
    common/include
    ${Boost_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIRS}
    ${GLUT_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
    )
file(GLOB sources "${CMAKE_SOURCE_DIR}/common/src/*.cpp"
                  "${CMAKE_SOURCE_DIR}/common/src/utility/*.cpp")

add_library(${PROJECT_NAME} ${sources} )

target_link_libraries(${PROJECT_NAME}
    ${Boost_LIBS}
    ${OpenCV_LIBS}
    ${OPENGL_LIBRARIES}
    ${PCL_LIBRARIES}
    ${GLFW_LIBRARIES}
    ${GLUT_LIBRARY}
    ${GLEW_LIBRARIES}
    # tinyobjloader::tinyobjloader
    tinyobjloader
    ${Open3D_LIBRARIES}


    )

add_executable(mesh_rgbd_rendering src/mesh_rgbd_rendering.cpp)
target_link_libraries(mesh_rgbd_rendering ${PROJECT_NAME})






#add_executable(open3d_test src/open3d_test.cpp)
#target_link_libraries(open3d_test ${OpenCV_LIBS} ${Open3D_LIBRARIES})



#add_executable(viewer src/viewer.cc src/trackball.cc)
#target_link_libraries(viewer ${PROJECT_NAME})



