cmake_minimum_required (VERSION 3.1)
project(wm3D VERSION 0.1.0 LANGUAGES CXX)

# Options
option(BUILD_CUDA_MODULE         "Build the CUDA module"                    ON)
option(WITH_OPENMP               "Use OpenMP multi-threading"               ON)


# Cuda module
# Cache variables for specifying the GPU architectures
set(CUDA_ARCH "Auto" CACHE STRING "Selects GPU architectures for code generation, \
one of (Auto|BasicPTX|User). Set to 'User' to set a custom list of architectures" )
set_property(CACHE CUDA_ARCH PROPERTY STRINGS Auto BasicPTX User)
set(CUDA_ARCH_USER "" CACHE STRING "User defined list of GPU architectures, e.g. 3.5 5.0+PTX Turing" )

# Build CUDA module by default if CUDA is available
# Compatible with CMake 3.8+
if (BUILD_CUDA_MODULE)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        find_package(CUDA REQUIRED) # required for cuda_select_nvcc_arch_flags
        message(STATUS "Building CUDA enabled")
        enable_language(CUDA)
        # get gencode flags
        if("${CUDA_ARCH}" STREQUAL "User")
            cuda_select_nvcc_arch_flags(CUDA_GENCODES "${CUDA_ARCH_USER}")
        elseif("${CUDA_ARCH}" STREQUAL "BasicPTX")
            # include oldest and most recent PTX and rely on JIT compilation
            set(CUDA_GENCODES "-gencode arch=compute_30,code=compute_30;-gencode arch=compute_75,code=compute_75")
        else()
            cuda_select_nvcc_arch_flags(CUDA_GENCODES "${CUDA_ARCH}")
        endif()
    else()
        set(BUILD_CUDA_MODULE OFF)
        message(STATUS "No CUDA support")
    endif()
    message("ARCH_FLAGS = ${CUDA_ARCH}")
endif ()

macro(ADD_SOURCE_GROUP MODULE_NAME)
    file(GLOB MODULE_HEADER_FILES "${MODULE_NAME}/*.hpp")
    source_group("Header Files\\${MODULE_NAME}" FILES ${MODULE_HEADER_FILES})
    file(GLOB MODULE_SOURCE_FILES "${MODULE_NAME}/*.cpp")
    source_group("Source Files\\${MODULE_NAME}" FILES ${MODULE_SOURCE_FILES})
endmacro(ADD_SOURCE_GROUP)


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
#add_subdirectory(elibs)
#include_directories(${CMAKE_SOURCE_DIR}/elibs)


##########

include_directories(
    include
    cuda/include
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
    
    
################



#Cuda directory
add_subdirectory(cuda)
#target_link_libraries(wm3D_CUDA_MODULE ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES})

#file(GLOB sources
#    "${CMAKE_SOURCE_DIR}/src/*.cpp"
#    "${CMAKE_SOURCE_DIR}/src/utility/*.cpp"
#    "${CMAKE_SOURCE_DIR}/src/visualization/*.cpp"
#    )

# GLOB all required files in this subdirectory
file(GLOB_RECURSE CPP_ALL_HEADER_FILES "${CMAKE_SOURCE_DIR}/include/*.hpp")
file(GLOB_RECURSE CPP_ALL_SOURCE_FILES "${CMAKE_SOURCE_DIR}/src/*.cpp") 
add_library(${PROJECT_NAME} 
             ${CPP_ALL_HEADER_FILES}
             ${CPP_ALL_SOURCE_FILES}
            )
target_link_libraries(
    ${PROJECT_NAME}
    ${Boost_LIBRARIES}
    ${OpenCV_LIBS}
    ${Open3D_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${PCL_LIBRARIES}
    ${GLFW_LIBRARIES}
    ${GLUT_LIBRARY}
    ${GLEW_LIBRARIES}
    #tinyobjloader
    ${realsense2_LIBRARIES}
    )

add_executable(mesh_rgbd_rendering test/mesh_rgbd_rendering.cpp)
target_link_libraries(mesh_rgbd_rendering ${PROJECT_NAME})


add_executable(cuda_test test/cuda_test.cpp)
target_link_libraries(cuda_test wm3D_CUDA_MODULE ${PROJECT_NAME})
#
add_executable(function_test test/function_test.cpp)
target_link_libraries(function_test wm3D_CUDA_MODULE ${PROJECT_NAME} )

add_executable(triangle_mesh_cuda_test test/triangle_mesh_cuda_test.cpp)
target_link_libraries(triangle_mesh_cuda_test wm3D_CUDA_MODULE ${PROJECT_NAME} )

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

