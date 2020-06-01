# Modified to add cuda files
macro(add_cuda_source_group module_name)
    file(GLOB MODULE_HEADER_FILES "${module_name}/*.cuh")
    source_group("Header Files\\${module_name}" FILES ${MODULE_HEADER_FILES})
    file(GLOB MODULE_SOURCE_FILES "${module_name}/*.cu")
    source_group("Source Files\\${module_name}" FILES ${MODULE_SOURCE_FILES})
endmacro()

#find_package(CUDAToolkit REQUIRED)
# GLOB all required files in this subdirectory
file(GLOB_RECURSE CUDA_ALL_HEADER_FILES "*.hpp")
file(GLOB_RECURSE CUDA_ALL_SOURCE_FILES "*.cpp")
file(GLOB_RECURSE CUDA_ALL_CUDA_HEADER_FILES "*.cuh")
file(GLOB_RECURSE CUDA_ALL_CUDA_SOURCE_FILES "*.cu")
#set(wm3D_CUDA_MODULE "wm3DCuda" CACHE STRING "CUDA component library")
# No need to cuda_add_library anymore
add_library(wm3D_CUDA_MODULE
        ${CUDA_ALL_HEADER_FILES}
        ${CUDA_ALL_SOURCE_FILES}
        ${CUDA_ALL_CUDA_HEADER_FILES}
        ${CUDA_ALL_CUDA_SOURCE_FILES}
        )

target_link_libraries(wm3D_CUDA_MODULE ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES})



