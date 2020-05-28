#pragma once
#include <cuda/camera/camera_intrinsic_cuda.hpp>
#include <cuda/common/common.hpp>
#include <cuda/common/utils_cuda.hpp>
#include <cuda/container/device_array.hpp>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <memory>
namespace cuda
{
void createVMap(const DeviceArray2D<unsigned short>& depth_map, DeviceArray2D<float3>& vertex_map, const CameraIntrinsicCuda& cam_params, const float depth_cutoff, const float depth_scale);

void createNMap(const DeviceArray2D<float3>& vmap, DeviceArray2D<float3>& nmap);

void createRenderMap(const DeviceArray2D<float3>& normals, DeviceArray2D<uchar3>& render_image);
void createGaussianFilter(const DeviceArray2D<float>& src, DeviceArray2D<float>& dst);
void computeFaceNormal(const DeviceArray<float3>& vertices);
void createDepthBoundaryMask(const DeviceArray2D<unsigned short>& src, DeviceArray2D<short>& dx, DeviceArray2D<short>& dy, DeviceArray2D<uchar>& mask_map, const float depth_threshold);
void hostMaskImage(const DeviceArray2D<float>& sobel_dx, const DeviceArray2D<float>& sobel_dy, DeviceArray2D<uchar>& mask_map, const float depth_threshold);

}  // namespace cuda