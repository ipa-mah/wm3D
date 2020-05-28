#pragma once

#include <cuda_runtime.h>
#include <cuda/camera/camera_intrinsic_cuda.hpp>
#include <cuda/common/common.hpp>
#include <cuda/common/utils_cuda.hpp>
#include <cuda/container/device_array.hpp>
#include <cuda/cuda_headers.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

namespace cuda
{
void initializeVolume(DeviceArray2D<float>& tsdf_volume, DeviceArray2D<float>& weight_volume, const Eigen::Vector3i& dims);
void integrateTsdfVolume(const DeviceArray2D<unsigned short>& depth_map, DeviceArray2D<float>& tsdf_volume, DeviceArray2D<float>& weight_volume, const Eigen::Vector3i& dims, const float voxel_length,
						 const float truncated_distance, const CameraIntrinsicCuda& cam_params, const Eigen::Matrix4f world_to_cam, const float depth_scale);

}  // namespace cuda