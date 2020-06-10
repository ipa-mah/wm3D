#pragma once
#include <cuda_runtime.h>
#include <cuda/camera/camera_intrinsic_cuda.hpp>
#include <cuda/common/common.hpp>
#include <cuda/common/utils_cuda.hpp>
#include <cuda/container/device_array.hpp>
#include <cuda/cuda_headers.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace cuda
{
void initializeVolume(DeviceArray2D<float>& tsdf_volume, DeviceArray2D<float>& weight_volume, const Eigen::Vector3i& dims);
void initializeVolume(cv::cuda::GpuMat& tsdf_volume, cv::cuda::GpuMat& weight_volume, const Eigen::Vector3i& dims);
void integrateTsdfVolume(const DeviceArray2D<unsigned short>& depth_map, DeviceArray2D<float>& tsdf_volume, DeviceArray2D<float>& weight_volume, const Eigen::Vector3i& dims, const float voxel_length,
						 const float truncated_distance, const CameraIntrinsicCuda& cam_params, const Eigen::Matrix4f& world_to_cam, const float depth_scale);

void integrateTsdfVolume(const cv::cuda::GpuMat& depth_map, cv::cuda::GpuMat& tsdf_volume, cv::cuda::GpuMat& weight_volume, const Eigen::Vector3i& dims, const float voxel_length,
						 const float truncated_distance, const CameraIntrinsicCuda& cam_params, const Eigen::Matrix4f world_to_cam, const float depth_scale);

void testTriangleMeshCuda(DeviceArray<Eigen::Vector3f>& vertices);
void createRenderMap(const DeviceArray2D<float3>& normals, DeviceArray2D<uchar3>& render_image);
void allocateVertexHost(const DeviceArray2D<float>& tsdf_volume, const DeviceArray2D<float>& weight_volume, DeviceArray2D<Eigen::Vector3i>& vertex_indices, DeviceArray2D<int>& table_indices,
						const Eigen::Vector3i& dims);
void hostAddDevice(int* vector_add, const int n, const int a);
}  // namespace cuda
