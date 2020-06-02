#pragma once
#include <cuda_runtime.h>
#include <cstdlib>
#include <cuda/camera/camera_intrinsic_cuda.hpp>
#include <cuda/common/common.hpp>
#include <cuda/common/utils_cuda.hpp>
#include <cuda/container/device_array.hpp>

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <tuple>
namespace cuda
{
/// TSDF host class
class TSDFVolumeCuda
{
  public:
	DeviceArray2D<float> weight_volume_;
	DeviceArray2D<float> tsdf_volume_;
	// DeviceArray2D<uchar3*> color_;

  public:
	Eigen::Vector3i dims_;
	float voxel_length_;
	float inv_voxel_length_;
	float sdf_trunc_;
	Eigen::Matrix4d volume_to_world_;
	Eigen::Matrix4d world_to_volume_;

  public:
	using Ptr = std::shared_ptr<TSDFVolumeCuda>;
	using ConstPtr = std::shared_ptr<const TSDFVolumeCuda>;
	TSDFVolumeCuda();
	TSDFVolumeCuda(Eigen::Vector3i dims, float voxel_length, float sdf_trunc);
	TSDFVolumeCuda(const TSDFVolumeCuda& other);
	TSDFVolumeCuda& operator=(const TSDFVolumeCuda& other);
	~TSDFVolumeCuda();

	void create(const Eigen::Vector3i& dims, const float voxel_length, const float sdf_trunc);
	void release();

	std::tuple<std::vector<float>, std::vector<float>, std::vector<Eigen::Vector3i>> downloadVolume();

  public:
	void initializeVolume();
	void integrateTsdfVolume(const DeviceArray2D<unsigned short>& depth_map, const CameraIntrinsicCuda& cam_params, const Eigen::Matrix4f& world_to_cam, const float depth_scale);
};

}  // namespace cuda
