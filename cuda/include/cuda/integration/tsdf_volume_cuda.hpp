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
class TSDFVolumeDevice 
{
public:
	PtrStepSz<float> tsdf_volume_;
	PtrStepSz<float> weight_volume_;

	int res_;
	float voxel_length_;
	float inv_voxel_lenth_;
	float sdf_trunc_;
	Eigen::Matrix4f volume_to_world_;
	Eigen::Matrix4f world_to_volume_;

public:
	__DEVICE__ inline float &tsdf_val(const Eigen::Vector3i& idx)
	{
		return tsdf_volume_.ptr(idx(2) * res_ + idx(1))[idx(0)];
	}
	__DEVICE__ inline float &weight_val(const Eigen::Vector3i& idx)
	{
		return weight_volume_.ptr(idx(2) * res_ + idx(1))[idx(0)];
	}

    /** Voxel level gradient -- NO trilinear interpolation.
     * This is especially useful for MarchingCubes **/
    __DEVICE__ Eigen::Vector3f gradient(const Eigen::Vector3i &idx);
public:
	__DEVICE__ void integrate(const Eigen::Vector3i& voxel_idx,const PtrStepSz<unsigned short>& depth_image, 
							 	const CameraIntrinsicCuda& cam_params, const Eigen::Matrix4f& world_to_cam, 
								const float depth_scale);
	__DEVICE__ Eigen::Vector3f rayCasting(const Eigen::Vector2i& p,const CameraIntrinsicCuda& cam_params,
					const Eigen::Matrix4f& cam_to_world,const float ray_step);						

};

// class TestTSDFVolume
// {

// };


/// TSDF host class
class TSDFVolumeCuda
{
  public:
	DeviceArray2D<float> weight_volume_;
	DeviceArray2D<float> tsdf_volume_;
	// DeviceArray2D<uchar3*> color_;

  public:
	int res_;
	float voxel_length_;
	float inv_voxel_length_;
	float sdf_trunc_;
	Eigen::Matrix4d volume_to_world_;
	Eigen::Matrix4d world_to_volume_;

  public:
	using Ptr = std::shared_ptr<TSDFVolumeCuda>;
	using ConstPtr = std::shared_ptr<const TSDFVolumeCuda>;
	TSDFVolumeCuda();
	TSDFVolumeCuda(int res, float voxel_length, float sdf_trunc);
	TSDFVolumeCuda(const TSDFVolumeCuda& other);
	TSDFVolumeCuda& operator=(const TSDFVolumeCuda& other);
	~TSDFVolumeCuda();

	void create(const int res, const float voxel_length, const float sdf_trunc);
	void release();

	std::tuple<std::vector<float>, std::vector<float>, std::vector<Eigen::Vector3i>> downloadVolume();

  public:
	void initializeVolume();
	void integrateTsdfVolume(const DeviceArray2D<unsigned short>& depth_map, 
					const CameraIntrinsicCuda& cam_params, const Eigen::Matrix4f& world_to_cam, 
				const float depth_scale);
	void rayCasting(DeviceArray2D<float3>& model_vertex,DeviceArray2D<float3>& model_normal, 
					const CameraIntrinsicCuda& cam_params,
					const Eigen::Matrix4f& cam_to_world,const float ray_step);
};

}  // namespace cuda
