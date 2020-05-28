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
#include <tuple>
namespace cuda
{
/// TSDF device class
class TSDFVolumeCudaDevice
{
  public:
	float* weight_;
	float* tsdf_;
	uchar3* color_;

  public:
	Eigen::Vector3i dims_;
	float voxel_length_;
	float inv_voxel_length_;
	float sdf_trunc_;
	Eigen::Matrix4d volume_to_world_;
	Eigen::Matrix4d world_to_volume_;

  public:
	__DEVICE__ inline Eigen::Vector3i vectorize(std::size_t index)
	{
		// #ifdef CUDA_DEBUG_ENABLE_ASSERTION
		//     assert(index < N * N * N);
		// #endif
		Eigen::Vector3i ret;
		ret(0) = int(index % dims_(0));
		ret(1) = int((index % (dims_(0) * dims_(1))) / dims_(2));
		ret(2) = int(index / (dims_(0) * dims_(1)));
		return ret;
	}
	__DEVICE__ inline int indexOf(int x, int y, int z) const
	{
		return x * dims_(1) * dims_(2) + y * dims_(2) + z;
	}
	__DEVICE__ inline int indexOf(const Eigen::Vector3i& xyz)
	{
		return indexOf(xyz(0), xyz(1), xyz(2));
	}

  public:
	__DEVICE__ inline float& tsdf(const Eigen::Vector3i& v)
	{
		return tsdf_[indexOf(v)];
	}
	__DEVICE__ inline float& weight(const Eigen::Vector3i& v)
	{
		return weight_[indexOf(v)];
	}
	__DEVICE__ inline uchar3& color(const Eigen::Vector3i& v)
	{
		return color_[indexOf(v)];
	}
	/** Voxel level gradient -- NO trilinear interpolation.
	 * This is especially useful for MarchingCubes **/
	__DEVICE__ Eigen::Vector3f gradient(const Eigen::Vector3i& x);

	/** Coordinate conversions **/
	__DEVICE__ inline bool inVolume(const Eigen::Vector3i& x);
	__DEVICE__ inline bool inVolumef(const Eigen::Vector3f& x);

	__DEVICE__ inline Eigen::Vector3f worldToVoxelf(const Eigen::Vector3f& x_w);
	__DEVICE__ inline Eigen::Vector3f voxelfToWorld(const Eigen::Vector3f& x);
	__DEVICE__ inline Eigen::Vector3f volumeToVoxelf(const Eigen::Vector3f& x_v);
	__DEVICE__ inline Eigen::Vector3f voxelfToVolume(const Eigen::Vector3f& x);

  public:
	/** Value interpolating **/
	__DEVICE__ float tsdfAt(const Eigen::Vector3f& x);
	__DEVICE__ uchar weightAt(const Eigen::Vector3f& x);
	__DEVICE__ uchar3 colorAt(const Eigen::Vector3f& x);
	__DEVICE__ Eigen::Vector3f gradientAt(const Eigen::Vector3f& x);

  public:
	__DEVICE__ void integrate(const Eigen::Vector3i& x, const PtrStepSz<uchar3>& color, const PtrStepSz<ushort>& depth, CameraIntrinsicCuda& intrins, const Eigen::Matrix4f& world_to_cam,
							  float depth_scale);
	__DEVICE__ void rayCasting(const Eigen::Vector2i& p, const Eigen::Matrix3d& intrins, const Eigen::Matrix4f& world_to_cam);

  public:
	friend class TSDFVolumeCuda;
};

/// TSDF host class
class TSDFVolumeCuda
{
  public:
	std::shared_ptr<TSDFVolumeCudaDevice> device_ = nullptr;

  public:
	Eigen::Vector3i dims_;

	float voxel_length_;
	float sdf_trunc_;

  public:
	using Ptr = std::shared_ptr<TSDFVolumeCuda>;
	using ConstPtr = std::shared_ptr<const TSDFVolumeCuda>;
	TSDFVolumeCuda();
	TSDFVolumeCuda(Eigen::Vector3i dims, float voxel_length, float sdf_trunc);
	TSDFVolumeCuda(const TSDFVolumeCuda& other);
	TSDFVolumeCuda& operator=(const TSDFVolumeCuda& other);
	~TSDFVolumeCuda();

	void create(const Eigen::Vector3i& dims);
	void release();
	void updateDevice();
	void reset();
	void uploadVolume(std::vector<float>& tsdf, std::vector<float>& weight, std::vector<uchar3>& color);
	std::tuple<std::vector<float>, std::vector<float>, std::vector<uchar3>> downloadVolume();

  public:
	void integrate(const DeviceArray2D<uchar3>& color_image, const DeviceArray2D<ushort>& depth_image, const CameraIntrinsicCuda& intrins, const Eigen::Matrix4f& world_to_cam,
				   const float depth_scale);
};

/// kernel class
class TSDFVolumeCudaKernel
{
  public:
	static void reset(TSDFVolumeCuda& volume);
	static void integrate(TSDFVolumeCuda& volume, const DeviceArray2D<uchar3>& color_image, const DeviceArray2D<ushort>& depth_image, const CameraIntrinsicCuda& intrin,
						  const Eigen::Matrix4f& world_to_cam, const float depth_scale);
	void integrateTsdfVolume(const DeviceArray2D<unsigned short>& depth_map, DeviceArray2D<float>& tsdf_volume, DeviceArray2D<float>& weight_volume, const int volume_res, const float voxel_size,
							 const float truncated_distance, const CameraIntrinsicCuda& cam_params, const Eigen::Matrix4f world_to_cam, const float depth_scale);

	static void rayCasting(TSDFVolumeCuda& volume, DeviceArray2D<float3>& image, CameraIntrinsicCuda& intrins, Eigen::Matrix4f& world_to_cam);

	void initializeVolume(DeviceArray2D<float>& tsdf_volume, DeviceArray2D<float>& weight_volume, const int volume_size);
};

__GLOBAL__ void resetTSDFVolumeCudaKernel(TSDFVolumeCudaDevice server);
__GLOBAL__ void integrateKernel(TSDFVolumeCudaDevice server, PtrStepSz<uchar3> color_image, PtrStepSz<ushort> depth_image, CameraIntrinsicCuda intrins, Eigen::Matrix4f world_to_cam,
								float depth_scale);
__GLOBAL__ void rayCastingKernel(TSDFVolumeCudaDevice server, PtrStepSz<float3> image, CameraIntrinsicCuda intrins, Eigen::Matrix4f world_to_cam);

}  // namespace cuda
