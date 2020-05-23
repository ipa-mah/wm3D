#pragma once
#include <cuda/common/common.hpp>
#include <cuda/container/device_array.hpp>

#include <cstdlib>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <memory>
namespace cuda
{
class TSDFVolumeCudaDevice
{
  public:
	int N_;
	u_char* weight_;
	float* tsdf_;
	Eigen::Vector3i* color_;

  public:
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
		ret(0) = int(index % N_);
		ret(1) = int((index % (N_ * N_)) / N_);
		ret(2) = int(index / (N_ * N_));
		return ret;
	}
	__DEVICE__ inline int indexOf(const Eigen::Vector3i& v)
	{
		return int(v(2) * (N_ * N_) + v(1) * N_ + v(0));
	}

  public:
	__DEVICE__ inline float& tsdf(const Eigen::Vector3i& v)
	{
		return tsdf_[indexOf(v)];
	}
	__DEVICE__ inline uchar& weight(const Eigen::Vector3i& v)
	{
		return weight_[indexOf(v)];
	}
	__DEVICE__ inline Eigen::Vector3i& color(const Eigen::Vector3i& v)
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
	__DEVICE__ Eigen::Vector3f colorAt(const Eigen::Vector3f& x);
	__DEVICE__ Eigen::Vector3f gradientAt(const Eigen::Vector3f& x);
	
  public:
	__DEVICE__ void integrate(const Eigen::Vector3i& x,
							 const PtrStepSz<uchar3>& color,
							 const PtrStepSz<ushort>& depth,
							 const Eigen::Matrix3f& intrins,
							 const Eigen::Matrix4f& cam_to_world,
							 int image_width,
						  	 int image_height,
							 float depth_scale);
	__DEVICE__ void rayCasting(const Eigen::Vector2i& p, const Eigen::Matrix3d& intrins,
						 const Eigen::Matrix4f& cam_to_world);

  public:
	friend class TSDFVolumeCuda;
};

class TSDFVolumeCuda
{
  public:
	std::shared_ptr<TSDFVolumeCudaDevice> device_ = nullptr;

  public:
	int N_;

	float voxel_length_;
	float sdf_trunc_;

  public:
	using Ptr = std::shared_ptr<TSDFVolumeCuda>;
	using ConstPtr = std::shared_ptr<const TSDFVolumeCuda>;
	TSDFVolumeCuda();
	TSDFVolumeCuda(int N, float voxel_length, float sdf_trunc);
	TSDFVolumeCuda(const TSDFVolumeCuda& other);
	TSDFVolumeCuda& operator=(const TSDFVolumeCuda& other);
	~TSDFVolumeCuda();

	void create(int N);
	void release();
	void updateDevice();
	void reset();
	void uploadVolume(std::vector<float>& tsdf, std::vector<uchar>& weight, std::vector<Eigen::Vector3i>& color);
	std::tuple<std::vector<float>, std::vector<uchar>, std::vector<Eigen::Vector3i>> downloadVolume();
};

class TSDFVolumeCudaKernel
{
  public:
	static void reset(TSDFVolumeCuda &volume);
	static void integrate(TSDFVolumeCuda &volume,
						  DeviceArray2D<ushort> &depth,
						  DeviceArray2D<uchar3> &color,
						  Eigen::Matrix3d& intrin,
						  Eigen::Matrix4d& cam_to_world,
						  int image_width,
						  int image_height);
	static void rayCasting(TSDFVolumeCuda &volume,
							DeviceArray2D<float3> &image,
							Eigen::Matrix3d& intrins,
							Eigen::Matrix4d& cam_to_world,
							int image_width,
							int image_height);
};

__GLOBAL__ void resetTSDFVolumeCudaKernel(TSDFVolumeCudaDevice server);
__GLOBAL__ void integrateKernel(TSDFVolumeCudaDevice server,
								PtrStepSz<ushort> depth,
								Eigen::Matrix3d intrins,
								Eigen::Matrix4d cam_to_world);
__GLOBAL__ void rayCastingKernel(TSDFVolumeCudaDevice server,
								PtrStepSz<float3> image,
								Eigen::Matrix3d intrins,
								Eigen::Matrix4d cam_to_world);

}  // namespace cuda
