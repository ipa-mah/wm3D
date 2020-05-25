#include <cuda/common/utils_cuda.hpp>
#include <cuda/integration/tsdf_volume_cuda.hpp>
namespace cuda
{
__device__ inline bool TSDFVolumeCudaDevice::inVolume(const Eigen::Vector3i& x)
{
	return 0 <= x(0) && x(0) < (dims_(0) - 1) && 0 <= x(1) && x(1) < (dims_(1) - 1) && 0 <= x(2) && x(2) < (dims_(2) - 1);
}

__device__ inline bool TSDFVolumeCudaDevice::inVolumef(const Eigen::Vector3f& x)
{
	return 0 <= x(0) && x(0) < (dims_(0) - 1) && 0 <= x(1) && x(1) < (dims_(1) - 1) && 0 <= x(2) && x(2) < (dims_(2) - 1);
}

__device__ inline Eigen::Vector3f TSDFVolumeCudaDevice::worldToVoxelf(const Eigen::Vector3f& x_w)
{
	Eigen::Vector4f x4_w = Eigen::Vector4f(x_w(0), x_w(1), x_w(2), 1);
	// transform to volume
	Eigen::Vector3f x_v = (world_to_volume_.template cast<float>() * x4_w).head<3>();
	return volumeToVoxelf(x_v);
}

__device__ inline Eigen::Vector3f TSDFVolumeCudaDevice::voxelfToWorld(const Eigen::Vector3f& x)
{
	Eigen::Vector3f v = voxelfToVolume(x);
	//volume_to_world_.setIdentity();
	return (volume_to_world_.template cast<float>() * Eigen::Vector4f(v(0), v(1), v(2), 1)).head<3>();
}

__device__ inline Eigen::Vector3f TSDFVolumeCudaDevice::voxelfToVolume(const Eigen::Vector3f& x)
{
	return Eigen::Vector3f((x(0) + 0.5f) * voxel_length_, (x(1) + 0.5f) * voxel_length_, (x(2) + 0.5f) * voxel_length_);
}

__device__ inline Eigen::Vector3f TSDFVolumeCudaDevice::volumeToVoxelf(const Eigen::Vector3f& x_v)
{
	return Eigen::Vector3f(x_v(0) * inv_voxel_length_ - 0.5f, x_v(1) * inv_voxel_length_ - 0.5f, x_v(2) * inv_voxel_length_ - 0.5f);
}

/** trilinear interpolations. **/
/** Ensure it is called within [0, N - 1)^3 **/
/*
__device__ float TSDFVolumeCudaDevice::tsdfAt(const Eigen::Vector3f& x)
{
	Eigen::Vector3i xi = x.template cast<int>();
	Eigen::Vector3f r = x - xi.template cast<float>();

	return (1 - r(0)) * ((1 - r(1)) * ((1 - r(2)) * tsdf_[indexOf(xi + Eigen::Vector3i(0, 0, 0))] + r(2) * tsdf_[indexOf(xi + Eigen::Vector3i(0, 0, 1))]) +
						 r(1) * ((1 - r(2)) * tsdf_[indexOf(xi + Eigen::Vector3i(0, 1, 0))] + r(2) * tsdf_[indexOf(xi + Eigen::Vector3i(0, 1, 1))])) +
		   r(0) * ((1 - r(1)) * ((1 - r(2)) * tsdf_[indexOf(xi + Eigen::Vector3i(1, 0, 0))] + r(2) * tsdf_[indexOf(xi + Eigen::Vector3i(1, 0, 1))]) +
				   r(1) * ((1 - r(2)) * tsdf_[indexOf(xi + Eigen::Vector3i(1, 1, 0))] + r(2) * tsdf_[indexOf(xi + Eigen::Vector3i(1, 1, 1))]));
}

__device__ uchar TSDFVolumeCudaDevice::weightAt(const Eigen::Vector3f& x)
{
	Eigen::Vector3i xi = x.template cast<int>();
	Eigen::Vector3f r = x - xi.template cast<float>();

	return uchar((1 - r(0)) * ((1 - r(1)) * ((1 - r(2)) * weight_[indexOf(xi + Eigen::Vector3i(0, 0, 0))] + r(2) * weight_[indexOf(xi + Eigen::Vector3i(0, 0, 1))]) +
							   r(1) * ((1 - r(2)) * weight_[indexOf(xi + Eigen::Vector3i(0, 1, 0))] + r(2) * weight_[indexOf(xi + Eigen::Vector3i(0, 1, 1))])) +
				 r(0) * ((1 - r(1)) * ((1 - r(2)) * weight_[indexOf(xi + Eigen::Vector3i(1, 0, 0))] + r(2) * weight_[indexOf(xi + Eigen::Vector3i(1, 0, 1))]) +
						 r(1) * ((1 - r(2)) * weight_[indexOf(xi + Eigen::Vector3i(1, 1, 0))] + r(2) * weight_[indexOf(xi + Eigen::Vector3i(1, 1, 1))])));
}

__device__ uchar4 TSDFVolumeCudaDevice::colorAt(const Eigen::Vector3f& x)
{
	Eigen::Vector3i xi = x.template cast<int>();
	Eigen::Vector3f r = x - xi.template cast<float>();

	uchar4 colorf =
		(1 - r(0)) * ((1 - r(1)) * ((1 - r(2)) * color_[indexOf(xi + Eigen::Vector3i(0, 0, 0))].template cast<float>() + r(2) * color_[indexOf(xi + Eigen::Vector3i(0, 0, 1))].template cast<float>()) +
					  r(1) * ((1 - r(2)) * color_[indexOf(xi + Eigen::Vector3i(0, 1, 0))].template cast<float>() + r(2) * color_[indexOf(xi + Eigen::Vector3i(0, 1, 1))].template cast<float>())) +
		r(0) * ((1 - r(1)) * ((1 - r(2)) * color_[indexOf(xi + Eigen::Vector3i(1, 0, 0))].template cast<float>() + r(2) * color_[indexOf(xi + Eigen::Vector3i(1, 0, 1))].template cast<float>()) +
				r(1) * ((1 - r(2)) * color_[indexOf(xi + Eigen::Vector3i(1, 1, 0))].template cast<float>() + r(2) * color_[indexOf(xi + Eigen::Vector3i(1, 1, 1))].template cast<float>()));
	return colorf;
}
*/
__device__ void TSDFVolumeCudaDevice::integrate(const Eigen::Vector3i& x, const PtrStepSz<uchar3>& color_image, const PtrStepSz<ushort>& depth_image, CameraIntrinsicCuda& intrins,
												const Eigen::Matrix4f& world_to_cam, float depth_scale)
{
	// Transform voxel from volume to world
	Eigen::Vector3f x_w = voxelfToWorld(x.template cast<float>());
	// transform voxel from world to camera
	Eigen::Vector3f x_c = (world_to_cam * Eigen::Vector4f(x_w(0), x_w(1), x_w(1), 1)).head<3>();
	int2 pixel = make_int2(__float2int_rn(intrins.fx_ * x_c(0) / x_c(2) + intrins.cx_), 
						__float2int_rn(intrins.fy_ * x_c(1) / x_c(2) + intrins.cy_));	
	if (pixel.x < 0 || pixel.x >= intrins.width_ || pixel.y < 0 || pixel.y >= intrins.height_)
		return;
	float d = depth_image.ptr(pixel.y)[pixel.x] * depth_scale;
	if (d <= 0.0001 || d > 5.0) return;
	float tsdf = d - x_c(2);
	if (tsdf <= -sdf_trunc_) return;
	tsdf = fminf(tsdf / sdf_trunc_, 1.0f);
	uchar3 color = color_image.ptr(pixel.y)[pixel.x];

	float& tsdf_sum = this->tsdf(x);
	uchar& weight_sum = this->weight(x);
	uchar3& color_sum = this->color(x);

	float w0 = 1 / (weight_sum + 1.0f);
	float w1 = 1 - w0;

	tsdf_sum = tsdf * w0 + tsdf_sum * w1;

	color_sum.x = color.x * w0 + color_sum.x * w1;
	color_sum.y = color.y * w0 + color_sum.y * w1; 
	color_sum.z = color.z * w0 + color_sum.z * w1;
	weight_sum = uchar(fminf(weight_sum + 1.0f, 255));
}

///////////////////////////////////////

__GLOBAL__ void resetTSDFVolumeCudaKernel(TSDFVolumeCudaDevice server)
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	const int z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x >= server.dims_(0) || y >= server.dims_(1) || z >= server.dims_(2)) return;
	Eigen::Vector3i voxel = Eigen::Vector3i(x, y, z);
	server.tsdf(voxel) = 1.0f;
}

__host__ void TSDFVolumeCudaKernel::reset(TSDFVolumeCuda& volume)
{
	const int num_blocks_x = DIV_CEILING(volume.dims_(0), THREAD_3D_UNIT);
	const int num_blocks_y = DIV_CEILING(volume.dims_(1), THREAD_3D_UNIT);
	const int num_blocks_z = DIV_CEILING(volume.dims_(2), THREAD_3D_UNIT);

	const dim3 blocks(num_blocks_x, num_blocks_y, num_blocks_z);
	const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);
	resetTSDFVolumeCudaKernel<<<blocks, threads>>>(*volume.device_);
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
}

__global__ void integrateKernel(TSDFVolumeCudaDevice server,
							 PtrStepSz<uchar3> color_image,
							 PtrStepSz<ushort> depth_image,
							  CameraIntrinsicCuda intrins,
							   Eigen::Matrix4f cam_to_world,
							    float depth_scale)
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	const int z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x >= server.dims_(0) || y >= server.dims_(1) || z >= server.dims_(2)) return;
	Eigen::Vector3i voxel_id = Eigen::Vector3i(x, y, z);
	server.integrate(voxel_id, color_image, depth_image, intrins, cam_to_world, depth_scale);
}

// caller from host
__host__ void TSDFVolumeCudaKernel::integrate(TSDFVolumeCuda& volume,
											const DeviceArray2D<uchar3>& color_image,
											const DeviceArray2D<ushort>& depth_image,
											const CameraIntrinsicCuda& intrin,
											const Eigen::Matrix4f& world_to_cam,
											const float depth_scale)
{
	
	
	const int num_blocks_x = DIV_CEILING(volume.dims_(0), THREAD_3D_UNIT);
	const int num_blocks_y = DIV_CEILING(volume.dims_(1), THREAD_3D_UNIT);
	const int num_blocks_z = DIV_CEILING(volume.dims_(2), THREAD_3D_UNIT);
	const dim3 blocks(num_blocks_x, num_blocks_y, num_blocks_z);
	const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);
	integrateKernel<<<blocks, threads>>>(*volume.device_, color_image, depth_image, intrin, world_to_cam, depth_scale);
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
}

}  // namespace cuda