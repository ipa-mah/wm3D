#include <cuda/common/utils_cuda.hpp>
#include <cuda/integration/tsdf_volume_cuda.hpp>
namespace cuda
{
__global__ void initializeVolumeKernel(PtrStepSz<float> tsdf_volume, PtrStepSz<float> weight_volume, const Eigen::Vector3i dims)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;
	if (x >= dims(0) || y >= dims(1) || z >= dims(2)) return;

	tsdf_volume.ptr(z * dims(1) + y)[x] = 0.0;
	weight_volume.ptr(z * dims(1) + y)[x] = 0.0;
}
void TSDFVolumeCuda::initializeVolume()
{
	const int num_blocks_x = DIV_CEILING(dims_(0), THREAD_3D_UNIT);
	const int num_blocks_y = DIV_CEILING(dims_(1), THREAD_3D_UNIT);
	const int num_blocks_z = DIV_CEILING(dims_(2), THREAD_3D_UNIT);
	const dim3 blocks(num_blocks_x, num_blocks_y, num_blocks_z);
	const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);

	initializeVolumeKernel<<<blocks, threads>>>(tsdf_volume_, weight_volume_, dims_);
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
}

__global__ void integrateKernel(const PtrStepSz<unsigned short> depth_image, PtrStepSz<float> tsdf_volume, PtrStepSz<float> weight_volume, const Eigen::Vector3i dims, float voxel_length,
								const float depth_scale, const CameraIntrinsicCuda cam_params, const float truncation_distance, const Eigen::Matrix4f world_to_cam)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= dims(0) || y >= dims(1) || z >= dims(2)) return;

	// Convert to voxel grid to global coordinate

	const float3 global_voxel = make_float3((static_cast<float>(x) + 0.5f) * voxel_length, (static_cast<float>(y) + 0.5f) * voxel_length, (static_cast<float>(z) + 0.5f) * voxel_length);
	// convert voxel from global to local camera coordinate
	const Eigen::Vector3f camera_voxel = (world_to_cam * Eigen::Vector4f(global_voxel.x, global_voxel.y, global_voxel.z, 1.0)).head<3>();

	if (camera_voxel(2) <= 0) return;
	// projection
	const int2 uv = make_int2(__float2int_rn(camera_voxel(0) / camera_voxel(2) * cam_params.fx_ + cam_params.cx_), __float2int_rn(camera_voxel(1) / camera_voxel(2) * cam_params.fy_ + cam_params.cy_));
	if (uv.x < 0 || uv.x >= depth_image.cols || uv.y < 0 || uv.y >= depth_image.rows) return;

	const float depth = depth_image.ptr(uv.y)[uv.x] * depth_scale;

	if (depth <= 0.0001 || depth > 5.0) return;
	const float sdf = (depth - camera_voxel(2));
	if (sdf >= -truncation_distance)
	{
		const float new_tsdf = fmin(1.f, sdf / truncation_distance);

		const float current_tsdf = tsdf_volume.ptr(z * dims(1) + y)[x];
		const short current_weight = weight_volume.ptr(z * dims(1) + y)[x];

		const float add_weight = 1;
		const float updated_tsdf = (current_weight * current_tsdf + add_weight * new_tsdf) / (current_weight + add_weight);

		const float new_weight = current_weight + add_weight;
		// const float new_weight = min(current_weight + add_weight, 128.0f);

		tsdf_volume.ptr(z * dims(1) + y)[x] = updated_tsdf;
		weight_volume.ptr(z * dims(1) + y)[x] = new_weight;
	}
}

void TSDFVolumeCuda::integrateTsdfVolume(const DeviceArray2D<unsigned short>& depth_map, const CameraIntrinsicCuda& cam_params, const Eigen::Matrix4f& world_to_cam, const float depth_scale)
{
	const int num_blocks_x = DIV_CEILING(dims_(0), THREAD_3D_UNIT);
	const int num_blocks_y = DIV_CEILING(dims_(1), THREAD_3D_UNIT);
	const int num_blocks_z = DIV_CEILING(dims_(2), THREAD_3D_UNIT);
	const dim3 blocks(num_blocks_x, num_blocks_y, num_blocks_z);
	const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);

	integrateKernel<<<blocks, threads>>>(depth_map, tsdf_volume_, weight_volume_, dims_, voxel_length_, depth_scale, cam_params, sdf_trunc_, world_to_cam);
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
}

}  // namespace cuda