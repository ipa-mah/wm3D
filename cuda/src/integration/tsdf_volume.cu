#include <cuda/cuda_headers.hpp>

namespace cuda
{
/**
 *  Kernel Volume: Compute TSDF Volume in device (GPU)
 *  depth_map: current depth image input
 *  color_map : current color image input
 *  tsdf_volume : output of tsdf volume
 *  color_volume : output of color volume
 *  rvect, tvec: transformation from current camera to base camera
 *  Detail can be found in Volumetric Representation chapter (listing 2)
 *  http://people.inf.ethz.ch/otmarh/download/Papers/p559-izadi(KinectFusion).pdf
 *
 */

__global__ void initializeVolumeKernel(PtrStepSz<float> tsdf_volume, PtrStepSz<float> weight_volume, const Eigen::Vector3i dims)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;
	if (x >= dims(0) || y >= dims(1) || z >= dims(2)) return;

	tsdf_volume.ptr(z * dims(1) + y)[x] = 0.0;
	weight_volume.ptr(z * dims(1) + y)[x] = 0.0;
}
void initializeVolume(DeviceArray2D<float>& tsdf_volume, DeviceArray2D<float>& weight_volume, const Eigen::Vector3i& dims)
{
	const int num_blocks_x = DIV_CEILING(dims(0), THREAD_3D_UNIT);
	const int num_blocks_y = DIV_CEILING(dims(1), THREAD_3D_UNIT);
	const int num_blocks_z = DIV_CEILING(dims(2), THREAD_3D_UNIT);
	const dim3 blocks(num_blocks_x, num_blocks_y, num_blocks_z);
	const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);

	initializeVolumeKernel<<<blocks, threads>>>(tsdf_volume, weight_volume, dims);
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

void integrateTsdfVolume(const DeviceArray2D<unsigned short>& depth_map, DeviceArray2D<float>& tsdf_volume, DeviceArray2D<float>& weight_volume, const Eigen::Vector3i& dims, const float voxel_length,
						 const float truncated_distance, const CameraIntrinsicCuda& cam_params, const Eigen::Matrix4f& world_to_cam, const float depth_scale)
{
	const int num_blocks_x = DIV_CEILING(dims(0), THREAD_3D_UNIT);
	const int num_blocks_y = DIV_CEILING(dims(1), THREAD_3D_UNIT);
	const int num_blocks_z = DIV_CEILING(dims(2), THREAD_3D_UNIT);
	const dim3 blocks(num_blocks_x, num_blocks_y, num_blocks_z);
	const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);

	integrateKernel<<<blocks, threads>>>(depth_map, tsdf_volume, weight_volume, dims, voxel_length, depth_scale, cam_params, truncated_distance, world_to_cam);
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
}

/*
__device__ __forceinline__
float trilinearInterpolation(const float3& point,
							 const PtrStepSz<float>& volume,
							 const int& volume_size)
{
  int3 point_in_grid = make_int3(__float2int_rn(point.x),
								 __float2int_rn(point.y),
								 __float2int_rn(point.z));

  const float vx = (__int2float_rn(point_in_grid.x) + 0.5f);
  const float vy = (__int2float_rn(point_in_grid.y) + 0.5f);
  const float vz = (__int2float_rn(point_in_grid.z) + 0.5f);

  point_in_grid.x = (point.x < vx) ? (point_in_grid.x - 1) : point_in_grid.x;
  point_in_grid.y = (point.y < vy) ? (point_in_grid.y - 1) : point_in_grid.y;
  point_in_grid.z = (point.z < vz) ? (point_in_grid.z - 1) : point_in_grid.z;

  const float a = (point.x - (__int2float_rn(point_in_grid.x) + 0.5f));
  const float b = (point.y - (__int2float_rn(point_in_grid.y) + 0.5f));
  const float c = (point.z - (__int2float_rn(point_in_grid.z) + 0.5f));

  return volume.ptr((point_in_grid.z) * volume_size + point_in_grid.y)[point_in_grid.x]  * (1 - a) * (1 - b) * (1 - c) +
	  volume.ptr((point_in_grid.z + 1) * volume_size + point_in_grid.y)[point_in_grid.x]  * (1 - a) * (1 - b) * c +
	  volume.ptr((point_in_grid.z) * volume_size + point_in_grid.y + 1)[point_in_grid.x]  * (1 - a) * b * (1 - c) +
	  volume.ptr((point_in_grid.z + 1) * volume_size + point_in_grid.y + 1)[point_in_grid.x]  * (1 - a) * b * c +
	  volume.ptr((point_in_grid.z) * volume_size + point_in_grid.y)[point_in_grid.x + 1]  * a * (1 - b) * (1 - c) +
	  volume.ptr((point_in_grid.z + 1) * volume_size + point_in_grid.y)[point_in_grid.x + 1]  * a * (1 - b) * c +
	  volume.ptr((point_in_grid.z) * volume_size + point_in_grid.y + 1)[point_in_grid.x + 1]  * a * b * (1 - c) +
	  volume.ptr((point_in_grid.z + 1) * volume_size + point_in_grid.y + 1)[point_in_grid.x + 1] * a * b * c;
}
__device__ __forceinline__
void getMaxMin(const float volume_range,const float3& origin,
			   const float3& direction, float& max_range, float& min_range)
{
  float txmin = ((direction.x > 0 ? 0.f : volume_range) - origin.x) / direction.x;
  float tymin = ((direction.y > 0 ? 0.f : volume_range) - origin.y) / direction.y;
  float tzmin = ((direction.z > 0 ? 0.f : volume_range) - origin.z) / direction.z;
  min_range = fmax(fmax(txmin, tymin), tzmin);
  float txmax = ((direction.x > 0 ? volume_range : 0.f) - origin.x) / direction.x;
  float tymax = ((direction.y > 0 ? volume_range : 0.f) - origin.y) / direction.y;
  float tzmax = ((direction.z > 0 ? volume_range : 0.f) - origin.z) / direction.z;

  max_range = fmin(fmin(txmax, tymax), tzmax);

}

__global__
void kernelRayCasting(const PtrStepSz<float> tsdf_volume,
					  PtrStepSz<float3> model_vertex,
					  PtrStepSz<float3> model_normal,
					  const int volume_size, const float voxel_scale,
					  const CameraParameters cam_parameters,
					  const float truncation_distance,
					  const float ray_step,
					  const mat33 cam_to_world_rot,
					  const float3 cam_to_world_trans)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= model_vertex.cols || y >= model_vertex.rows)
	return;
  model_vertex.ptr(y)[x] = make_float3(0, 0, 0);
  model_normal.ptr(y)[x] = make_float3(0, 0, 0);
  const float volume_max = volume_size * voxel_scale;
  const float3 pixel_position= make_float3(
		(x - cam_parameters.c_x) / cam_parameters.focal_x,
		(y - cam_parameters.c_y) / cam_parameters.focal_y,
		1.f);

  float3 ray_direction = (cam_to_world_rot * pixel_position);
  ray_direction = normalized(ray_direction);

  float min_range, max_range;
  getMaxMin(volume_max,cam_to_world_trans,ray_direction,max_range,min_range);
  float ray_length = fmax(min_range,0.f);
  if (ray_length >= max_range)
	return;
  ray_length += voxel_scale;
  float3 grid = (cam_to_world_trans + (ray_direction * ray_length)) / voxel_scale;

  float tsdf = tsdf_volume.ptr(
		__float2int_rd(grid.z) * volume_size + __float2int_rd(grid.y))[__float2int_rd(grid.x)];


  const float max_search_length = ray_length + volume_max * sqrt(2.f);

  for (; ray_length < max_search_length; ray_length += truncation_distance * ray_step) {
	grid = ((cam_to_world_trans + (ray_direction * (ray_length + truncation_distance * ray_step))) / voxel_scale);

	if (grid.x < 1 || grid.x >= volume_size - 1 || grid.y < 1 ||
		grid.y >= volume_size - 1 ||
		grid.z < 1 || grid.z >= volume_size - 1)
	  continue;

	const float previous_tsdf = tsdf;
	tsdf = tsdf_volume.ptr(
		  __float2int_rd(grid.z) * volume_size + __float2int_rd(grid.y))[__float2int_rd(
		  grid.x)];


	if (previous_tsdf < 0.f && tsdf > 0.f) //Zero crossing from behind
	  break;
	if (previous_tsdf > 0.f && tsdf < 0.f) { //Zero crossing
	  const float t_star =
		  ray_length - truncation_distance * ray_step * previous_tsdf / (tsdf - previous_tsdf);

	  const float3 vertex = cam_to_world_trans + ray_direction * t_star;

	  const float3 location_in_grid = (vertex / voxel_scale);
	  if (location_in_grid.x < 1 | location_in_grid.x >= volume_size - 1 ||
		  location_in_grid.y < 1 || location_in_grid.y >= volume_size - 1 ||
		  location_in_grid.z < 1 || location_in_grid.z >= volume_size - 1)
		break;
	  //Compute normal
	  float3 normal, shifted;
	  shifted = location_in_grid;
	  shifted.x += 1;
	  if (shifted.x >= volume_size - 1)
		break;
	  const float Fx1 = trilinearInterpolation(shifted, tsdf_volume, volume_size);
	  shifted = location_in_grid;
	  shifted.x -= 1;
	  if (shifted.x < 1)
		break;
	  const float Fx2 = trilinearInterpolation(shifted, tsdf_volume, volume_size);

	  normal.x = (Fx1 - Fx2);

	  shifted = location_in_grid;
	  shifted.y += 1;
	  if (shifted.y >= volume_size - 1)
		break;

	  const float Fy1 = trilinearInterpolation(shifted, tsdf_volume, volume_size);

	  shifted = location_in_grid;
	  shifted.y -= 1;
	  if (shifted.y < 1)
		break;
	  const float Fy2 = trilinearInterpolation(shifted, tsdf_volume, volume_size);

	  normal.y = (Fy1 - Fy2);

	  shifted = location_in_grid;
	  shifted.z += 1;
	  if (shifted.z >= volume_size - 1)
		break;
	  const float Fz1 = trilinearInterpolation(shifted, tsdf_volume, volume_size);

	  shifted = location_in_grid;
	  shifted.z -= 1;
	  if (shifted.z < 1)
		break;
	  const float Fz2 = trilinearInterpolation(shifted, tsdf_volume, volume_size);

	  normal.z = (Fz1 - Fz2);

	  if (norm(normal) == 0)
		break;

	  normal = normalized(normal);
	  //   printf("%f %f %f \n",vertex.x(), vertex.y(), vertex.z());

	  model_vertex.ptr(y)[x] = make_float3(vertex.x, vertex.y, vertex.z);
	  model_normal.ptr(y)[x] = make_float3(normal.x, normal.y, normal.z);
	  break;
	}
  }

}


void hostRayCasting(const DeviceArray2D<float>& tsdf_volume, DeviceArray2D<float3>& model_vertex,
					DeviceArray2D<float3>& model_normal,
					const CameraParameters& cam_params,const float truncation_distance,
					const int volume_res, const float voxel_size, const float ray_step,
					const mat33& cam_to_world_rot,const float3& cam_to_world_trans)
{



  dim3 block(32,8);
  dim3 grid((model_vertex.cols() + block.x - 1) / block.x,
			(model_vertex.rows() + block.y - 1) / block.y);

  kernelRayCasting<<<grid,block>>>(tsdf_volume,model_vertex,model_normal,volume_res,
								   voxel_size,cam_params,truncation_distance, ray_step,
								   cam_to_world_rot,cam_to_world_trans);

  CudaSafeCall ( cudaGetLastError () );
  CudaSafeCall (cudaDeviceSynchronize ());
}
*/

}  // namespace cuda
