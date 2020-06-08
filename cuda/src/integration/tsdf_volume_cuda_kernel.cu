#include <cuda/integration/tsdf_volume_cuda.hpp>
#include <cuda/common/operators.cuh>
namespace cuda
{
__global__ void initializeVolumeKernel(PtrStepSz<float> tsdf_volume, PtrStepSz<float> weight_volume, 
              const int res)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;
	if (x >= res || y >= res || z >= res) return;

	tsdf_volume.ptr(z * res + y)[x] = 0.0;
	weight_volume.ptr(z * res + y)[x] = 0.0;
}
void TSDFVolumeCuda::initializeVolume()
{
	const int num_blocks_x = DIV_CEILING(res_, THREAD_3D_UNIT);
	const int num_blocks_y = DIV_CEILING(res_, THREAD_3D_UNIT);
	const int num_blocks_z = DIV_CEILING(res_, THREAD_3D_UNIT);
	const dim3 blocks(num_blocks_x, num_blocks_y, num_blocks_z);
	const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);

	initializeVolumeKernel<<<blocks, threads>>>(tsdf_volume_, weight_volume_, res_);
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
}

__global__ void integrateKernel(const PtrStepSz<unsigned short> depth_image,
					 PtrStepSz<float> tsdf_volume, PtrStepSz<float> weight_volume,
					  			const int res, float voxel_length,
								const float depth_scale, 
								const CameraIntrinsicCuda cam_params,
								const float truncation_distance, 
								const Eigen::Matrix4f world_to_cam)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= res || y >= res || z >= res) return;

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

		const float current_tsdf = tsdf_volume.ptr(z * res + y)[x];
		const short current_weight = weight_volume.ptr(z * res + y)[x];

		const float add_weight = 1;
		const float updated_tsdf = (current_weight * current_tsdf + add_weight * new_tsdf) / (current_weight + add_weight);

		const float new_weight = current_weight + add_weight;
		// const float new_weight = min(current_weight + add_weight, 128.0f);

		tsdf_volume.ptr(z * res + y)[x] = updated_tsdf;
		weight_volume.ptr(z * res + y)[x] = new_weight;
	}
}

void TSDFVolumeCuda::integrateTsdfVolume(const DeviceArray2D<unsigned short>& depth_map, 
  const CameraIntrinsicCuda& cam_params, const Eigen::Matrix4f& world_to_cam, const float depth_scale)
{
	const int num_blocks_x = DIV_CEILING(res_, THREAD_3D_UNIT);
	const int num_blocks_y = DIV_CEILING(res_, THREAD_3D_UNIT);
	const int num_blocks_z = DIV_CEILING(res_, THREAD_3D_UNIT);
	const dim3 blocks(num_blocks_x, num_blocks_y, num_blocks_z);
	const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);

	integrateKernel<<<blocks, threads>>>(depth_map, tsdf_volume_, weight_volume_, res_, voxel_length_, depth_scale, cam_params, sdf_trunc_, world_to_cam);
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
}

/// Ray casting



/*Ray Casting*/

__device__ __forceinline__
float trilinearInterpolation(const Eigen::Vector3f& point,
                             const PtrStepSz<float>& volume,
                             const int& volume_size)
{
  int3 point_in_grid = make_int3(__float2int_rn(point(0)),
                                 __float2int_rn(point(1)),
                                 __float2int_rn(point(2)));

  const float vx = (__int2float_rn(point_in_grid.x) + 0.5f);
  const float vy = (__int2float_rn(point_in_grid.y) + 0.5f);
  const float vz = (__int2float_rn(point_in_grid.z) + 0.5f);

  point_in_grid.x = (point(0) < vx) ? (point_in_grid.x - 1) : point_in_grid.x;
  point_in_grid.y = (point(1) < vy) ? (point_in_grid.y - 1) : point_in_grid.y;
  point_in_grid.z = (point(2) < vz) ? (point_in_grid.z - 1) : point_in_grid.z;

  const float a = (point(0) - (__int2float_rn(point_in_grid.x) + 0.5f));
  const float b = (point(1) - (__int2float_rn(point_in_grid.y) + 0.5f));
  const float c = (point(2) - (__int2float_rn(point_in_grid.z) + 0.5f));

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
void getMaxMin(const float volume_max,const Eigen::Vector3f& origin,
               const Eigen::Vector3f& ray_dir, float& max_range, float& min_range)
{
  float txmin = ((ray_dir(0) > 0 ? 0.f : volume_max) - origin(0)) / ray_dir(0);
  float tymin = ((ray_dir(1) > 0 ? 0.f : volume_max) - origin(1)) / ray_dir(1);
  float tzmin = ((ray_dir(2) > 0 ? 0.f : volume_max) - origin(2)) / ray_dir(2);
  min_range = fmax(fmax(txmin, tymin), tzmin);
  float txmax = ((ray_dir(0) > 0 ? volume_max : 0.f) - origin(0)) / ray_dir(0);
  float tymax = ((ray_dir(1) > 0 ? volume_max : 0.f) - origin(1)) / ray_dir(1);
  float tzmax = ((ray_dir(2) > 0 ? volume_max : 0.f) - origin(2)) / ray_dir(2);

  max_range = fmin(fmin(txmax, tymax), tzmax);

}

__global__
void kernelRayCasting(const PtrStepSz<float> tsdf_volume,
                      PtrStepSz<float3> model_vertex,
                      PtrStepSz<float3> model_normal,
                      const int volume_size, const float voxel_size,
                      const CameraIntrinsicCuda intrins,
                      const float truncation_distance,
                      const float ray_step,
                      const Eigen::Matrix3f cam_to_world_rot,
                      const Eigen::Vector3f cam_to_world_trans)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= model_vertex.cols || y >= model_vertex.rows)
    return;
  model_vertex.ptr(y)[x] = make_float3(0, 0, 0);
  model_normal.ptr(y)[x] = make_float3(0, 0, 0);
  float volume_max = volume_size * voxel_size;
  // ray direction w.r.t camera
  const Eigen::Vector3f ray_dir_cam= Eigen::Vector3f(
        (x - intrins.cx_) / intrins.fx_,
        (y - intrins.cy_) / intrins.fy_,
        1.f);
  // ray direction w.r.t world
  Eigen::Vector3f ray_dir = (cam_to_world_rot * ray_dir_cam);
  ray_dir.normalize();

  float min_range, max_range;
  getMaxMin(volume_max,cam_to_world_trans,ray_dir,max_range,min_range);
  //original length of ray
  float ray_length = fmax(min_range,0.f);
  if (ray_length >= max_range)
    return;
  ray_length += voxel_size;
  Eigen::Vector3f grid = (cam_to_world_trans + (ray_dir * ray_length)) / voxel_size;

  float tsdf = tsdf_volume.ptr(
        __float2int_rd(grid(2)) * volume_size + __float2int_rd(grid(1)))[__float2int_rd(grid(0))];


  const float max_search_length = ray_length + volume_max * sqrt(2.f);

  for (; ray_length < max_search_length; ray_length += truncation_distance * ray_step) {
    grid = ((cam_to_world_trans + (ray_dir * (ray_length + truncation_distance * ray_step))) / voxel_size);

	if (grid(0) < 1 || grid(0) >= volume_size - 1 ||
	    grid(1) < 1 || grid(1) >= volume_size - 1 ||
        grid(2) < 1 || grid(2) >= volume_size - 1)
      continue;

    const float previous_tsdf = tsdf;
    tsdf = tsdf_volume.ptr(
		  __float2int_rd(grid(2)) * volume_size + 
		  __float2int_rd(grid(1)))[__float2int_rd(grid(0))];


    if (previous_tsdf < 0.f && tsdf > 0.f) //Zero crossing from behind
      break;
    if (previous_tsdf > 0.f && tsdf < 0.f) { //Zero crossing
      const float t_star =
          ray_length - truncation_distance * ray_step * previous_tsdf / (tsdf - previous_tsdf);

      const Eigen::Vector3f vertex = cam_to_world_trans + ray_dir * t_star;

      const Eigen::Vector3f location_in_grid = (vertex / voxel_size);
      if (location_in_grid(0) < 1 | location_in_grid(0) >= volume_size - 1 ||
          location_in_grid(1) < 1 || location_in_grid(1) >= volume_size - 1 ||
          location_in_grid(2) < 1 || location_in_grid(2) >= volume_size - 1)
        break;
      //Compute normal
      Eigen::Vector3f normal, shifted;
      shifted = location_in_grid;
      shifted(0) += 1;
      if (shifted(0) >= volume_size - 1)
        break;
      const float Fx1 = trilinearInterpolation(shifted, tsdf_volume, volume_size);
      shifted = location_in_grid;
      shifted(0) -= 1;
      if (shifted(0) < 1)
        break;
      const float Fx2 = trilinearInterpolation(shifted, tsdf_volume, volume_size);

      normal(0) = (Fx1 - Fx2);

      shifted = location_in_grid;
      shifted(1) += 1;
      if (shifted(1) >= volume_size - 1)
        break;

      const float Fy1 = trilinearInterpolation(shifted, tsdf_volume, volume_size);

      shifted = location_in_grid;
      shifted(1) -= 1;
      if (shifted(1) < 1)
        break;
      const float Fy2 = trilinearInterpolation(shifted, tsdf_volume, volume_size);

      normal(1) = (Fy1 - Fy2);

      shifted = location_in_grid;
      shifted(2) += 1;
      if (shifted(2) >= volume_size - 1)
        break;
      const float Fz1 = trilinearInterpolation(shifted, tsdf_volume, volume_size);

      shifted = location_in_grid;
      shifted(2) -= 1;
      if (shifted(2) < 1)
        break;
      const float Fz2 = trilinearInterpolation(shifted, tsdf_volume, volume_size);

      normal(2) = (Fz1 - Fz2);

      if (normal.norm() == 0)
        break;

      normal.normalize();
      model_vertex.ptr(y)[x] = make_float3(vertex(0), vertex(1), vertex(2));
      model_normal.ptr(y)[x] = make_float3(normal(0), normal(1), normal(2));
      break;
    }
  }

}


void TSDFVolumeCuda::rayCasting(DeviceArray2D<float3>& model_vertex,
                    DeviceArray2D<float3>& model_normal,
                    const CameraIntrinsicCuda& intrins,
					const Eigen::Matrix4f& cam_to_world,
					const float ray_step)
{


	Eigen::Matrix3f cam_to_world_rot = cam_to_world.topLeftCorner(3, 3);
	Eigen::Vector3f cam_to_world_trans = cam_to_world.topRightCorner(3,1);
	model_normal.create(intrins.height_,intrins.width_);
	model_vertex.create(intrins.height_,intrins.width_);

	const dim3 blocks(DIV_CEILING(intrins.width_, THREAD_2D_UNIT),
					 DIV_CEILING(intrins.height_, THREAD_2D_UNIT));
	const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);


  kernelRayCasting<<<blocks,threads>>>(tsdf_volume_,model_vertex,model_normal,res_,
                                   voxel_length_,intrins,sdf_trunc_, ray_step,
                                   cam_to_world_rot,cam_to_world_trans);

	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
}


class AddDevice
{
public:
  AddDevice(const int a);
  int a_;
  
  __device__ __forceinline__ 
  void operator()(int * vector_add, const int n) const 
  {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= n) return;
    vector_add[idx] += a_;
  }
};

inline AddDevice::AddDevice(const int a) : a_(a) 
{  
}

__global__ void print_device_kernel(const AddDevice device , int* vector_add, const int n)
{
  device(vector_add,n);
};

void hostAddDevice(int * vector_add, const int n, const int a)
{
  AddDevice device(a);
  const dim3 blocks(DIV_CEILING(n,THREAD_1D_UNIT));
  const dim3 threads(THREAD_1D_UNIT);
  print_device_kernel<<<blocks,threads>>>(device,vector_add,n);
  CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
}
/////////////////////////////////////////////////////////////////////////////////////////////


__device__ Eigen::Vector3f TSDFVolumeDevice::gradient(const Eigen::Vector3i &idx)
{
  Eigen::Vector3f n = Eigen::Vector3f::Zero();
  Eigen::Vector3i idx1 = idx, idx0 = idx;

#pragma unroll 1
  for(size_t k = 0 ; k < 3 ; k ++)
  {
      idx1(k) = WM3D_MIN(idx(k) + 1, res_ - 1);
      idx0(k) = WM3D_MAX(idx(k) - 1, 0);
         if(weight_val(idx1) != 0 && weight_val(idx0) != 0)
         {
            n(k) = tsdf_val(idx1) - tsdf_val(idx0);
            n(k) *= 0.5;
         } 
         else if(weight_val(idx1) != 0)
         {
          n(k) = tsdf_val(idx1) - tsdf_val(idx);
         }
         else if (weight_val(idx0) != 0)
         {
           n(k) =  tsdf_val(idx) - tsdf_val(idx0);
         }
         else
         {
           n(k) = 0;
         }
         idx1(k) = idx0(k) = idx(k);

  }
  return n;

}

__device__ void TSDFVolumeDevice::integrate(const Eigen::Vector3i& voxel_idx,
        const PtrStepSz<unsigned short>& depth_image, 
        const CameraIntrinsicCuda& cam_params, const Eigen::Matrix4f& world_to_cam, 
        const float depth_scale)
 {

// Convert to voxel grid to global coordinate
const float3 global_voxel = make_float3((static_cast<float>(voxel_idx(0)) + 0.5f) * voxel_length_, 
                                        (static_cast<float>(voxel_idx(1)) + 0.5f) * voxel_length_, 
                                        (static_cast<float>(voxel_idx(2)) + 0.5f) * voxel_length_);
// convert voxel from global to local camera coordinate
const Eigen::Vector3f camera_voxel = (world_to_cam * Eigen::Vector4f(global_voxel.x, global_voxel.y, global_voxel.z, 1.0)).head<3>();
if (camera_voxel(2) <= 0) return;
	// projection
  const int2 uv = make_int2(__float2int_rn(camera_voxel(0) / camera_voxel(2) * cam_params.fx_ + cam_params.cx_),
                           __float2int_rn(camera_voxel(1) / camera_voxel(2) * cam_params.fy_ + cam_params.cy_));
	if (uv.x < 0 || uv.x >= depth_image.cols || uv.y < 0 || uv.y >= depth_image.rows) return;

	const float depth = depth_image.ptr(uv.y)[uv.x] * depth_scale;

	if (depth <= 0.0001 || depth > 5.0) return;
	const float sdf = (depth - camera_voxel(2));
	if (sdf >= -sdf_trunc_)
	{
		const float new_tsdf = fmin(1.f, sdf / sdf_trunc_);

		const float current_tsdf = tsdf_volume_.ptr(voxel_idx(2) * res_ + voxel_idx(1))[voxel_idx(0)];
		const short current_weight = weight_volume_.ptr(voxel_idx(2) * res_ + voxel_idx(1))[voxel_idx(0)];

		const float add_weight = 1;
		const float updated_tsdf = (current_weight * current_tsdf + add_weight * new_tsdf) / (current_weight + add_weight);

		const float new_weight = current_weight + add_weight;
		// const float new_weight = min(current_weight + add_weight, 128.0f);

		tsdf_volume_.ptr(voxel_idx(2) * res_ + voxel_idx(1))[voxel_idx(0)] = updated_tsdf;
		weight_volume_.ptr(voxel_idx(2) * res_ + voxel_idx(1))[voxel_idx(0)] = new_weight;
	}


 }


}  // namespace cuda