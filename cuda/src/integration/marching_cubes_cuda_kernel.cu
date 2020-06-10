#include <cuda/common/utils_cuda.hpp>
#include <cuda/cuda_headers.hpp>
#include <cuda/integration/marching_cubes_table_cuda.hpp>
static const int VERTEX_TO_ALLOCATE = -1;
namespace cuda
{
/*
__device__ Eigen::Vector3f gradient(const PtrStepSz<float> tsdf_volume, const PtrStepSz<float> weight_volume,
							const Eigen::Vector3i &idx, const int resolution) {
		Eigen::Vector3f n = Eigen::Vector3f::Zeros();
		Eigen::Vector3i idx1 = idx, idx0 = idx;

	#pragma unroll 1
		for (size_t k = 0; k < 3; ++k) {
			idx1(k) = WM3D_MIN(idx(k) + 1, resolution - 1);
			idx0(k) = WM3D_MAX(idx(k) - 1, 0);

			// if (weight_[IndexOf(X1)] != 0 && weight_[IndexOf(X0)] != 0) {
			//     n(k) = tsdf_[IndexOf(X1)] - tsdf_[IndexOf(X0)];
			//     n(k) *= 0.5;
			// } else if (weight_[IndexOf(X1)] != 0) {
			//     n(k) = tsdf_[IndexOf(X1)] - tsdf_[IndexOf(X)];
			// } else if (weight_[IndexOf(X0)] != 0) {
			//     n(k) = tsdf_[IndexOf(X)] - tsdf_[IndexOf(X0)];
			// } else {
			//     n(k) = 0;
			// }

			if(weight_volume.ptr(idx1(2) * resolution + idx1(1))[idx1(0)] != 0 &&
			weight_volume.ptr(idx0(2) * resolution + idx0(1))[idx0(0)] !=0 )
			{
				n(k) = tsdf_volume.ptr()
			}

			X1(k) = X0(k) = X(k);
		}
		return n;
	}
*/
__global__ void allocateVertexKernel(const PtrStepSz<float> tsdf_volume, const PtrStepSz<float> weight_volume, PtrStepSz<Eigen::Vector3i> vertex_indices, PtrStepSz<int> table_indices,
									 const Eigen::Vector3i dims)
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	const int z = threadIdx.z + blockIdx.x * blockDim.z;

	if (x >= dims(0) - 1 || y >= dims(1) - 1 || z >= dims(2) - 1) return;

	int tmp_table_index = 0;
	for (size_t corner = 0; corner < 8; corner++)
	{
		Eigen::Vector3i x_corner = Eigen::Vector3i(x + shift[corner][0], y + shift[corner][1], z + shift[corner][2]);
		float weight = weight_volume.ptr(x_corner(2) * dims(1) + x_corner(1))[x_corner(0)];
		if (weight == 0) return;
		float tsdf = tsdf_volume.ptr(x_corner(2) * dims(1) + x_corner(1))[x_corner(0)];
		if (fabs(tsdf) > 0.95f) return;
		tmp_table_index |= ((tsdf < 0) ? (1 << corner) : 0);
	}
	if (tmp_table_index == 0 || tmp_table_index == 255) return;
	table_indices.ptr(z * dims(1) + y)[x] = tmp_table_index;
	/** Tell them they will be extracted. Conflict can be ignored **/
	int edges = edge_table[tmp_table_index];
#pragma unroll 12
	for (int edge = 0; edge < 12; edge++)
	{
		if (edges & (1 << edge))
		{
			Eigen::Vector3i x_edge_holder = Eigen::Vector3i(x + edge_shift[edge][0], y + edge_shift[edge][1], z + edge_shift[edge][2]);
			vertex_indices.ptr(x_edge_holder(2) * dims(1) + x_edge_holder(1))[x_edge_holder(0)](edge_shift[edge][3]) = VERTEX_TO_ALLOCATE;
		}
	}
}

void allocateVertexHost(const DeviceArray2D<float>& tsdf_volume, const DeviceArray2D<float>& weight_volume, DeviceArray2D<Eigen::Vector3i>& vertex_indices, DeviceArray2D<int>& table_indices,
						const Eigen::Vector3i& dims)
{
	const int num_blocks_x = DIV_CEILING(dims(0), THREAD_3D_UNIT);
	const int num_blocks_y = DIV_CEILING(dims(1), THREAD_3D_UNIT);
	const int num_blocks_z = DIV_CEILING(dims(2), THREAD_3D_UNIT);
	const dim3 blocks(num_blocks_x, num_blocks_y, num_blocks_z);
	const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);
	allocateVertexKernel<<<blocks, threads>>>(tsdf_volume, weight_volume, vertex_indices, table_indices, dims);
}

}  // namespace cuda