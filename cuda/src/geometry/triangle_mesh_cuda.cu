
#include <cuda/cuda_headers.hpp>
namespace cuda
{
__global__ void printMeshInfoKernel(PtrSz<Eigen::Vector3f> vertices, const int N)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= N) return;
	printf("%f %f %f \n", vertices[idx](0), vertices[idx](1), vertices[idx](2));
}

void testTriangleMeshCuda(DeviceArray<Eigen::Vector3f>& vertices)
{
	const dim3 blocks(DIV_CEILING(vertices.size(), THREAD_1D_UNIT));
	const dim3 threads(THREAD_1D_UNIT);

	printMeshInfoKernel<<<blocks, threads>>>(vertices, vertices.size());
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
}

}  // namespace cuda
