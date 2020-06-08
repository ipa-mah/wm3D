
#include <cuda/camera/camera_intrinsic_cuda.hpp>
#include <cuda/container/device_array.hpp>
#include <cuda/cuda_headers.hpp>
#include <cuda/geometry/triangle_mesh_cuda.hpp>
#include <cuda/integration/tsdf_volume_cuda.hpp>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <wm3D/integration/tsdf_volume.hpp>
#include <wm3D/utility/utils.hpp>
#include <cuda_runtime_api.h>

int main()
{
	int n = 100;
	int * vector_add = new int[n];
	for (size_t i = 0; i < n; i++)
	{
		vector_add[i] = 1;

	}
	int * d_vec ;
	cudaMalloc((void**)&d_vec, n*sizeof(int));
	cudaMemcpy(d_vec,vector_add, n * sizeof(int), cudaMemcpyHostToDevice);
	cuda::hostAddDevice(d_vec,n,10);
	cudaMemcpy(vector_add,d_vec,n * sizeof(int), cudaMemcpyDeviceToHost);
		for (size_t i = 0; i < n; i++)
	{
		std::cout<<vector_add[i]<<std::endl;
	}
	return 0;
}
