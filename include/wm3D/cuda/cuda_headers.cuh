/*!
 *****************************************************************
 * @file cuda_headers.cuh
 *****************************************************************
 *
 * @note Copyright (c) 2019 Fraunhofer Institute for Manufacturing Engineering and Automation (IPA)
 * @note Project name: ipa_object_modeling
 * @author Author: Manh Ha Hoang
 *
 * @date Date of creation: 02.2019
 *
 *
 */
#pragma once
#include <cuda_runtime.h>
#include <wm3D/cuda/device_array.hpp>
#include <wm3D/cuda/data_types.cuh>
namespace Gpu
{
// define warp size for NVIDA GPU
#define WARP_SIZE 32
// Define this to turn on error checking
#define CUDA_ERROR_CHECK
#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define CudaCheckError() __cudaCheckError(__FILE__, __LINE__)
inline void __cudaSafeCall(cudaError err, const char* file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif
	return;
}

void createRenderMap(const DeviceArray2D<float3>& normals, DeviceArray2D<uchar3>& render_image);
void createVMap(const DeviceArray2D<unsigned short>& depth_map, DeviceArray2D<float3>& vertex_map, const CameraParameters& cam_params, const float depth_cutoff = 5.0f,
				const float depth_scale = 0.001f);
void createNMap(const DeviceArray2D<float3>& vmap, DeviceArray2D<float3>& nmap);
void createGaussianFilter(const DeviceArray2D<float>& src, DeviceArray2D<float>& dst);
void computeFaceNormal(const DeviceArray<float3>& vertices);
void createDepthBoundaryMask(const DeviceArray2D<unsigned short>& src, DeviceArray2D<short>& dx, DeviceArray2D<short>& dy, DeviceArray2D<uchar>& mask_map, const float depth_threshold);
void hostMaskImage(const DeviceArray2D<float>& sobel_dx, const DeviceArray2D<float>& sobel_dy, DeviceArray2D<uchar>& mask_map, const float depth_threshold);


}  // namespace Gpu
