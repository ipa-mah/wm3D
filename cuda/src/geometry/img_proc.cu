#include <cuda/geometry/img_proc.hpp>

namespace cuda
{
__global__ void vertexMapKernel(const PtrStepSz<unsigned short> depth_map, PtrStep<float3> vertex_map, const CameraIntrinsicCuda cam_params, const float depth_cutoff, const float depth_scale)
{
	// Get id of each thread in dimension x,y
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	// if index is outside image dimension
	if (x >= depth_map.cols || y >= depth_map.rows) return;
	// Value in meters of pixel (x,y)
	float depth_value = depth_map.ptr(y)[x] * depth_scale;
	if (depth_value != 0 && depth_value < depth_cutoff)
	{
		vertex_map.ptr(y)[x] = make_float3((x - cam_params.cx_) * depth_value / cam_params.fx_, (y - cam_params.cy_) * depth_value / cam_params.fy_, depth_value);
	}
	else
	{
		vertex_map.ptr(y)[x] = make_float3(__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));
	}
}
void createVMap(const DeviceArray2D<unsigned short>& depth_map, DeviceArray2D<float3>& vertex_map, const CameraIntrinsicCuda& cam_params, const float depth_cutoff, const float depth_scale)
{
	vertex_map.create(depth_map.rows(), depth_map.cols());

	const dim3 blocks(DIV_CEILING(depth_map.cols(), THREAD_2D_UNIT), DIV_CEILING(depth_map.rows(), THREAD_2D_UNIT));
	const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);

	vertexMapKernel<<<blocks, threads>>>(depth_map, vertex_map, cam_params, depth_cutoff, depth_scale);

	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
}

__global__ void computeNmapKernel(const PtrStepSz<float3> vmap, PtrStep<float3> nmap)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= vmap.cols || y >= vmap.rows) return;
	float3 n_out = make_float3(__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));

	if (x < vmap.cols - 1 && y < vmap.rows - 1)
	{
		Eigen::Vector3f v00(vmap.ptr(y)[x].x, vmap.ptr(y)[x].y, vmap.ptr(y)[x].z);
		Eigen::Vector3f v01(vmap.ptr(y)[x + 1].x, vmap.ptr(y)[x + 1].y, vmap.ptr(y)[x + 1].z);
		Eigen::Vector3f v10(vmap.ptr(y + 1)[x].x, vmap.ptr(y + 1)[x].y, vmap.ptr(y + 1)[x].z);

		if (v00(2) * v01(2) * v10(2) != 0)
		{
			Eigen::Vector3f d010(v01(0) - v00(0), v01(1) - v00(1), v01(2) - v00(2));
			Eigen::Vector3f d100(v10(0) - v00(0), v10(1) - v00(1), v10(2) - v00(2));
			Eigen::Vector3f normal = d010.cross(d100);
			normal.normalize();
			n_out = make_float3(normal(0), normal(1), normal(2));
		}
	}
	nmap.ptr(y)[x] = n_out;
}

void createNMap(const DeviceArray2D<float3>& vmap, DeviceArray2D<float3>& nmap)
{
	nmap.create(vmap.rows(), vmap.cols());

	const dim3 blocks(DIV_CEILING(vmap.cols(), THREAD_2D_UNIT), DIV_CEILING(vmap.rows(), THREAD_2D_UNIT));
	const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);

	computeNmapKernel<<<blocks, threads>>>(vmap, nmap);
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
}

__global__ void computeRenderMapKernel(const PtrStepSz<float3> normals, PtrStep<uchar3> render_image)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= normals.cols || y >= normals.rows) return;

	float3 n = normals.ptr(y)[x];

#if 0
  unsigned char r = static_cast<unsigned char>(__saturatef((-n.x + 1.f)/2.f) * 255.f);
  unsigned char g = static_cast<unsigned char>(__saturatef((-n.y + 1.f)/2.f) * 255.f);
  unsigned char b = static_cast<unsigned char>(__saturatef((-n.z + 1.f)/2.f) * 255.f);
#else
	unsigned char r = static_cast<unsigned char>((5.f - n.x * 3.5f) * 25.5f);
	unsigned char g = static_cast<unsigned char>((5.f - n.y * 2.5f) * 25.5f);
	unsigned char b = static_cast<unsigned char>((5.f - n.z * 3.5f) * 25.5f);
#endif
	render_image.ptr(y)[x] = make_uchar3(b, g, r);
}

void createRenderMap(const DeviceArray2D<float3>& normals, DeviceArray2D<uchar3>& render_image)
{
	render_image.create(normals.rows(), normals.cols());

	const dim3 blocks(DIV_CEILING(normals.cols(), THREAD_2D_UNIT), DIV_CEILING(normals.rows(), THREAD_2D_UNIT));
	const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);

	computeRenderMapKernel<<<blocks, threads>>>(normals, render_image);

	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
}

/*
 * Compute gaussian filter kernel
 */
__global__ void kernelGaussFilter(const PtrStepSz<float> src, PtrStep<float> dst, float* gauss_kernel)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= src.cols || y >= src.rows) return;
	const int D = 5;

	float center = src.ptr(2 * y)[2 * x];
	// bounding
	int tx = min(2 * x - D / 2 + D, src.cols - 1);
	int ty = min(2 * y - D / 2 + D, src.rows - 1);
	int cy = max(0, 2 * y - D / 2);

	float sum = 0;
	int count = 0;

	for (; cy < ty; ++cy)
	{
		for (int cx = max(0, 2 * x - D / 2); cx < tx; ++cx)
		{
			if (!isnan(src.ptr(cy)[cx]))
			{
				sum += src.ptr(cy)[cx] * gauss_kernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
				count += gauss_kernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
			}
		}
	}
	dst.ptr(y)[x] = (float)(sum / (float)count);
}

void createGaussianFilter(const DeviceArray2D<float>& src, DeviceArray2D<float>& dst)
{
	dst.create(src.rows(), src.cols());

	const dim3 blocks(DIV_CEILING(src.cols(), THREAD_2D_UNIT), DIV_CEILING(src.rows(), THREAD_2D_UNIT));
	const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);

	const float gaussKernel[25] = {1, 4, 6, 4, 1, 4, 16, 24, 16, 4, 6, 24, 36, 24, 6, 4, 16, 24, 16, 4, 1, 4, 6, 4, 1};
	float* gauss_cuda;
	cudaMalloc((void**)&gauss_cuda, sizeof(float) * 25);
	cudaMemcpy(gauss_cuda, &gaussKernel[0], sizeof(float) * 25, cudaMemcpyHostToDevice);
	kernelGaussFilter<<<blocks, threads>>>(src, dst, gauss_cuda);

	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
}

__global__ void kernelComputeFaceNormals(const float3* src, const int size)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= size) return;
	printf("%d %f %f %f \n", size, src[x].x, src[x].y, src[x].z);
}
void computeFaceNormal(const DeviceArray<float3>& vertices)
{
	dim3 block(32, 8);
	dim3 grid((vertices.size() + block.x - 1) / block.x);

	kernelComputeFaceNormals<<<grid, block>>>(vertices, vertices.size());
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
}
/////////TODO////////////////////

__constant__ float gsobel_x3x3[9];
__constant__ float gsobel_y3x3[9];

__global__ void applyKernel(const PtrStepSz<unsigned short> src, PtrStep<short> dx, PtrStep<short> dy, PtrStep<uchar> mask_map, const float depth_threshold)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= src.cols || y >= src.rows) return;
	// by default, mask pixel = 255 -> white
	mask_map.ptr(y)[x] = 255;
	// if depth pixel = 0, return
	if (src.ptr(y)[x] == 0) return;

	float dxVal = 0;
	float dyVal = 0;

	int kernelIndex = 8;
	for (int j = max(y - 1, 0); j <= min(y + 1, src.rows - 1); j++)
	{
		for (int i = max(x - 1, 0); i <= min(x + 1, src.cols - 1); i++)
		{
			dxVal += (float)src.ptr(j)[i] * gsobel_x3x3[kernelIndex];
			dyVal += (float)src.ptr(j)[i] * gsobel_y3x3[kernelIndex];
			--kernelIndex;
		}
	}

	dx.ptr(y)[x] = dxVal;
	dy.ptr(y)[x] = dyVal;
	float mag = static_cast<float>(sqrt((float)(dxVal * dxVal + dyVal * dyVal))) * 0.001;  // in meters
	if (mag < depth_threshold) mask_map.ptr(y)[x] = 0;
}
void createDepthBoundaryMask(const DeviceArray2D<unsigned short>& src, DeviceArray2D<short>& dx, DeviceArray2D<short>& dy, DeviceArray2D<uchar>& mask_map, const float depth_threshold)
{
	dx.create(src.rows(), src.cols());
	dy.create(src.rows(), src.cols());
	mask_map.create(src.rows(), src.cols());
	static bool once = false;

	if (!once)
	{
		//    float gsx3x3[9] = {0.52201,  0.00000, -0.52201,
		//                       0.79451, -0.00000, -0.79451,
		//                       0.52201,  0.00000, -0.52201};

		//    float gsy3x3[9] = {0.52201, 0.79451, 0.52201,
		//                       0.00000, 0.00000, 0.00000,
		//                       -0.52201, -0.79451, -0.52201};
		float gsx3x3[9] = {-1.000, 0.00000, 1.000, -2.000, -0.00000, 2.000, -1.000, 0.00000, 1.000};

		float gsy3x3[9] = {-1.00, -2.000, -1.000, 0.00000, 0.00000, 0.00000, 1.000, 2.000, 1.000};

		cudaMemcpyToSymbol(gsobel_x3x3, gsx3x3, sizeof(float) * 9);
		cudaMemcpyToSymbol(gsobel_y3x3, gsy3x3, sizeof(float) * 9);

		CheckCuda(cudaDeviceSynchronize());
		CheckCuda(cudaGetLastError());

		once = true;
	}

	dim3 block(32, 8);
	dim3 grid((src.cols() + block.x - 1) / block.x, (src.rows() + block.y - 1) / block.y);

	applyKernel<<<grid, block>>>(src, dx, dy, mask_map, depth_threshold);

	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
}

__global__ void kernelMaskImage(const PtrStepSz<float> sobel_dx, const PtrStepSz<float> sobel_dy, PtrStep<uchar> mask_map, const float depth_threshold)
{
	// Get id of each thread in dimension x,y
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	// if index is outside image dimension
	if (x >= sobel_dx.cols || y >= sobel_dx.rows) return;
	float dx = sobel_dx.ptr(y)[x];
	float dy = sobel_dy.ptr(y)[x];
	float mag = static_cast<float>(sqrt((float)(dx * dx + dy * dy)));  // in meters
	// printf("%f \n",mag);
	if (mag > depth_threshold)
	{
		mask_map.ptr(y)[x] = 255;
	}
	else
	{
		mask_map.ptr(y)[x] = 0;
	}
}
void hostMaskImage(const DeviceArray2D<float>& sobel_dx, const DeviceArray2D<float>& sobel_dy, DeviceArray2D<uchar>& mask_map, const float depth_threshold)
{
	mask_map.create(sobel_dx.rows(), sobel_dx.cols());

	const dim3 blocks(DIV_CEILING(sobel_dx.cols(), THREAD_2D_UNIT), DIV_CEILING(sobel_dx.rows(), THREAD_2D_UNIT));
	const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);

	kernelMaskImage<<<blocks, threads>>>(sobel_dx, sobel_dy, mask_map, depth_threshold);
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
}

}  // namespace cuda
