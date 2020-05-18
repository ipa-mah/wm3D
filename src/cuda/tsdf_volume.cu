#include <wm3D/cuda/cuda_headers.cuh>
#include <wm3D/cuda/operators.cuh>

namespace Gpu
{
/*
__global__ void clearVolumeKernel (TsdfVolume tsdf)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if(x<tsdf.dims.x && y < tsdf.dims.y)
  {
	short2 *beg = tsdf.beg(x, y);
	short2 *end = beg + tsdf.dims.x * tsdf.dims.y * tsdf.dims.z;

	for(short2* pos = beg; pos != end; pos = tsdf.zstep(pos))
		  *pos = pack_tsdf (0.f, 0);
  }

}

void clearVolume(TsdfVolume volume)
{
  dim3 block (32, 8);
  dim3 grid (1, 1, 1);
  grid.x = divUp (volume.dims.x, block.x);
  grid.y = divUp (volume.dims.y, block.y);

  clearVolumeKernel<<<grid, block>>>(volume);
  cudaSafeCall ( cudaGetLastError () );
}

*/
}
