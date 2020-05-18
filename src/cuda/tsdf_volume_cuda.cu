#include <wm3D/cuda/tsdf_volume_cuda.h>
#include <cuda_runtime.h>
namespace cuda {

__device__ inline bool TSDFVolumeCudaDevice::inVolume(const Eigen::Vector3i& X)
{
    return 0 <= X(0) && X(0) < (N_ - 1) && 0 <= X(1) && X(1) < (N_ - 1) &&
    0 <= X(2) && X(2) < (N_ - 1);
}

} 