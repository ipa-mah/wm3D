#include <cuda/integration/tsdf_volume_cuda.hpp>
namespace cuda {
TSDFVolumeCuda::TSDFVolumeCuda()
{
    dims_ = Eigen::Vector3i(-1,-1,-1);
}

TSDFVolumeCuda::~TSDFVolumeCuda()
{

}

TSDFVolumeCuda::create(const Eigen::Vector3i& dims)
{
    if(device_ != nullptr)
    {
        std::cout<< "Already created"<<std::endl;
        return;
    }
    dims_ = dims;
    device_ = std::make_shared<TSDFVolumeCudaDevice>();

}

TSDFVolumeCuda::release()
{
    if(device_ !=nullptr && device_.use_count() == 1)
    {
        CheckCuda(cudaFree(device_->tsdf_));
        CheckCuda(cudaFree(device_->weight_));
        CheckCuda(cudaFree(device_->color_));
    }
    device_ = nullptr;
}
}