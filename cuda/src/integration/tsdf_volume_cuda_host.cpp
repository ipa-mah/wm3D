#include <cuda/integration/tsdf_volume_cuda.hpp>
namespace cuda {
TSDFVolumeCuda::TSDFVolumeCuda()
{
    dims_ = Eigen::Vector3i(-1,-1,-1);
}

TSDFVolumeCuda::TSDFVolumeCuda(Eigen::Vector3i dims, float voxel_length, float sdf_trunc)
{
    voxel_length_ = voxel_length;
    sdf_trunc_ = sdf_trunc;
    create(dims);
    
}

TSDFVolumeCuda::~TSDFVolumeCuda()
{

}

void TSDFVolumeCuda::create(const Eigen::Vector3i& dims)
{
    if(device_ != nullptr)
    {
        std::cout<< "Already created"<<std::endl;
        return;
    }
    dims_ = dims;
    device_ = std::make_shared<TSDFVolumeCudaDevice>();
    const size_t NNN = dims_(0)* dims_(1) * dims_(2);
    CheckCuda(cudaMalloc(&(device_->tsdf_), sizeof(float) * NNN ));
    CheckCuda(cudaMalloc(&(device_->weight_), sizeof(uchar) * NNN));
    CheckCuda(cudaMalloc(&(device_->color_), sizeof(uchar3) * NNN));

    updateDevice();
    reset();

}

void TSDFVolumeCuda::release()
{
    if(device_ !=nullptr && device_.use_count() == 1)
    {
        CheckCuda(cudaFree(device_->tsdf_));
        CheckCuda(cudaFree(device_->weight_));
        CheckCuda(cudaFree(device_->color_));
    }
    device_ = nullptr;
}

void TSDFVolumeCuda::updateDevice()
{
    if(device_ !=nullptr)
    device_->dims_ = dims_;
    device_->voxel_length_ = voxel_length_;
    device_->inv_voxel_length_ = 1.0 / voxel_length_;
    device_->sdf_trunc_ = sdf_trunc_;

}

void TSDFVolumeCuda::reset()
{
    if(device_ !=nullptr){
        const size_t NNN = dims_(0)* dims_(1) * dims_(2);
 
        CheckCuda(cudaMemset(device_->color_,0, sizeof(uchar3) * NNN));
        CheckCuda(cudaMemset(device_->weight_,0, sizeof(uchar) * NNN));
        TSDFVolumeCudaKernel::reset(*this);
    }
}

void TSDFVolumeCuda::uploadVolume(std::vector<float>& tsdf, std::vector<uchar>& weight, std::vector<uchar3>& color)
{
    assert(device_ != nullptr);
    const size_t NNN = dims_(0)* dims_(1) * dims_(2);
    assert(tsdf.size() == NNN);
    assert(weight.size() == NNN);
    assert(color.size() == NNN);
    CheckCuda(cudaMemcpy(device_->tsdf_,tsdf.data(),sizeof(float) * NNN, cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(device_->color_,color.data(),sizeof(uchar3) * NNN, cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(device_->weight_, weight.data(), sizeof(uchar) * NNN, cudaMemcpyHostToDevice));
}

std::tuple<std::vector<float>, std::vector<uchar>, std::vector<uchar3>> TSDFVolumeCuda::downloadVolume()
{
    assert(device_ != nullptr);

    std::vector<float> tsdf;
    std::vector<uchar> weight;
    std::vector<uchar3> color;
    if (device_ == nullptr) {
        printf("Server not available!\n");
        return std::make_tuple(tsdf, weight, color);
    }
    const size_t NNN = dims_(0)* dims_(1) * dims_(2);
    tsdf.resize(NNN);
    weight.resize(NNN);
    color.resize(NNN);

    CheckCuda(cudaMemcpy(tsdf.data(),device_->tsdf_,sizeof(float) * NNN, cudaMemcpyDeviceToHost));
    CheckCuda(cudaMemcpy(weight.data(),device_->weight_,sizeof(uchar) * NNN, cudaMemcpyDeviceToHost));
    CheckCuda(cudaMemcpy(color.data(),device_->color_,sizeof(uchar3) * NNN, cudaMemcpyDeviceToHost));
    
    
    return std::make_tuple(std::move(tsdf), std::move(weight),
                           std::move(color));
}

void TSDFVolumeCuda::integrate(const DeviceArray2D<uchar3>& color_image,
                                const DeviceArray2D<ushort>& depth_image,
					            const CameraIntrinsicCuda& intrins,
                                const Eigen::Matrix4f& world_to_cam,
                                const float depth_scale)
    {

        assert(device_ != nullptr);
        TSDFVolumeCudaKernel::integrate(*this,color_image,depth_image,intrins,world_to_cam,depth_scale);
    
    }
                    

}