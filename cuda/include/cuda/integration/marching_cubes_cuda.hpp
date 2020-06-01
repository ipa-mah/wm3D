#pragma once
#include <eigen3/Eigen/Core>
#include <cuda/common/common.hpp>
#include <cuda/container/device_array.hpp>

class MarchingCubesVolumeCudaDevice
{
private:
    uchar *table_indices_;
    Eigen::Vector3i *vertex_indices_;
public:
    MarchingCubesVolumeCudaDevice(/* args */);
    ~MarchingCubesVolumeCudaDevice();
};

MarchingCubesVolumeCudaDevice::MarchingCubesVolumeCudaDevice(/* args */)
{
}

MarchingCubesVolumeCudaDevice::~MarchingCubesVolumeCudaDevice()
{
}
