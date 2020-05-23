#pragma once
#include <cuda/common/common.hpp>
#include <eigen3/Eigen/Core>

class CameraIntrinsicCuda
{
public:
    int width_;
    int height_;
    float fx_;
    float fy_;
    float cx_;
    float cy_;
    float inv_fx_;
    float inv_fy_;
private:
    /* data */

public:

    CameraIntrinsicCuda(/* args */);
    ~CameraIntrinsicCuda();
};

CameraIntrinsicCuda::CameraIntrinsicCuda(/* args */)
{
}

CameraIntrinsicCuda::~CameraIntrinsicCuda()
{
}
