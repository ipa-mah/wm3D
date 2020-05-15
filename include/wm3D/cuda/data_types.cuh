/*!
 *****************************************************************
 * @file cpp_headers.hpp
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
#include <iostream>
#include <opencv2/opencv.hpp>
#include "wm3D/cuda/device_array.hpp"

#if !defined(__CUDACC__)
#include <eigen3/Eigen/Core>
#endif

#if __CUDA_ARCH__ < 300
#define MAX_THREADS 512
#else
#define MAX_THREADS 1024
#endif

#define cudaSafeCall(expr)  ___cudaSafeCall(expr, __FILE__, __LINE__, __func__)


static inline void error(const char *error_string, const char *file, const int line, const char *func)
{
    std::cout << "Error: " << error_string << "\t" << file << ":" << line << std::endl;
    exit(0);
}

static inline void ___cudaSafeCall(cudaError_t err, const char *file, const int line, const char *func = "")
{
    if (cudaSuccess != err)
        error(cudaGetErrorString(err), file, line, func);
}

//Tsdf fixed point divisor (if old format is enabled)
const int DIVISOR = 32767;     // SHRT_MAX;

struct mat33
{
    mat33() {}

#if !defined(__CUDACC__)
    mat33(Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & e)
    {
        memcpy(data, e.data(), sizeof(mat33));
    }
#endif

    float3 data[3];
};

//static inline int divUp(int total, int grain) { return (total + grain - 1) / grain; }
struct CameraParameters{
    //camera matrix
    int image_width,image_height;
    float focal_x, focal_y;
    float c_x,c_y;
    CameraParameters():image_width(0),image_height(0),focal_x(0),focal_y(0),c_x(0),c_y(0)
    {}

    CameraParameters(float fx, float fy, float cx, float cy,int height,int width) :
        focal_x(fx), focal_y(fy), c_x(cx), c_y(cy),image_height(height), image_width(width){}
    CameraParameters(float fx, float fy, float cx, float cy) :
        focal_x(fx), focal_y(fy), c_x(cx), c_y(cy){}

    /**
   * Build a paramid of input image
   */
    CameraParameters operator()(int level) const
    {
        int div = 1 << level;
        return (CameraParameters (focal_x / div, focal_y / div, c_x / div, c_y / div));
    }
    void printInfo()
    {
        std::cout<<this->focal_x<<" "<<0<<" "<<this->c_x<<std::endl;
        std::cout<<0<<" "<<this->focal_y<<" "<<this->c_y<<std::endl;
        std::cout<<0<<" "<<0<<" "<<1<<std::endl;
        std::cout<<"image width: "<<this->image_width<<std::endl;
        std::cout<<"image height: "<<this->image_height<<std::endl;
    }
    void initCamParams(const cv::Mat& camera_matrix,int height,int width)
    {
        this->focal_x = camera_matrix.at<float>(0,0);
        this->focal_y =  camera_matrix.at<float>(1,1);
        this->c_x =  camera_matrix.at<float>(0,2);
        this->c_y =  camera_matrix.at<float>(1,2);
        this->image_height = height;
        this->image_width = width;
    }
    cv::Mat cvMatMatrix()
    {
        return (cv::Mat_<float>(3,3)<< this->focal_x, 0.0f,this->c_x,
                                 0.0f ,this->focal_y, this->c_y,
                                 0.0f,0.0f,1.0f);
    }
};


