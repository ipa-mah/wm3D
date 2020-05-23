#pragma once
#include <cuda/common/common.hpp>
#include <eigen3/Eigen/Core>

namespace cuda
{
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

  public:
	__HOSTDEVICE__ int width()
	{
		return width_;
	}
	__HOSTDEVICE__ int height()
	{
		return height_;
	}

	__HOSTDEVICE__ CameraIntrinsicCuda()
	{
		width_ = -1;
		height_ = -1;
	}
	__HOSTDEVICE__ CameraIntrinsicCuda(int width, int height, float fx, float fy, float cx, float cy)
	{
		setIntrinsics(width, height, fx, fy, cx, cy);
	}
	__HOST__ explicit CameraIntrinsicCuda(const Eigen::Matrix3f& intrinsic, int width, int height)
	{
		width_ = width;
		height_ = height_;
		fx_ = intrinsic(0, 0);
		fy_ = intrinsic(1, 1);
		cx_ = intrinsic(0, 2);
		cy_ = intrinsic(1, 2);
		width_ = width;
		height_ = height;
	}
	__HOSTDEVICE__ void setIntrinsics(int width, int height, float fx, float fy, float cx, float cy)
	{
		width_ = width;
		height_ = height;
		fx_ = fx;
		inv_fx_ = 1.0f / fx_;
		fy_ = fy;
		inv_fy_ = 1.0f / fy_;
		cx_ = cx;
		cy_ = cy;
	}
	__HOSTDEVICE__ CameraIntrinsicCuda downSample()
	{
		CameraIntrinsicCuda ret;
		ret.setIntrinsics(width_ >> 1, height_ >> 1, fx_ * 0.5, fy_ * 0.5, cx_ * 0.5, cy_ * 0.5);
		return ret;
	}
	__HOSTDEVICE__ bool isValidPixel(const Eigen::Vector2f& p)
	{
		return p(0) >= 0 && p(0) < width_ - 1 && p(1) >= 0 && p(1) < height_ - 1;
	}
	__HOSTDEVICE__ bool isValidPixel(const Eigen::Vector2i& p)
	{
		return p(0) >= 0 && p(0) < width_ && p(1) >= 0 && p(1) < height_;
	}

	__HOSTDEVICE__ Eigen::Vector2f project3DPoint(const Eigen::Vector3f& pt, size_t level = 0)
	{
		return Eigen::Vector2f((fx_ * pt(0)) / pt(2) + cx_, (fy_ * pt(1)) / pt(2) + cy_);
	}
	__HOSTDEVICE__ Eigen::Vector3f inverseProjectPixel(const Eigen::Vector2f& pixel, float d)
	{
		return Eigen::Vector3f(d * (pixel(0) - cx_) * inv_fx_, d * (pixel(1) - cy_) * inv_fy_, d);
	}
	__HOSTDEVICE__ Eigen::Vector3f inverseProjectPixel(const Eigen::Vector2i& pixel, float d)
	{
		return Eigen::Vector3f(d * (pixel(0) - cx_) * inv_fx_, d * (pixel(1) - cy_) * inv_fy_, d);
	}
	~CameraIntrinsicCuda(){

	};
};
}  // namespace cuda
