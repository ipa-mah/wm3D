#pragma once
#include <eigen3/Eigen/Core>
#include <opencv2/imgproc.hpp>
#include <tuple>
class CameraIntrinsic
{
  public:
	/// \brief Default Constructor.
	CameraIntrinsic();

	CameraIntrinsic(const CameraIntrinsic& other);

	CameraIntrinsic(int width, int height, double fx, double fy, double cx, double cy);

	CameraIntrinsic(int width, int height, const Eigen::Matrix3d& intrinsic);

	CameraIntrinsic(int width, int height, const cv::Mat& cv_intrinsic);

	void setIntrinsics(int width, int height, int fx, int fy, int cx, int cy)
	{
		width_ = width;
		height_ = height;
		intrinsic_.setIdentity();
		intrinsic_(0, 0) = fx;
		intrinsic_(1, 1) = fy;
		intrinsic_(0, 2) = cx;
		intrinsic_(1, 2) = cy;
	}
	~CameraIntrinsic();

	Eigen::Vector2d project3DPoint(const Eigen::Vector3d& cam_pt);

	Eigen::Vector3d inverseProjectPixel(const Eigen::Vector2d& pixel, double d);
	Eigen::Vector3d inverseProjectPixel(const Eigen::Vector2i& pixel, double d);
	/// Returns `true` if both the width and height are greater than 0.
	bool isValid() const
	{
		return (width_ > 0 && height_ > 0);
	}
	cv::Mat cvMatMatrix()
	{
		return (cv::Mat_<double>(3, 3) << intrinsic_(0, 0), 0.0f, intrinsic_(0, 2), 0.0f, intrinsic_(1, 1), intrinsic_(1, 2), 0.0f, 0.0f, 1.0f);
	};

  public:
	int width_ = -1;
	int height_ = -1;
	Eigen::Matrix3d intrinsic_;
};
