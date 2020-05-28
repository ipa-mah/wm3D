#include <wm3D/camera/camera_intrinsic.hpp>

CameraIntrinsic::CameraIntrinsic() : width_(-1), height_(-1), intrinsic_(Eigen::Matrix3d::Zero())
{
}
CameraIntrinsic::CameraIntrinsic(const CameraIntrinsic& other)
{
	intrinsic_ = other.intrinsic_;
	width_ = other.width_;
	height_ = other.height_;
}
CameraIntrinsic::CameraIntrinsic(int width, int height, double fx, double fy, double cx, double cy)
{
	setIntrinsics(width, height, fx, fy, cx, fy);
}
CameraIntrinsic::CameraIntrinsic(int width, int height, const cv::Mat& cv_intrinsic)
{
	setIntrinsics(width, height, cv_intrinsic.at<double>(0, 0), cv_intrinsic.at<double>(1, 1), cv_intrinsic.at<double>(0, 2), cv_intrinsic.at<double>(1, 2));
}

CameraIntrinsic::CameraIntrinsic(int width, int height, const Eigen::Matrix3d& intrinsic)
{
	width_ = width;
	height_ = height;
	intrinsic_ = intrinsic;
}

CameraIntrinsic::~CameraIntrinsic()
{
}

Eigen::Vector2d CameraIntrinsic::project3DPoint(const Eigen::Vector3d& cam_pt)
{
	return Eigen::Vector2d((intrinsic_(0, 0) * cam_pt(0)) / cam_pt(2) + intrinsic_(0, 2), (intrinsic_(1, 1) * cam_pt(1)) / cam_pt(2) + intrinsic_(1, 2));
}

Eigen::Vector3d CameraIntrinsic::inverseProjectPixel(const Eigen::Vector2d& pixel, double d)
{
	return Eigen::Vector3d(d * (pixel(0) - intrinsic_(0, 2)) / intrinsic_(0, 0), d * (pixel(1) - intrinsic_(1, 2)) / intrinsic_(1, 1), d);
}
Eigen::Vector3d CameraIntrinsic::inverseProjectPixel(const Eigen::Vector2i& pixel, double d)
{
	return Eigen::Vector3d(d * (pixel(0) - intrinsic_(0, 2)) / intrinsic_(0, 0), d * (pixel(1) - intrinsic_(1, 2)) / intrinsic_(1, 1), d);
}