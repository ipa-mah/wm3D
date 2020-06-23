#pragma once

#include <algorithm>  // std::min
#include <iterator>   // for back_inserter
#include <thread>
#include <unordered_map>

#include <cuda/cuda_headers.hpp>
#include <cuda/common/utils_cuda.hpp>
#include <wm3D/camera/camera_intrinsic.hpp>
#include <wm3D/integration/marching_cubes_table.hpp>
#include <wm3D/utility/utils.hpp>
// OpenCV
#include <opencv2/opencv.hpp>
// PCL
#include <pcl/PolygonMesh.h>
#include <pcl/common/transforms.h>
#include <pcl/io/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// Headers
class TSDFVolume
{
  public:
	using Ptr = std::shared_ptr<TSDFVolume>;
	using ConstPtr = std::shared_ptr<const TSDFVolume>;
	TSDFVolume(const Eigen::Vector3i& dims, const float voxel_length);
	pcl::PolygonMesh extractMesh(const Eigen::Vector3d& crop_min, const Eigen::Vector3d& crop_max);
	void downloadTsdfAndWeights(const DeviceArray2D<float>& device_tsdf, const DeviceArray2D<float>& device_weights);

	pcl::PointCloud<pcl::PointNormal>::Ptr extractPointCloud(const Eigen::Vector3d& crop_min, const Eigen::Vector3d& crop_max);
	///
	/// \brief IndexOf
	/// \param x
	/// \param y
	/// \param z
	/// \return return index of voxel in tsdf volume
	///
	void cpuTSDF(const cv::Mat& depth_map, const Eigen::Matrix4d& cam2world);

  private:
	std::vector<float> tsdf_;
	std::vector<float> weights_;
	Eigen::Vector3d origin_;
	Eigen::Vector3i dims_;
	float voxel_length_;
	inline int IndexOf(int x, int y, int z) const
	{
		return z * dims_(0) * dims_(1) + y * dims_(0) + x;
	}
	inline int IndexOf(const Eigen::Vector3i& xyz) const
	{
		return IndexOf(xyz(0), xyz(1), xyz(2));
	}
	float getTSDFAt(const Eigen::Vector3d& p);
	Eigen::Vector3d getNormalAt(const Eigen::Vector3d& p);
};
