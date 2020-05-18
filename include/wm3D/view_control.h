#include <json/json.h>
#include <pcl/TextureMesh.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/ply_io.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <opencv2/opencv.hpp>

#include <pcl/io/vtk_lib_io.h>
#include <pcl/visualization/cloud_viewer.h>
class ViewControl
{
  public:
	virtual ~ViewControl();
	using Ptr = std::shared_ptr<ViewControl>;
	using ConstPtr = std::shared_ptr<const ViewControl>;
	explicit ViewControl(const Eigen::Matrix3d& intrins, const int vertical_views, const int horizontal_views, const double distance);
	std::vector<Eigen::Matrix4d> getExtrinsics() const
	{
		return extrinsics_;
	};
	double getFieldOfView() const
	{
		return field_of_view_;
	}

  private:
	Eigen::Matrix3d intrinsic_;
	std::vector<Eigen::Matrix4d> extrinsics_;
	double field_of_view_;
};
