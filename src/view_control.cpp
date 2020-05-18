
#include <wm3D/view_control.h>
const double FIELD_OF_VIEW_MAX = 90.0;
const double FIELD_OF_VIEW_MIN = 5.0;
ViewControl::~ViewControl(){};
ViewControl::ViewControl(const Eigen::Matrix3d& intrins, const int vertical_views, const int horizontal_views, const double distance) : intrinsic_(intrins)
{
	double vertical_angle_step = M_PI / vertical_views;
	double horizontal_angle_step = 2 * M_PI / horizontal_views;
	for (int j = 0; j < horizontal_views; j++)
		for (int i = 0; i < vertical_views; i++)
		{
			int index = i * horizontal_views + j;

			Eigen::Matrix4d extrinsic = Eigen::Matrix4d::Zero();

			double vertical_rad = vertical_angle_step * (i);
			double horizontal_rad = horizontal_angle_step * (j);
			double z = distance * sin(vertical_rad);
			double oxy = distance * cos(vertical_rad);
			double x = oxy * sin(horizontal_rad);
			double y = oxy * cos(horizontal_rad);

			Eigen::Vector3d eye = Eigen::Vector3d(x, y, z);  // camera position
			for (int v = 0; v < 3; v++)
				if (std::fabs(eye[v]) < 0.00001) eye[v] = 0;

			Eigen::Vector3d look_at(0, 0, 0);								   // original point
			Eigen::Vector3d up_dir(0, 1, 0);								   // y-plane
			Eigen::Vector3d front_dir = (look_at - eye).normalized();		   // z -plane
			Eigen::Vector3d right_dir = up_dir.cross(front_dir).normalized();  // x - plane
			up_dir = front_dir.cross(right_dir).normalized();				   // y -plane
			extrinsic.block<1, 3>(0, 0) = right_dir.transpose();
			extrinsic.block<1, 3>(1, 0) = up_dir.transpose();
			extrinsic.block<1, 3>(2, 0) = front_dir.transpose();
			extrinsic(0, 3) = -right_dir.dot(eye);
			extrinsic(1, 3) = -up_dir.dot(eye);
			extrinsic(2, 3) = -front_dir.dot(eye);
			extrinsic(3, 3) = 1;
			extrinsics_.push_back(extrinsic);
			if (std::isnan(extrinsic.inverse().determinant()))
			{
				std::cout << "View " << index << " is not correct" << std::endl;
				std::cout << "extrinsic : " << std::endl << extrinsic << std::endl;
				continue;
			}
		}
	// Get field of view
	int height = (int)2.0 * (intrinsic_(1, 2) + 0.5);
	int width = (int)2.0 * (intrinsic_(0, 2) + 0.5);

	double tan_half_fov = (double)height / (intrinsic_(1, 1) * 2.0);
	double fov_rad = std::atan(tan_half_fov) * 2.0;
	field_of_view_ = std::max(std::min(fov_rad * 180.0 / M_PI, FIELD_OF_VIEW_MAX), FIELD_OF_VIEW_MIN);
}
