#ifndef OPEN3D_HELPER_HPP
#define OPEN3D_HELPER_HPP
#include <unordered_map>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Core>
#include <Open3D/Open3D.h>
#include <pcl/PolygonMesh.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include "wm3D/utility/utils.hpp"
#include "wm3D/texture_mesh.h"
namespace  Open3DHelper
{
bool open3DMesh2TextureMesh(const open3d::geometry::TriangleMesh& open3d_mesh,TextureMeshPtr& mesh);
bool open3DMesh2PCLMesh(const open3d::geometry::TriangleMesh& open3d_mesh,pcl::PolygonMesh& pcl_mesh);


}



#endif // OPEN3D_HELPER_HPP
