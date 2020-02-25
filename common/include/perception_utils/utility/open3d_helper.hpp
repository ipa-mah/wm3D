#ifndef OPEN3D_HELPER_HPP
#define OPEN3D_HELPER_HPP
#include <unordered_map>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Core>
#include <Open3D/Open3D.h>

#include "perception_utils/utility/utils.hpp"
#include "perception_utils/texture_mesh.h"
namespace  Open3DHelper
{
bool open3DMesh2TextureMesh(const open3d::geometry::TriangleMesh& open3d_mesh,TextureMeshPtr& mesh);



}



#endif // OPEN3D_HELPER_HPP
