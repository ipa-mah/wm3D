#include "wm3D/utility/open3d_helper.hpp"
namespace  Open3DHelper
{
bool open3DMesh2TextureMesh(const open3d::geometry::TriangleMesh& open3d_mesh,TextureMeshPtr& mesh)
{
    mesh->vertices_.resize(open3d_mesh.vertices_.size());
    mesh->vertex_colors_.resize(open3d_mesh.vertex_colors_.size());
    // copy vertex and data
//    for (size_t vidx = 0; vidx < open3d_mesh.vertices_.size(); vidx ++) {

//        mesh.vertices_.push_back(Eigen::Vector3d(vx, vy, vz));
//    }

//    for (size_t vidx = 0; vidx < attrib.colors.size(); vidx += 3) {
//        tinyobj::real_t r = attrib.colors[vidx + 0];
//        tinyobj::real_t g = attrib.colors[vidx + 1];
//        tinyobj::real_t b = attrib.colors[vidx + 2];
//        mesh.vertex_colors_.push_back(Eigen::Vector3d(r, g, b));
//    }
}
bool textureMesh2open3DMesh(const TextureMeshPtr& mesh, open3d::geometry::TriangleMesh& open3d_mesh)
{

}

}



