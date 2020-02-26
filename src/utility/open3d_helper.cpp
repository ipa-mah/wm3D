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

bool open3DMesh2PCLMesh(const open3d::geometry::TriangleMesh& open3d_mesh,pcl::PolygonMesh& pcl_mesh)
{
    assert(open3d_mesh.HasVertices() == true && "input mesh has no vertices");

    pcl::PointCloud<pcl::PointNormal> cloud;
    for(std::size_t face_id= 0; face_id< open3d_mesh.triangles_.size(); face_id++)
    {
        const Eigen::Vector3i& vix = open3d_mesh.triangles_[face_id];
        pcl::Vertices v;
        v.vertices.push_back(vix[0]);
        v.vertices.push_back(vix[1]);
        v.vertices.push_back(vix[2]);
        pcl_mesh.polygons.push_back(v);
    }


    for(std::size_t i = 0; i< open3d_mesh.vertices_.size(); i++)
    {
       pcl::PointNormal pt;
       pt.x = open3d_mesh.vertices_[i].cast<float>()[0];
       pt.y = open3d_mesh.vertices_[i].cast<float>()[1];
       pt.z = open3d_mesh.vertices_[i].cast<float>()[2];
       pt.normal_x = open3d_mesh.vertex_normals_[i].cast<float>()[0];
       pt.normal_y = open3d_mesh.vertex_normals_[i].cast<float>()[1];
       pt.normal_z = open3d_mesh.vertex_normals_[i].cast<float>()[2];
//       pt.r = open3d_mesh.vertex_colors_[i].cast<uint8_t>()[0];
//       pt.g = open3d_mesh.vertex_colors_[i].cast<uint8_t>()[1];
//       pt.b = open3d_mesh.vertex_colors_[i].cast<uint8_t>()[2];
       cloud.push_back(pt);
    }
    pcl::toPCLPointCloud2(cloud,pcl_mesh.cloud);

    return true;
}

}

