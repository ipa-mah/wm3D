#include <Open3D/Open3D.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main( int argc, char** argv )
{
    std::string data_path = "../data1/";

    std::string texture_file = data_path+"textured.obj";
    auto mesh_ptr = std::make_shared<open3d::geometry::TriangleMesh>();
    if (open3d::io::ReadTriangleMesh(texture_file, *mesh_ptr)) {
        open3d::utility::LogInfo("Successfully read {}", texture_file);
    } else {
        open3d::utility::LogWarning("Failed to read {}", texture_file);
        return 1;
    }
    mesh_ptr->ComputeVertexNormals();
    open3d::visualization::DrawGeometries({mesh_ptr}, "Mesh", 1600, 900);

}
