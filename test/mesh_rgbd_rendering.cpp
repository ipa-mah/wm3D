#include <iostream>
#include <json/json.h>
// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>
#include <eigen3/Eigen/Core>

#include <pcl/io/ply_io.h>
#include <pcl/io/obj_io.h>
#include "wm3D/visualization/visualizer.hpp"
#include "wm3D/utility/vision_utils.hpp"
#include "wm3D/utility/open3d_helper.hpp"
#include "wm3D/visualization/render_texture_mesh.hpp"
bool readDataFromJsonFile(const std::string& file_name, std::vector<Eigen::Matrix4d>& extrinsics,
                          Eigen::Matrix3d& intrins)
{


    intrins.setIdentity();


    Json::Value root;
    std::ifstream config_doc(file_name, std::ifstream::binary);
    if(!config_doc.is_open())
    {
        std::cout<<"No config.json file in the data path"<<std::endl;
        return false;
    }
    config_doc >> root;

    intrins(0,0) = root["camera_matrix"].get("focal_x",500).asDouble();
    intrins(1,1) = root["camera_matrix"].get("focal_y",500).asDouble();
    intrins(0,2) = root["camera_matrix"].get("c_x",320).asDouble();
    intrins(1,2) = root["camera_matrix"].get("c_y",240).asDouble();
    for(const auto& node : root["views"])
    {
        Eigen::Affine3d cam2world;
        cam2world.setIdentity();
        Eigen::Matrix3d rot;
        cam2world.translation() = Eigen::Vector3d(node["translation"][0].asDouble(),node["translation"][1].asDouble(),node["translation"][2].asDouble());
        for(int i=0;i<node["rotation"].size();i++)
        {
            int r = i / 3 ;
            int c = i % 3 ;
            rot(r,c) = node["rotation"][i].asDouble();
        }
        cam2world.rotate(rot);
        extrinsics.push_back(cam2world.matrix().inverse());
    }

    std::cout<<"Artificial Intrinsics:"<<std::endl<<intrins<<std::endl;
}
int main( int argc, char** argv )
{

    std::string data_path = "../sample_data/";
    std::string texture_file = data_path+"texture_model.obj";
    std::string vert_shader = "/home/ipa-mah/1_projects/wm3D/src/shaders/TextureSimpleVertexShader.glsl";
    std::string frag_shader = "/home/ipa-mah/1_projects/wm3D/src/shaders/TextureSimpleFragmentShader.glsl";

    std::vector<Eigen::Matrix4d> extrinsics;
    Eigen::Matrix3d intrins;

    readDataFromJsonFile(data_path+"config.json",extrinsics,intrins);
    int width = (intrins(0,2)+0.5)*2;
    int height = (intrins(1,2)+0.5)*2;

    std::shared_ptr<open3d::geometry::TriangleMesh> mesh = std::make_shared<open3d::geometry::TriangleMesh>();
    open3d::io::ReadTriangleMesh(data_path+"texture_model.obj",*mesh);

    std::shared_ptr<RenderTextureMesh> render =
            std::make_shared<RenderTextureMesh>("render_texture_mesh",vert_shader,frag_shader);
    render->CreateVisualizerWindow("wm3D",width,height,50,50,true);
    render->compileShaders();


    render->readTextureMesh(mesh);
    render->rendering(intrins,extrinsics[0]);
    return 0;
}

