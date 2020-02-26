#include <iostream>
#include <json/json.h>
// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>
#include "wm3D/visualizer.hpp"
#include "wm3D/utility/vision_utils.hpp"
#include <eigen3/Eigen/Core>

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

    std::cout<<"Virtual Intrinsics:"<<std::endl<<intrins<<std::endl;
}

int main( int argc, char** argv )
{

    std::string data_path = "../sample_data/";
    std::string texture_file = data_path+"texture_model.obj";
    std::string vert_shader = "/home/ipa-mah/1_projects/wm3D/shaders/rendermode.vert";
    std::string frag_shader = "/home/ipa-mah/1_projects/wm3D/shaders/rendermode.frag";

    std::vector<Eigen::Matrix4d> extrinsics;
    Eigen::Matrix3d intrins;

    readDataFromJsonFile(data_path+"config.json",extrinsics,intrins);
    Visualizer::Ptr visual = Visualizer::Ptr(new Visualizer);
    std::shared_ptr<open3d::geometry::TriangleMesh> mesh_ptr = std::make_shared<open3d::geometry::TriangleMesh>();
    if (open3d::io::ReadTriangleMesh(texture_file, *mesh_ptr)) {
        open3d::utility::LogInfo("Successfully read {}", texture_file);
    } else {
        open3d::utility::LogWarning("Failed to read {}", texture_file);
        return 1;
    }
    mesh_ptr->ComputeVertexNormals();
    visual->readTextureMeshAndData(intrins,mesh_ptr);

    visual->createVisualizerWindow("rendering");
    visual->bindingMesh();
    visual->render3DModel(vert_shader,frag_shader,extrinsics);
    return 0;
}

