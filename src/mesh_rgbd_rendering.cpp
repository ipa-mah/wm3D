#include <iostream>

// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>
#include "perception_utils/visualizer.hpp"
#include "perception_utils/vision_utils.hpp"
#include <eigen3/Eigen/Core>

int main( int argc, char** argv )
{

    std::string data_path = "../data1/";
    std::string texture_file = data_path+"textured.obj";
    Eigen::Vector2d value(0,0);

    Visualizer::Ptr visual = Visualizer::Ptr(new Visualizer);
    TextureMeshPtr texture_mesh;
    VisionUtils::readOBJFromFile(texture_file,texture_mesh);

    return 0;
}

