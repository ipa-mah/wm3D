#ifndef VISUALIZER_HPP
#define VISUALIZER_HPP

//#include "tiny_obj_loader.h"

#include "perception_utils/texture_mesh.h"

#include <iostream>
#include <unordered_map>
#include <memory>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Core>
#include <pcl/PolygonMesh.h>
// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>

//#include <Open3D/Open3D.h>

class Visualizer
{

public:
    Visualizer();
    using Ptr = std::shared_ptr<Visualizer>;
    using ConstPtr = std::shared_ptr<const Visualizer>;
    virtual ~Visualizer();
    //    Visualizer(Visualizer &&) = delete;
    //    Visualizer(const Visualizer &) = delete;
    //    Visualizer &operator=(const Visualizer &) = delete;



public:

    /// Function to create a window and initialize GLFW
    /// This function MUST be called from the main thread.
    bool createVisualizerWindow(const std::string &window_name = "OpenGLWindow",
                                const int width = 640,
                                const int height = 480,
                                const int left = 50,
                                const int top = 50,
                                const bool visible = true);

    std::function<void(bool)>           on_left_mouse_ = [](bool) {};
    std::function<void(double, double)> on_mouse_scroll_ = [](double, double) {};
    std::function<void(double, double)> on_mouse_move_ = [](double, double) {};
    std::function<void(int)>            on_key_release_ = [](int) {};
    std::function<void(int,int)> frame_size_call_back_ = [](int,int){};

private:
    // window
    GLFWwindow *window_ = NULL;
    std::string window_name_ = "OpenGLWindow";

};












#endif //VISUALIZER_HPP
