#ifndef VISUALIZER_HPP
#define VISUALIZER_HPP


#include "wm3D/texture_mesh.h"
#include "wm3D/shader.h"
#include "wm3D/utility/utils.hpp"
#include "wm3D/gbuffer.h"

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

#include <Open3D/Open3D.h>
#include <Open3D/Visualization/Utility/GLHelper.h>
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

    void close();

    std::function<void(bool)>           on_left_mouse_ = [](bool) {};
    std::function<void(double, double)> on_mouse_scroll_ = [](double, double) {};
    std::function<void(double, double)> on_mouse_move_ = [](double, double) {};
    std::function<void(int)>            on_key_release_ = [](int) {};
    std::function<void(int,int)> frame_size_call_back_ = [](int,int){};
public:



    ////
    /// \brief setViewMatrices set view points which contains OpenGL context and calls OpenGL functions to set view point
    /// \param extrinsic camera view point
    ///
    void setViewMatrices(const Eigen::Matrix4d &extrinsic = Eigen::Matrix4d::Identity());
    bool readTextureMeshAndData(Eigen::Matrix3d& intrins,
                                const std::shared_ptr<open3d::geometry::TriangleMesh>& mesh);

    bool prepareBinding(std::vector<Eigen::Vector3f> &points,
                        std::vector<Eigen::Vector3f> &normals,
                        std::vector<Eigen::Vector2f> &uvs);
    bool bindingMesh();
    bool render3DModel(const std::string& vert_file, const std::string& frag_file,
                       const std::vector<Eigen::Matrix4d>& extrinsics);
    bool mesh2RGBDImage();
protected:
    //3D model
    std::shared_ptr<open3d::geometry::TriangleMesh> mesh_;

    // window
    GLFWwindow *window_ = NULL;
    std::string window_name_ = "OpenGLWindow";
    int image_width_;
    int image_height_;



    float z_near_;
    float z_far_;
    float constant_z_near_ = -1;
    float constant_z_far_ = -1;
    // unsigned int VAO_, VBO_, EBO_;
    GLuint element_buffer_object_; //render triangle
    GLuint vao_id_;


    Eigen::Matrix<GLfloat, 4, 4, Eigen::ColMajor> projection_matrix_;
    Eigen::Matrix<GLfloat, 4, 4, Eigen::ColMajor> look_at_matrix_;
    Eigen::Matrix<GLfloat, 4, 4, Eigen::ColMajor> view_matrix_;

    int num_materials_;
    std::vector<int> array_offsets_;
    std::vector<GLsizei> draw_array_sizes_;


    GLuint vertex_position_;
    GLuint vertex_uv_;
    GLuint vertex_normal_;
    std::vector<GLuint> vertex_position_buffers_;  //render vertex
    std::vector<GLuint> vertex_uv_buffers_; //render uv
    std::vector<GLuint> vertex_normal_buffers_; //render normal
    std::vector<GLuint> diffuse_texture_buffers_;
    GLenum draw_arrays_mode_;
};












#endif //VISUALIZER_HPP
