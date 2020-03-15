#ifndef VISUALIZER_HPP
#define VISUALIZER_HPP


#include "wm3D/texture_mesh.h"
#include "wm3D/visualization/shader.h"
#include "wm3D/utility/utils.hpp"
#include "wm3D/visualization/gbuffer.h"

#include <iostream>
#include <unordered_map>
#include <memory>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Core>
#include <pcl/PolygonMesh.h>
#include <pcl/conversions.h>
// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>

#include <Open3D/Open3D.h>
#include <Open3D/Visualization/Utility/GLHelper.h>
#include <Open3D/Visualization/Shader/ShaderWrapper.h>

class Visualizer : public open3d::visualization::glsl::ShaderWrapper
{

public:
    Visualizer(const std::string& name): ShaderWrapper(name)
    {
        Compile();
    };
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

    bool renderPCLMesh(const pcl::PolygonMesh& mesh,const Eigen::Matrix3d& intrins,
                       const Eigen::Matrix4d& extrins);


public:
    bool render(const std::shared_ptr<open3d::geometry::TriangleMesh>& mesh);
    bool validateShader(GLuint shader_index);
    bool validateProgram(GLuint program_index);
    void printShaderWarning(const std::string& mes);
    bool Compile() final;


    void releasePrograme();
public:
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

    // perspective transformation
    Eigen::Matrix<GLfloat, 4, 4, Eigen::ColMajor> projection_matrix_;

    Eigen::Matrix<GLfloat, 4, 4, Eigen::ColMajor> look_at_matrix_;

    Eigen::Matrix<GLfloat, 4, 4, Eigen::ColMajor> view_matrix_;

    int num_materials_;
    std::vector<int> array_offsets_;
    std::vector<GLsizei> draw_array_sizes_;


    GLuint vertex_position_; // vertex position name in glsl file
    GLuint vertex_uv_;  // vertex uv nam in glsl file
    GLuint vertex_normal_;  // vertex normal in glsl file



    std::vector<GLuint> vertex_position_buffers_;  //render vertex
    std::vector<GLuint> vertex_uv_buffers_; //render uv
    std::vector<GLuint> vertex_normal_buffers_; //render normal
    std::vector<GLuint> diffuse_texture_buffers_;

public:
    GLuint vertex_shader_;
    GLuint geometry_shader_;
    GLuint fragment_shader_;
    GLuint program_;
    GLenum draw_arrays_mode_ = GL_POINTS;
    GLsizei draw_arrays_size_ = 0;
    bool compiled_ = false;
    bool bound_ = false;

    void SetShaderName(const std::string &shader_name) {
        shader_name_ = shader_name;
    }

private:
    std::string shader_name_ = "ShaderWrapper";


protected:
    bool ValidateShader(GLuint shader_index);
    bool ValidateProgram(GLuint program_index);
    bool CompileShaders(const char *const vertex_shader_code,
                        const char *const geometry_shader_code,
                        const char *const fragment_shader_code);
    void ReleaseProgram();
};












#endif //VISUALIZER_HPP
