#include "wm3D/visualization/render_texture_mesh.hpp"
#include <Open3D/Visualization/Shader/Shader.h>
bool RenderMesh::Compile(){

    if (CompileShaders(vertex_shader_code_.c_str(), NULL,
                       fragment_shader_code_.c_str()) == false) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }

    return true;
}

bool RenderMesh::loadShaders()
{
    std::ifstream vertex_shader_stream(render_texture_mesh_vertex_shader_file_, std::ios::in);
    if(vertex_shader_stream.is_open())
    {
        std::string line = "";
        while (std::getline(vertex_shader_stream, line))
            vertex_shader_code_ += "\n" + line;
        vertex_shader_stream.close();
    }
    else
    {
        PRINT_RED("Impossible to open %s. Are you in the right directory ? !\n"
                  , render_texture_mesh_vertex_shader_file_.c_str());
        getchar();
        return false;
    }


    std::ifstream fragment_shader_stream(render_texture_mesh_fragment_shader_file_, std::ios::in);
    if(fragment_shader_stream.is_open())
    {
        std::string line = "";
        while (std::getline(fragment_shader_stream, line))
            fragment_shader_code_ += "\n" + line;
        fragment_shader_stream.close();
    }
    else
    {
        PRINT_RED("Impossible to open %s. Are you in the right directory ? !\n"
                  , render_texture_mesh_fragment_shader_file_.c_str());
        getchar();
        return false;
    }

    return true;
}


bool RenderMesh::CreateVisualizerWindow(
                                                const std::string &window_name,
                                                const int width, const int height,
                                                const int left, const int top, const bool visible)
{

    window_name_ = window_name;
    if (window_) {  // window already created
        glfwSetWindowPos(window_, left, top);
        glfwSetWindowSize(window_, width, height);
        return true;
    }
    if (!glfwInit()) {
        PRINT_RED("RenderMesh::CreateVisualizerWindow::Failed to initialize GLFW");
        glfwTerminate();
        return false;
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
#ifndef HEADLESS_RENDERING
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, visible ? 1 : 0);

    window_ = glfwCreateWindow(width, height, window_name_.c_str(), NULL, NULL);
    if (!window_) {
        PRINT_RED("RenderMesh::CreateVisualizerWindow::Failed to create window");
        glfwTerminate();
        return false;
    }
    // Initialize GLEW
    glfwMakeContextCurrent(window_);
    glewExperimental = GL_TRUE; // Needed for core profile
    if (glewInit() != GLEW_OK) {
        PRINT_RED("RenderMesh::CreateVisualizerWindow::Failed to initialize GLEW");
        glfwTerminate();
        return false;
    }

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window_, GLFW_STICKY_KEYS, GL_TRUE);
    // Dark blue background
    glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
    // Enable depth test
    glEnable(GL_DEPTH_TEST);
    // Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS);


    return true;

}




bool RenderMesh::BindGeometry(const open3d::geometry::Geometry &geometry,
                                     const open3d::visualization::RenderOption &option,
                                     const open3d::visualization::ViewControl &view)
{

}

bool RenderMesh::RenderGeometry(const open3d::geometry::Geometry &geometry,
                                       const open3d::visualization::RenderOption &option,
                                       const open3d::visualization::ViewControl &view)
{

};
void RenderMesh::UnbindGeometry()
{

};

void RenderMesh::Release() {
    UnbindGeometry();
    ReleaseProgram();
}


bool RenderTextureMesh::prepareRendering(const open3d::geometry::Geometry &geometry,
                      const open3d::visualization::RenderOption &option,
                      const open3d::visualization::ViewControl &view)
{

}

bool RenderTextureMesh::prepareBinding(const open3d::geometry::Geometry &geometry,
                    const open3d::visualization::RenderOption &option,
                    const open3d::visualization::ViewControl &view,
                    std::vector<Eigen::Vector3f> &points,
                    std::vector<Eigen::Vector2f> &uvs)
{

}
