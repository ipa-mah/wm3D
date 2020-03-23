#include "wm3D/visualization/render_texture_mesh.hpp"
#include <Open3D/Visualization/Shader/Shader.h>
bool RenderMesh::Compile(){

    if (CompileShaders(vertex_shader_code_.c_str(), NULL,
                       fragment_shader_code_.c_str()) == false) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }

    vertex_position_ = glGetAttribLocation(program_,"vertex_position");
    vertex_uv_ = glGetAttribLocation(program_,"vertex_uv");
    MVP_ = glGetUniformLocation(program_,"MVP");
    texture_ = glGetUniformLocation(program_,"diffuse_texture");
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


bool RenderMesh::CreateVisualizerWindow( const std::string &window_name,
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

    // 1. bind Vertex Array Object
    glGenVertexArrays(1, &vao_id_);
    glBindVertexArray(vao_id_);

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window_, GLFW_STICKY_KEYS, GL_TRUE);
    // Dark blue background
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);


    return true;

}






bool RenderMesh::BindGeometry(const open3d::geometry::Geometry &geometry,
                              const open3d::visualization::RenderOption &option,
                              const open3d::visualization::ViewControl &view)
{
    UnbindGeometry();
    std::vector<Eigen::Vector3f> points;
    std::vector<Eigen::Vector2f> uvs;
    if (prepareBinding(geometry, option, view, points, uvs) == false) {
        PRINT_RED("Binding failed when preparing data.");
        return false;
    }
    // create buffer and bind geometry
    for(int mi = 0; mi<num_materials_ ; mi++)
    {
        //vertex
        glGenBuffers(1,&vertex_position_buffers_[mi]);
        glBindBuffer(GL_ARRAY_BUFFER,vertex_position_buffers_[mi]);
        glBufferData(GL_ARRAY_BUFFER,draw_array_sizes_[mi] * sizeof (Eigen::Vector3f),
                     points.data()+array_offsets_[mi],GL_STATIC_DRAW);

        //uv
        glGenBuffers(1,&vertex_uv_buffers_[mi]);
        glBindBuffer(GL_ARRAY_BUFFER,vertex_uv_buffers_[mi]);
        glBufferData(GL_ARRAY_BUFFER,draw_array_sizes_[mi] * sizeof (Eigen::Vector2f),
                     uvs.data()+array_offsets_[mi],GL_STATIC_DRAW);
    }
    bound_ = true;
    return true;
}

bool RenderMesh::RenderGeometry(const open3d::geometry::Geometry &geometry,
                                const open3d::visualization::RenderOption &option,
                                const open3d::visualization::ViewControl &view)
{

    if(prepareRendering(geometry,option,view) == false)
    {
        PrintShaderWarning("Rendering failed during preparation.");
        return false;
    }

    return true;
}



void RenderMesh::UnbindGeometry()
{
    if (bound_) {
        for (auto buf : vertex_position_buffers_) {
            glDeleteBuffers(1, &buf);
        }
        for (auto buf : vertex_uv_buffers_) {
            glDeleteBuffers(1, &buf);
        }
        for (auto buf : texture_buffers_) {
            glDeleteTextures(1, &buf);
        }

        vertex_position_buffers_.clear();
        vertex_uv_buffers_.clear();
        texture_buffers_.clear();
        draw_array_sizes_.clear();
        array_offsets_.clear();
        num_materials_ = 0;
        bound_ = false;
    }
};

void RenderMesh::Release() {
    UnbindGeometry();
    ReleaseProgram();
}






void RenderTextureMesh::readTextureMesh(const std::shared_ptr<open3d::geometry::TriangleMesh>& texture_mesh)
{
    std::vector<Eigen::Vector3f> points;
    std::vector<Eigen::Vector2f> uvs;
    open3d::visualization::RenderOption option;
    open3d::visualization::ViewControl view;
    BindGeometry(*texture_mesh,option,view);

}

void RenderTextureMesh::rendering(const Eigen::Matrix3d& intrins,const Eigen::Matrix4d& extrinsics)
{




    // Get a handle for our "MVP" uniform
    open3d::visualization::GLHelper::GLMatrix4f projection_matrix;
    open3d::visualization::GLHelper::GLMatrix4f look_at_matrix;
    open3d::visualization::GLHelper::GLMatrix4f view_matrix;


    float z_near = 0.1f;
    float z_far = 10.0f;
    int image_width = static_cast<int>((intrins(0,2)+0.5)*2);
    int image_height = static_cast<int>((intrins(1,2)+0.5)*2);
    projection_matrix.setZero();
    projection_matrix(0,0) = static_cast<float>(intrins(0,0)/intrins(0,2));
    projection_matrix(1,1) = static_cast<float>(intrins(1,1)/intrins(1,2));
    projection_matrix(2,2) = (z_near + z_far) / (z_near - z_far);
    projection_matrix(2,3) = -2.0f * z_far * z_near / (z_far - z_near);
    projection_matrix(3,2) = -1;


    look_at_matrix.setIdentity();
    look_at_matrix(1,1) = -1.0f;
    look_at_matrix(2,2) = -1.0f;

    open3d::visualization::GLHelper::GLMatrix4f gl_trans_scale;
    gl_trans_scale.setIdentity();
    view_matrix = projection_matrix * look_at_matrix *
            gl_trans_scale * extrinsics.cast<GLfloat>();


    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(program_);
    for(int mi=0; mi< num_materials_; mi++)
    {
        glUniformMatrix4fv(MVP_,1,GL_FALSE,view_matrix.data());

        glUniform1i(texture_, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture_buffers_[mi]);

        glEnableVertexAttribArray(vertex_position_);
        glBindBuffer(GL_ARRAY_BUFFER,vertex_position_buffers_[mi]);
        glVertexAttribPointer(vertex_position_,3,GL_FLOAT,GL_FALSE,0,NULL);

        glEnableVertexAttribArray(vertex_uv_);
        glBindBuffer(GL_ARRAY_BUFFER,vertex_uv_buffers_[mi]);
        glVertexAttribPointer(vertex_uv_,2,GL_FLOAT,GL_FALSE,0,NULL);

        //Draw
        glViewport(0, 0, image_width, image_height);
        glDrawArrays(draw_arrays_mode_, 0, draw_array_sizes_[mi]);



        glDisableVertexAttribArray(vertex_position_);
        glDisableVertexAttribArray(vertex_uv_);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    // snippet of code to save color image
    /*
    glFinish();
    //glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    //glReadBuffer(GL_COLOR_ATTACHMENT0 );
    //std::cout<<GBUFFER_TEXTURE_TYPE_COLOR<<std::endl;
    //std::cout<<GL_COLOR_ATTACHMENT0<<std::endl;

    cv::Mat mat(image_height, image_width, CV_8UC3);
    glPixelStorei(GL_PACK_ALIGNMENT, (mat.step & 3) ? 1 : 4);
    glPixelStorei(GL_PACK_ROW_LENGTH, mat.step/mat.elemSize());
    glReadPixels(0, 0, mat.cols, mat.rows, GL_BGR, GL_UNSIGNED_BYTE, mat.data);

    cv::flip(mat, mat, 0);
    cv::imwrite("color.png",mat);
    */




}



bool RenderTextureMesh::prepareRendering(const open3d::geometry::Geometry &geometry,
                                         const open3d::visualization::RenderOption &option,
                                         const open3d::visualization::ViewControl &view)
{

    glEnable(GL_DEPTH_TEST);
    // Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS);


}

bool RenderTextureMesh::prepareBinding(const open3d::geometry::Geometry &geometry,
                                       const open3d::visualization::RenderOption &option,
                                       const open3d::visualization::ViewControl &view,
                                       std::vector<Eigen::Vector3f> &points,
                                       std::vector<Eigen::Vector2f> &uvs)
{

    if(geometry.GetGeometryType() !=  open3d::geometry::Geometry::GeometryType::TriangleMesh &&
            geometry.GetGeometryType() !=
            open3d::geometry::Geometry::GeometryType::HalfEdgeTriangleMesh)
    {
        PRINT_RED("Rendering type is not geometry::TriangleMesh.");
        return false;
    }

    const open3d::geometry::TriangleMesh &mesh = (const open3d::geometry::TriangleMesh &) geometry;
    if(mesh.HasTriangles() == false)
    {
        PRINT_RED("Binding failed with empty triangle mesh.");
        return false;
    }

    std::vector<std::vector<Eigen::Vector3f>> tmp_points;
    std::vector<std::vector<Eigen::Vector2f>> tmp_uvs;
    num_materials_ = (int)mesh.textures_.size();

    array_offsets_.resize(num_materials_);
    draw_array_sizes_.resize(num_materials_);
    vertex_position_buffers_.resize(num_materials_);
    vertex_uv_buffers_.resize(num_materials_);
    texture_buffers_.resize(num_materials_);

    tmp_uvs.resize(num_materials_);
    tmp_points.resize(num_materials_);


    //Bind vertex and uv per material

    for(std::size_t i = 0 ; i < mesh.triangles_.size(); i++) {
        const Eigen::Vector3i& triangle = mesh.triangles_[i];
        int mi = mesh.triangle_material_ids_[i]; // material id

        for(std::size_t j=0; j < 3 ; j++) {
            std::size_t idx = 3 * i +j;
            int vertex_idx = triangle(j);
            tmp_points[mi].push_back(mesh.vertices_[vertex_idx].cast<float>());
            tmp_uvs[mi].push_back(mesh.triangle_uvs_[idx].cast<float>());
        }
    }


    // Bind textures

    for (int mi = 0; mi < num_materials_; mi ++) {
        glGenTextures(1, & texture_buffers_[mi]);
        glBindTexture(GL_TEXTURE_2D,texture_buffers_[mi]);
        GLenum format, type;
        auto it = open3d::visualization::GLHelper::texture_format_map_.find
                (mesh.textures_[mi].num_of_channels_);
        if(it == open3d::visualization::GLHelper::texture_format_map_.end())
        {
            PRINT_RED("Unknown texture format, abort \n");
            return false;
        }

        format = it->second;

        it = open3d::visualization::GLHelper::texture_type_map_.find(mesh.textures_[mi].bytes_per_channel_);
        if(it == open3d::visualization::GLHelper::texture_type_map_.end())
        {
            PRINT_RED("Unknown texture type, abort \n");
            return false;
        }
        type = it->second;
        glTexImage2D(GL_TEXTURE_2D,0,format,mesh.textures_[mi].width_,mesh.textures_[mi].height_,0,
                     format,type,mesh.textures_[mi].data_.data());

        // Set texture clamping method
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    }

    //Point separations

    // Point seperations
    array_offsets_[0] = 0;
    draw_array_sizes_[0] = tmp_points[0].size();
    for (int mi = 1; mi < num_materials_; ++mi) {
        draw_array_sizes_[mi] = tmp_points[mi].size();
        array_offsets_[mi] = array_offsets_[mi - 1] + draw_array_sizes_[mi - 1];
    }

    //prepare chunk of points and uvs
    points.clear();
    uvs.clear();
    for(int mi = 0; mi< num_materials_ ; mi++)
    {
        points.insert(points.end(),tmp_points[mi].begin(),tmp_points[mi].end());
        uvs.insert(uvs.end(),tmp_uvs[mi].begin(),tmp_uvs[mi].end());
    }
    draw_arrays_mode_ = GL_TRIANGLES;

    return true;

}
