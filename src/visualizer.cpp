#include "wm3D/visualizer.hpp"


std::function<void(int,int)> frame_size_call_back_ = [](int,int){};

void processInput(GLFWwindow *window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}
Visualizer::Visualizer() {
    image_width_ = 0;
    image_height_ = 0;

}
Visualizer::~Visualizer()
{
    glfwDestroyWindow(window_);
    glfwTerminate();
}
void Visualizer::close()
{
    glfwSetWindowShouldClose(window_,1);
}
bool Visualizer::createVisualizerWindow(const std::string &window_name,
                                        const int width ,
                                        const int height,
                                        const int left,
                                        const int top,
                                        const bool visible)
{
    if (image_width_ <= 0 || image_height_ <= 0) {
        PRINT_RED("[Visualizer] createVisualizerWindow() failed because window height and width are not set.");
        return false;
    }

    window_name_ = window_name;
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, visible ? 1 : 0); // hide window after creation

    window_ = glfwCreateWindow(image_width_,image_height_,window_name_.c_str(),NULL,NULL);
    if (!window_) {
        std::cerr<<"Failed to create window\n"<<std::endl;
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window_);
    glfwSetWindowUserPointer(window_,this);

    //init GLEW
    glewExperimental = true;  // Needed for core profile
    if (glewInit() != GLEW_OK)
    {
        fprintf(stderr, "Failed to initialize GLEW\n");
        getchar();
        glfwTerminate();
        return false;
    }

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window_, GLFW_STICKY_KEYS, GL_TRUE);
    // black color background
    glClearColor(0.0f, 0.0f, 0.f, 0.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);



    /*
    glfwSetMouseButtonCallback(window_, [](GLFWwindow * w, int button, int action, int mods)
    {
        auto s = (Visualizer*)glfwGetWindowUserPointer(w);
        if (button == 0) s->on_left_mouse_(action == GLFW_PRESS);
    });

    glfwSetScrollCallback(window_, [](GLFWwindow * w, double xoffset, double yoffset)
    {
        auto s = (Visualizer*)glfwGetWindowUserPointer(w);
        s->on_mouse_scroll_(xoffset, yoffset);
    });

    glfwSetCursorPosCallback(window_, [](GLFWwindow * w, double x, double y)
    {
        auto s = (Visualizer*)glfwGetWindowUserPointer(w);
        s->on_mouse_move_(x, y);
    });

    glfwSetKeyCallback(window_, [](GLFWwindow * w, int key, int scancode, int action, int mods)
    {
        auto s = (Visualizer*)glfwGetWindowUserPointer(w);
        if (0 == action) // on key release
        {
            s->on_key_release_(key);
        }
    });
    */

    return true;
}

void Visualizer::setViewMatrices(const Eigen::Matrix4d &extrinsic)
{
    if (image_width_ <= 0 || image_height_ <= 0) {
        PRINT_RED("[Visualizer] setViewMatrices() failed because window height and width are not set.");
        return;
    }

    //Only show perspective projection
    view_matrix_ = projection_matrix_ * look_at_matrix_ * extrinsic.cast<GLfloat>();

}

bool Visualizer::prepareBinding(std::vector<Eigen::Vector3f> &points,
                                std::vector<Eigen::Vector3f> &normals,
                                std::vector<Eigen::Vector2f> &uvs)
{


    if (mesh_->HasTriangles() == false) {
        PRINT_RED("Binding failed with empty triangle mesh.");
        return false;
    }
    if (mesh_->HasTriangleNormals() == false ||
            mesh_->HasVertexNormals() == false) {
        PRINT_RED("Binding failed because mesh has no normals.");
        PRINT_RED("Call ComputeVertexNormals() before binding.");
        return false;
    }

    std::vector<std::vector<Eigen::Vector3f>> tmp_points;
    std::vector<std::vector<Eigen::Vector3f>> tmp_normals;
    std::vector<std::vector<Eigen::Vector2f>> tmp_uvs;

    num_materials_ = (int)mesh_->textures_.size();

    array_offsets_.resize(num_materials_);
    draw_array_sizes_.resize(num_materials_);
    vertex_position_buffers_.resize(num_materials_);
    vertex_normal_buffers_.resize(num_materials_);
    vertex_uv_buffers_.resize(num_materials_);
    diffuse_texture_buffers_.resize(num_materials_);

    tmp_points.resize(num_materials_);
    tmp_normals.resize(num_materials_);
    tmp_uvs.resize(num_materials_);

    for(size_t i=0; i< mesh_->triangles_.size(); i++)
    {
        const auto& triangle = mesh_->triangles_[i];
        int mi = mesh_->triangle_material_ids_[i];
        for(int j=0; j<3; j++)
        {
            size_t idx = i*3+j; // index of uv
            size_t vi = triangle(j); // index of vertex
            tmp_points[mi].push_back(mesh_->vertices_[vi].cast<float>());
            tmp_uvs[mi].push_back(mesh_->triangle_uvs_[idx].cast<float>());
            tmp_normals[mi].push_back(mesh_->triangle_normals_[i].cast<float>());
        }
    }
    //Bind textures

    for(int mi=0; mi<num_materials_; mi++)
    {
        glGenTextures(1,&diffuse_texture_buffers_[mi]);
        glBindTexture(GL_TEXTURE_2D,diffuse_texture_buffers_[mi]);


        GLenum format, type;
        auto it = open3d::visualization::GLHelper::texture_format_map_.find(
                    mesh_->textures_[mi].num_of_channels_);
        if (it == open3d::visualization::GLHelper::texture_format_map_.end()) {
            PRINT_RED("Unknown texture format, abort!");
            return false;
        }
        format = it->second;

        it = open3d::visualization::GLHelper::texture_type_map_.find(
                    mesh_->textures_[mi].bytes_per_channel_);
        if (it == open3d::visualization::GLHelper::texture_type_map_.end()) {
            PRINT_RED("Unknown texture type, abort!");
            return false;
        }
        type = it->second;
        // NOTE: this will generate texture on GPU. It may crash on a GPU card with insufficient memory if
        // using a super large texture image.
        glTexImage2D(GL_TEXTURE_2D, 0, format, mesh_->textures_[mi].width_,
                     mesh_->textures_[mi].height_, 0, format, type,
                     mesh_->textures_[mi].data_.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    // Point seperations
    array_offsets_[0] = 0;
    draw_array_sizes_[0] = tmp_points[0].size();
    for (int mi = 1; mi < num_materials_; ++mi) {
        draw_array_sizes_[mi] = tmp_points[mi].size();
        array_offsets_[mi] = array_offsets_[mi - 1] + draw_array_sizes_[mi - 1];
    }
    // Prepared chunk of points and uvs
    points.clear();
    uvs.clear();
    normals.clear();
    for (int mi = 0; mi < num_materials_; ++mi) {
        points.insert(points.end(), tmp_points[mi].begin(),
                      tmp_points[mi].end());
        uvs.insert(uvs.end(), tmp_uvs[mi].begin(), tmp_uvs[mi].end());
        normals.insert(normals.end(), tmp_normals[mi].begin(),
                       tmp_normals[mi].end());
    }

    draw_arrays_mode_ = GL_TRIANGLES;
    return true;

}

bool Visualizer::readTextureMeshAndData(Eigen::Matrix3d& intrins,
                                        const std::shared_ptr<open3d::geometry::TriangleMesh>& mesh)
{

    mesh_ = mesh;
    image_width_ = static_cast<int>((intrins(0,2)+0.5)*2);
    image_height_ = static_cast<int>((intrins(1,2)+0.5)*2);

    z_near_ = 0.1f;
    z_far_ = 10.0f;

    projection_matrix_.setZero();
    projection_matrix_(0,0) = static_cast<float>(intrins(0,0)/intrins(0,2));
    projection_matrix_(1,1) = static_cast<float>(intrins(1,1)/intrins(1,2));
    projection_matrix_(2,2) = (z_near_ + z_far_) / (z_near_ - z_far_);
    projection_matrix_(3,2) = -1;
    projection_matrix_(2,3) = -2.0f * z_far_ * z_near_ / (z_far_ - z_near_);

    look_at_matrix_.setIdentity();
    look_at_matrix_(2,2) = -1.0f;
    look_at_matrix_(3,3) = -1.0f;

    view_matrix_.setIdentity();


}
bool Visualizer::bindingMesh()
{
    //prepare binding
    // Prepare data to be passed to GPU
    // Since we have multiple texture files, we need to create 2D vector array
    std::vector<Eigen::Vector3f> points;
    std::vector<Eigen::Vector3f> normals;
    std::vector<Eigen::Vector2f> uvs;
    prepareBinding(points,normals,uvs);

    // Create buffers and bind the geometry
    for (int mi=0;mi < num_materials_; mi++) {
        glGenBuffers(1,&vertex_position_buffers_[mi]);
        glBindBuffer(GL_ARRAY_BUFFER,vertex_position_buffers_[mi]);
        glBufferData(GL_ARRAY_BUFFER,draw_array_sizes_[mi]* sizeof(Eigen::Vector3f),
                     points.data()+array_offsets_[mi],GL_STATIC_DRAW);

        glGenBuffers(1, &vertex_normal_buffers_[mi]);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_normal_buffers_[mi]);
        glBufferData(GL_ARRAY_BUFFER,
                     draw_array_sizes_[mi] * sizeof(Eigen::Vector3f),
                     normals.data() + array_offsets_[mi], GL_STATIC_DRAW);

        glGenBuffers(1, &vertex_uv_buffers_[mi]);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_uv_buffers_[mi]);
        glBufferData(GL_ARRAY_BUFFER,
                     draw_array_sizes_[mi] * sizeof(Eigen::Vector2f),
                     uvs.data() + array_offsets_[mi], GL_STATIC_DRAW);

    }

    glGenerateMipmap(GL_TEXTURE_2D);
    glBindVertexArray(0);



    return true;
}


bool Visualizer::render3DModel(const std::string& vert_file, const std::string& frag_file,
                               const std::vector<Eigen::Matrix4d>& extrinsics)
{
    // // Set attributes for vertices
    // vertex positions

//    for (int mi=0; mi < num_materials_;mi++) {
//         glEnableVertexAttribArray(vert_file);
//        glVertexAttribPointer()
//    }


    Shader::Ptr shader = std::make_shared<Shader>();
    shader->LoadShaders(vert_file.c_str(),frag_file.c_str());
    shader->setInt("save_texture",0);

    for(size_t i=0; i< extrinsics.size();i++)
    {
        glClear(GL_COLOR_BUFFER_BIT||GL_DEPTH_BUFFER_BIT);
        setViewMatrices(extrinsics[i]);
        shader->useProgram();
        shader->setFloat("near",z_near_);
        shader->setFloat("far",z_far_);
        shader->setBool("flag_show_color", false);
        shader->setBool("flag_show_texture", true);
        setViewMatrices(extrinsics[i]);
        shader->setMat4("transform",view_matrix_);
        for (int mi=0; mi < num_materials_;mi++) {

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D,diffuse_texture_buffers_[mi]);

        }



    }

}


