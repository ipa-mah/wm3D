#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <json/json.h>
#include <Open3D/Open3D.h>
struct Vertex
{
    std::vector<Eigen::Vector3f> points;
    std::vector<Eigen::Vector3f> uvs;
};

enum GBUFFER_TEXTURE_TYPE {
    GBUFFER_TEXTURE_TYPE_DEPTH,
    GBUFFER_TEXTURE_TYPE_COLOR,
    GBUFFER_NUM_TEXTURES
};
GLFWwindow* window;
GLuint LoadShaders(const char * vertex_file_path,const char * fragment_file_path){

    // Create the shaders
    GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
    GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

    // Read the Vertex Shader code from the file
    std::string VertexShaderCode;
    std::ifstream VertexShaderStream(vertex_file_path, std::ios::in);
    if(VertexShaderStream.is_open()){
        std::stringstream sstr;
        sstr << VertexShaderStream.rdbuf();
        VertexShaderCode = sstr.str();
        VertexShaderStream.close();
    }else{
        printf("Impossible to open %s. Are you in the right directory ? Don't forget to read the FAQ !\n", vertex_file_path);
        getchar();
        return 0;
    }

    // Read the Fragment Shader code from the file
    std::string FragmentShaderCode;
    std::ifstream FragmentShaderStream(fragment_file_path, std::ios::in);
    if(FragmentShaderStream.is_open()){
        std::stringstream sstr;
        sstr << FragmentShaderStream.rdbuf();
        FragmentShaderCode = sstr.str();
        FragmentShaderStream.close();
    }

    GLint Result = GL_FALSE;
    int InfoLogLength;


    // Compile Vertex Shader
    printf("Compiling shader : %s\n", vertex_file_path);
    char const * VertexSourcePointer = VertexShaderCode.c_str();
    glShaderSource(VertexShaderID, 1, &VertexSourcePointer , NULL);
    glCompileShader(VertexShaderID);

    // Check Vertex Shader
    glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if ( InfoLogLength > 0 ){
        std::vector<char> VertexShaderErrorMessage(InfoLogLength+1);
        glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
        printf("%s\n", &VertexShaderErrorMessage[0]);
    }



    // Compile Fragment Shader
    printf("Compiling shader : %s\n", fragment_file_path);
    char const * FragmentSourcePointer = FragmentShaderCode.c_str();
    glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer , NULL);
    glCompileShader(FragmentShaderID);

    // Check Fragment Shader
    glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if ( InfoLogLength > 0 ){
        std::vector<char> FragmentShaderErrorMessage(InfoLogLength+1);
        glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
        printf("%s\n", &FragmentShaderErrorMessage[0]);
    }



    // Link the program
    printf("Linking program\n");
    GLuint ProgramID = glCreateProgram();
    glAttachShader(ProgramID, VertexShaderID);
    glAttachShader(ProgramID, FragmentShaderID);
    glLinkProgram(ProgramID);

    // Check the program
    glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
    glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if ( InfoLogLength > 0 ){
        std::vector<char> ProgramErrorMessage(InfoLogLength+1);
        glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
        printf("%s\n", &ProgramErrorMessage[0]);
    }


    glDetachShader(ProgramID, VertexShaderID);
    glDetachShader(ProgramID, FragmentShaderID);

    glDeleteShader(VertexShaderID);
    glDeleteShader(FragmentShaderID);

    return ProgramID;
}

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
        cam2world.translation() =
                Eigen::Vector3d(node["translation"][0].asDouble(),
                node["translation"][1].asDouble(),node["translation"][2].asDouble());
        for(int i=0;i<node["rotation"].size();i++)
        {
            int r = i / 3 ;
            int c = i % 3 ;
            rot(r,c) = node["rotation"][i].asDouble();
        }
        cam2world.rotate(rot);
        //        std::cout<<cam2world.matrix()<<std::endl;
        //        std::cout<<cam2world.matrix().inverse()<<std::endl;

        extrinsics.push_back(cam2world.matrix().inverse());
    }

    std::cout<<"Artificial Intrinsics:"<<std::endl<<intrins<<std::endl;
}

bool createVisualizerWindow(
        const std::string &window_name = "wm3D",
        const int width = 640,
        const int height = 480,
        const int left = 50,
        const int top = 50,
        const bool visible = true
        )
{

    if (window) {  // window already created
        glfwSetWindowPos(window, left, top);
        glfwSetWindowSize(window, width, height);
        return true;
    }

    if (!glfwInit()) {
        printf("RenderMesh::CreateVisualizerWindow::Failed to initialize GLFW");
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

    window = glfwCreateWindow(width, height, window_name.c_str(), NULL, NULL);
    if (!window) {
        printf("RenderMesh::CreateVisualizerWindow::Failed to create window");
        glfwTerminate();
        return false;
    }

    // Initialize GLEW
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE; // Needed for core profile
    if (glewInit() != GLEW_OK) {
        printf("RenderMesh::CreateVisualizerWindow::Failed to initialize GLEW");
        glfwTerminate();
        return false;
    }

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
    // Dark blue background
    glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
    // Enable depth test
    glEnable(GL_DEPTH_TEST);
    // Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS);


    return true;



}


bool loadShaders(GLuint& programID_,
                 std::string& render_texture_mesh_vertex_shader_file_,
                 std::string& render_texture_mesh_fragment_shader_file_)
{


    // Create the shaders
    GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);

    GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

    // Read the Vertex Shader code from the file
    std::string VertexShaderCode;
    std::ifstream VertexShaderStream(render_texture_mesh_vertex_shader_file_.c_str(), std::ios::in);
    if (VertexShaderStream.is_open())
    {
        std::string Line = "";
        while (getline(VertexShaderStream, Line))
            VertexShaderCode += "\n" + Line;
        VertexShaderStream.close();
    }
    else
    {
        printf(
                    "Impossible to open %s. Are you in the right directory ? Don't forget to read the FAQ !\n",
                    render_texture_mesh_vertex_shader_file_.c_str());
        getchar();
        return false;
    }

    // Read the Fragment Shader code from the file
    std::string FragmentShaderCode;
    std::ifstream FragmentShaderStream(render_texture_mesh_fragment_shader_file_.c_str(), std::ios::in);
    if (FragmentShaderStream.is_open())
    {
        std::string Line = "";
        while (getline(FragmentShaderStream, Line))
            FragmentShaderCode += "\n" + Line;
        FragmentShaderStream.close();
    }

    GLint Result = GL_FALSE;
    int InfoLogLength;

    // Compile Vertex Shader
    printf("Compiling shader : %s\n", render_texture_mesh_vertex_shader_file_.c_str());
    char const *VertexSourcePointer = VertexShaderCode.c_str();
    glShaderSource(VertexShaderID, 1, &VertexSourcePointer, NULL);
    glCompileShader(VertexShaderID);

    // Check Vertex Shader
    glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if (InfoLogLength > 0)
    {
        std::vector<char> VertexShaderErrorMessage(InfoLogLength + 1);
        glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
        printf("%s\n", &VertexShaderErrorMessage[0]);
    }

    // Compile Fragment Shader
    printf("Compiling shader : %s\n", render_texture_mesh_fragment_shader_file_.c_str());
    char const *FragmentSourcePointer = FragmentShaderCode.c_str();
    glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer, NULL);
    glCompileShader(FragmentShaderID);

    // Check Fragment Shader
    glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if (InfoLogLength > 0)
    {
        std::vector<char> FragmentShaderErrorMessage(InfoLogLength + 1);
        glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
        printf("%s\n", &FragmentShaderErrorMessage[0]);
    }

    // Link the program
    printf("Linking program ... \n");
    programID_ = glCreateProgram();
    glAttachShader(programID_, VertexShaderID);
    glAttachShader(programID_, FragmentShaderID);
    glLinkProgram(programID_);

    // Check the program
    glGetProgramiv(programID_, GL_LINK_STATUS, &Result);
    glGetProgramiv(programID_, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if (InfoLogLength > 0)
    {
        std::vector<char> ProgramErrorMessage(InfoLogLength + 1);
        glGetProgramInfoLog(programID_, InfoLogLength, NULL, &ProgramErrorMessage[0]);
        printf("%s\n", &ProgramErrorMessage[0]);
    }

    glDetachShader(programID_, VertexShaderID);
    glDetachShader(programID_, FragmentShaderID);

    glDeleteShader(VertexShaderID);
    glDeleteShader(FragmentShaderID);
    return true;
}


int main( void )
{


    std::string data_path = "../sample_data/";
    std::string texture_file = data_path+"texture_model.obj";
    std::string vert_shader = "/home/ipa-mah/1_projects/wm3D/src/shaders/TextureSimpleVertexShader.glsl";
    std::string frag_shader = "/home/ipa-mah/1_projects/wm3D/src/shaders/TextureSimpleFragmentShader.glsl";

    std::vector<Eigen::Matrix4d> extrinsics;
    Eigen::Matrix3d intrins;

    readDataFromJsonFile(data_path+"config.json",extrinsics,intrins);


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
            gl_trans_scale * extrinsics[4].cast<GLfloat>();


    std::shared_ptr<open3d::geometry::TriangleMesh> mesh = std::make_shared<open3d::geometry::TriangleMesh>();
    open3d::io::ReadTriangleMesh(data_path+"texture_model.obj",*mesh);


    std::vector<std::vector<Eigen::Vector3f>> tmp_points;
    std::vector<std::vector<Eigen::Vector2f>> tmp_uvs;
    int num_materials_ = (int)mesh->textures_.size();

    std::vector<int> array_offsets_; //??
    std::vector<GLsizei> draw_array_sizes_;
    std::vector<GLuint> vertex_position_buffers_;
    std::vector<GLuint> vertex_uv_buffers_;
    std::vector<GLuint> texture_buffers_;

    array_offsets_.resize(num_materials_);
    draw_array_sizes_.resize(num_materials_);
    vertex_position_buffers_.resize(num_materials_);
    vertex_uv_buffers_.resize(num_materials_);
    texture_buffers_.resize(num_materials_);

    tmp_uvs.resize(num_materials_);
    tmp_points.resize(num_materials_);

    unsigned short index = 0;
    std::vector<unsigned short> indices;
    //Bind vertex and uv per material

    for(std::size_t i = 0 ; i < mesh->triangles_.size(); i++) {
        const Eigen::Vector3i& triangle = mesh->triangles_[i];
        int mi = mesh->triangle_material_ids_[i]; // material id

        for(std::size_t j=0; j < 3 ; j++) {
            std::size_t idx = 3 * i +j;
            int vertex_idx = triangle(j);
            tmp_points[mi].push_back(mesh->vertices_[vertex_idx].cast<float>());
            tmp_uvs[mi].push_back(mesh->triangle_uvs_[idx].cast<float>());
            indices.push_back(index++);
        }
    }


    createVisualizerWindow("test",image_width,image_height,50,50,true);
    GLuint vao_id;
    // 1. bind Vertex Array Object
    glGenVertexArrays(1, &vao_id);
    glBindVertexArray(vao_id);



    GLuint program_id = LoadShaders(vert_shader.c_str(),frag_shader.c_str());


    GLuint vertex_position;
    GLuint vertex_uv;
    GLuint mvp_position;
    GLuint texture_;
    //get a handle for vertex position
    vertex_position = glGetAttribLocation(program_id,"vertex_position");

    // get a handle for uv position
    vertex_uv = glGetAttribLocation(program_id,"vertex_uv");

    //get a handle for "mvp" uniform
    mvp_position = glGetUniformLocation(program_id,"MVP");

    // get a handle for "diffuse_texture" uniform
    texture_ = glGetUniformLocation(program_id,"diffuse_texture");




    // Bind textures
    for (int mi = 0; mi < num_materials_; ++mi) {
        glGenTextures(1, &texture_buffers_[mi]);
        glBindTexture(GL_TEXTURE_2D, texture_buffers_[mi]);

        GLenum format, type;
        auto it = open3d::visualization::GLHelper::texture_format_map_.find(
                    mesh->textures_[mi].num_of_channels_);
        if (it == open3d::visualization::GLHelper::texture_format_map_.end()) {
            open3d::utility::LogWarning("Unknown texture format, abort!");
            abort();
        }
        format = it->second;

        it = open3d::visualization::GLHelper::texture_type_map_.find(
                    mesh->textures_[mi].bytes_per_channel_);
        if (it == open3d::visualization::GLHelper::texture_type_map_.end()) {
            open3d::utility::LogWarning("Unknown texture type, abort!");
            abort();
        }
        type = it->second;

        glTexImage2D(GL_TEXTURE_2D, 0, format, mesh->textures_[mi].width_,
                     mesh->textures_[mi].height_, 0, format, type,
                     mesh->textures_[mi].data_.data());

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
    std::vector<Eigen::Vector3f> points;
    std::vector<Eigen::Vector2f> uvs;
    for (int mi = 0; mi < num_materials_; ++mi) {
        points.insert(points.end(), tmp_points[mi].begin(),
                      tmp_points[mi].end());
        uvs.insert(uvs.end(), tmp_uvs[mi].begin(), tmp_uvs[mi].end());
    }

    GLenum draw_arrays_mode_ = GL_TRIANGLES;


    // Create buffers and bind the geometry
    for (int mi = 0; mi < num_materials_; ++mi) {
        glGenBuffers(1, &vertex_position_buffers_[mi]);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffers_[mi]);
        glBufferData(GL_ARRAY_BUFFER,
                     draw_array_sizes_[mi] * sizeof(Eigen::Vector3f),
                     points.data() + array_offsets_[mi], GL_STATIC_DRAW);

        glGenBuffers(1, &vertex_uv_buffers_[mi]);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_uv_buffers_[mi]);
        glBufferData(GL_ARRAY_BUFFER,
                     draw_array_sizes_[mi] * sizeof(Eigen::Vector2f),
                     uvs.data() + array_offsets_[mi], GL_STATIC_DRAW);
    }

    /*
    do
    {

        glClear(GL_COLOR_BUFFER_BIT |GL_DEPTH_BUFFER_BIT);


        glUseProgram(program_id);
        for (int mi = 0; mi < num_materials_; ++mi) {
            glUniformMatrix4fv(mvp_position, 1, GL_FALSE, view_matrix.data());

            glUniform1i(texture_, 0);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture_buffers_[mi]);

            glEnableVertexAttribArray(vertex_position);
            glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffers_[mi]);
            glVertexAttribPointer(vertex_position, 3, GL_FLOAT, GL_FALSE, 0, NULL);

            glEnableVertexAttribArray(vertex_uv);
            glBindBuffer(GL_ARRAY_BUFFER, vertex_uv_buffers_[mi]);
            glVertexAttribPointer(vertex_uv, 2, GL_FLOAT, GL_FALSE, 0, NULL);
            glViewport(0,0,image_width,image_height);
            glDrawArrays(draw_arrays_mode_, 0, draw_array_sizes_[mi]);

            glDisableVertexAttribArray(vertex_position);
            glDisableVertexAttribArray(vertex_uv);
            glBindTexture(GL_TEXTURE_2D, 0);
        }


        // Swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();
    } // Check if the ESC key was pressed or the window was closed
    while( glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
           glfwWindowShouldClose(window) == 0 );

    */


    glClear(GL_COLOR_BUFFER_BIT |GL_DEPTH_BUFFER_BIT);


    glUseProgram(program_id);
    for (int mi = 0; mi < num_materials_; ++mi) {
        glUniformMatrix4fv(mvp_position, 1, GL_FALSE, view_matrix.data());

        glUniform1i(texture_, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture_buffers_[mi]);

        glEnableVertexAttribArray(vertex_position);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffers_[mi]);
        glVertexAttribPointer(vertex_position, 3, GL_FLOAT, GL_FALSE, 0, NULL);

        glEnableVertexAttribArray(vertex_uv);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_uv_buffers_[mi]);
        glVertexAttribPointer(vertex_uv, 2, GL_FLOAT, GL_FALSE, 0, NULL);
        glViewport(0,0,image_width,image_height);
        glDrawArrays(draw_arrays_mode_, 0, draw_array_sizes_[mi]);

        glDisableVertexAttribArray(vertex_position);
        glDisableVertexAttribArray(vertex_uv);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    // glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    // glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    // glReadBuffer(GL_COLOR_ATTACHMENT0 + GBUFFER_TEXTURE_TYPE_COLOR);

    cv::Mat mat(image_height, image_width, CV_8UC3);
    //use fast 4-byte alignment (default anyway) if possible
    glPixelStorei(GL_PACK_ALIGNMENT, (mat.step & 3) ? 1 : 4);
    //set length of one complete row in destination data (doesn't need to equal img.cols)
    glPixelStorei(GL_PACK_ROW_LENGTH, mat.step/mat.elemSize());
    glReadPixels(0, 0, mat.cols, mat.rows, GL_BGR, GL_UNSIGNED_BYTE, mat.data);
    cv::flip(mat, mat, 0);
    cv::imwrite("mat.png",mat);

    return 0;
}

