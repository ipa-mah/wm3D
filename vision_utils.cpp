#include "perception_utils/vision_utils.hpp"


std::vector<int> image_y_bases;
bool VisionUtils::readMTLandTextureImages(const std::string obj_folder, const std::string mtl_fname, std::unordered_map<std::string, int>& material_names,
                                          std::vector<cv::Mat>& texture_images,cv::Mat& texture_atlas)
{
    std::ifstream readin(mtl_fname, std::ios::in);
    if (readin.fail() || readin.eof())
    {
        std::cout << "Cannot read MTL file " << mtl_fname << std::endl;
        return false;
    }
    std::string str_line, str_first, str_img, str_mtl;
    while (!readin.eof() && !readin.fail())
    {
        getline(readin, str_line);
        if (!readin.good() || readin.eof())
            break;
        std::istringstream iss(str_line);
        if (iss >> str_first)
        {
            if (str_first == "newmtl")
            {
                iss >> str_mtl;
                material_names[str_mtl] = int(texture_images.size());
            }
            if (str_first == "map_Kd")
            {
                iss >> str_img;
                str_img = obj_folder + str_img;
                cv::Mat img = cv::imread(str_img, CV_LOAD_IMAGE_COLOR);
                if (img.empty() || img.depth() != CV_8U)
                {
                    std::cout << "ERROR: cannot read color image " << str_img << std::endl;
                    return false;
                }
                texture_images.push_back(std::move(img));
            }
        }
    }
    readin.close();
    std::cout<<"VisionUtils::readMTLandTextureImages: number of texture images: "<<texture_images.size()<<std::endl;

    // Stick multiple texture images into one final image for rendering.
    // The final image size will be (texture-image-columns, n * texture-image-height),
    // and both height and width are enlarged to values power of 2. Here n is the number
    // of input texture images (suppose all texture images have exactly the same resolution).

    int width = 0, height = 0;
    int y_base = 0;

    for (int i = 0; i < texture_images.size(); ++i)
    {
        width = std::max(width, texture_images[i].cols);
        image_y_bases.push_back(y_base);
        y_base += texture_images[i].rows;
    }
    height = y_base;

    // Ths size of texture image loaded in OpenGL must be power of 2
    int n = 2;
    while (n < width)
        n *= 2;
    int texture_atlas_width = n;
    n = 2;
    while (n < height)
        n *= 2;
    int texture_atlas_height_ = n;
    // texture_image_ = cv::imread("test_red.png"); // debug
    texture_atlas = cv::Mat(texture_atlas_height_, texture_atlas_width, CV_8UC3, cv::Scalar(0, 0, 0));
    //copy texture images to texture atlas
    for (int i = 0; i < texture_images.size(); ++i)
    {
        texture_images[i].copyTo(texture_atlas(cv::Rect(0, texture_atlas_height_ - image_y_bases[i] - texture_images[i].rows,
                                                        texture_images[i].cols, texture_images[i].rows)));

    }
    //cv::imwrite("test.png", texture_atlas); // debug
    return true;


}
bool VisionUtils::readOBJFile(const std::string file_name,TriangleMesh::Ptr& mesh)
{
    std::ifstream readin(file_name, std::ios::in);
    if (readin.fail() || readin.eof())
    {
        std::cout << "Cannot read OBJ file " << file_name << std::endl;
        return false;
    }
    std::string str_line, str_first, str_mtl_name, mtl_fname;
    std::vector<glm::vec3> vertices, normals;
    std::vector<glm::vec2> uvs;
    float x, y, z;
    unsigned int f, vt, vn, cur_tex_idx;
    std::string obj_folder;
    int face_vidx = 0;
    while (!readin.eof() && !readin.fail())
    {
        getline(readin, str_line);
        if (!readin.good() || readin.eof())
            break;
        if (str_line.size() <= 1)
            continue;
        std::istringstream iss(str_line);
        iss >> str_first;
        if (str_first[0] == '#')
            continue;
        else if (str_first == "mtllib")
        {  // mtl file
            iss >> mtl_fname;
            size_t pos = file_name.find_last_of("\\/");
            if (pos != std::string::npos)
                obj_folder = file_name.substr(0, pos + 1);
            mtl_fname = obj_folder + mtl_fname;
            if (!readMTLandTextureImages(obj_folder,mtl_fname,mesh.material_names,
                                         mesh.texture_images,mesh.texture_atlas))
                return false;
        }
        else if (str_first == "v")
        {
            iss >> x >> y >> z;
            vertices.push_back(glm::vec3(x, y, z));
            pcl::PointXYZRGBNormal pt;
            pt.normal_x = 0;
            pt.normal_y = 0;
            pt.normal_z = 0;
            pt.x = x;
            pt.y = y;
            pt.z = z;

            mesh->cloud->points.push_back(pt);
        }
        else if (str_first == "vt")
        {
            iss >> x >> y;
            uvs.push_back(glm::vec2(x, y));
            if (!mesh.vtx_texture)
                mesh.vtx_texture = true;
        }
        else if (str_first == "vn")
        {
            iss >> x >> y >> z;
            normals.push_back(glm::vec3(x, y, z));
            if (!mesh.vtx_normal)
                mesh.vtx_normal = true;
        }
        else if (str_first == "usemtl")
        {
            iss >> str_mtl_name;
            if (mesh.material_names.find(str_mtl_name) == mesh.material_names.end())
            {
                std::cout << "ERROR: cannot find this material " << str_mtl_name << " in the mtl file " << mtl_fname << std::endl;
                return false;
            }
            cur_tex_idx = mesh.material_names[str_mtl_name];
        }
        else if (str_first == "f")
        {
            int loop = 3;
            while (loop-- > 0)
            {
                iss >> f;

                f--;
                //vtx.pos = vertices[f];
                if (mesh->vtx_texture)
                {
                    if (mesh->vtx_normal)
                    {  // 'f/vt/vn'
                        iss.get();
                        iss >> vt;
                        vt--;
                        iss.get();
                        iss >> vn;
                        vn--;
                    }
                    else
                    {  // 'f/vt'
                        iss.get();
                        iss >> vt;
                        vt--;
                    }
                    // Since we put/stick multiple input texture images into one large final image, so we need to
                    // modify corresponding texture uv coordinates.
                    vtx.uv[0] = uvs[vt][0] * mesh.texture_images[cur_tex_idx].cols / mesh.texture_atlas.cols;
                    vtx.uv[1] = (uvs[vt][1] * mesh.texture_images[cur_tex_idx].rows + image_y_bases[cur_tex_idx]) /
                            ( mesh.texture_atlas.rows);
                    if (mesh.vtx_normal)
                        vtx.normal = normals[vn];
                }
                else if (mesh.vtx_normal)
                {  // 'f//vn'
                    iss.get();
                    iss.get();
                    iss >> vn;
                    vn--;
                    vtx.normal = normals[vn];
                }


                mesh->faces.push_back(face_vidx++);
            }
        }
        // otherwise continue -- unrecognized line
    }
    readin.close();
    mesh->vertex_num_ = int(vertices.size());
    mesh->face_num_ = face_vidx / 3;
    mesh->mesh_suffix_ = "obj";
    return true;

}
bool VisionUtils::readPLYFile(const std::string file_name, TriangleMesh::Ptr& mesh)
{
    pcl::PolygonMesh polygon_mesh;
    pcl::io::loadPLYFile(file_name,polygon_mesh);
    pcl::PointCloud<pcl::PointNormal> cloud;
    pcl::fromPCLPointCloud2(polygon_mesh.cloud,cloud);
    pcl::fromPCLPointCloud2(polygon_mesh.cloud,*mesh->cloud);

  //  mesh->vertices.resize(polygon_mesh.polygons.size()*3);
    mesh->faces.resize(polygon_mesh.polygons.size()*3);
    mesh->vtx_normal = true;


    for (int i=0;i<polygon_mesh.polygons.size();i++) {
        pcl::Vertices v = polygon_mesh.polygons[i];
        mesh->faces[i*3+0]= v.vertices[0];
        mesh->faces[i*3+1]= v.vertices[1];
        mesh->faces[i*3+2]= v.vertices[2];
    }
    mesh->vertex_num_ = int(cloud.points.size());
    mesh->face_num_ = polygon_mesh.polygons.size();
    mesh->mesh_suffix_ = "ply";
    return true;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}
bool VisionUtils::initGLFWWindow(GLFWwindow* window)
{
    if (!glfwInit())
    {
        std::cout << "Failed to initialize GLFW" << std::endl;
        return false;
    }
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // To make MacOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // We don't want the old OpenGL
    //if (program_mode_ != RENDER_MODEL)
    //  glfwWindowHint(GLFW_VISIBLE, false);  // hide window after creation
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);  // uncomment this statement to fix compilation on OS X
#endif

    // Open a window and create its OpenGL context
    window = glfwCreateWindow(1920, 1080, "RenderingWindow", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version "
                     "of the tutorials."
                  << std::endl; // this is from some online tutorial
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // Initialize GLEW
    glewExperimental = true;  // Needed for core profile
    if (glewInit() != GLEW_OK)
    {
        fprintf(stderr, "Failed to initialize GLEW\n");
        getchar();
        glfwTerminate();
        return false;
    }

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

    // Dark blue background
    glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    return true;
}
