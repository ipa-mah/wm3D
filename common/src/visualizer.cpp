#include "perception_utils/visualizer.hpp"
#include "perception_utils/utils.hpp"


std::function<void(int,int)> frame_size_call_back_ = [](int,int){};

void processInput(GLFWwindow *window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}
Visualizer::Visualizer() {}
Visualizer::~Visualizer()
{
    glfwDestroyWindow(window_);
    glfwTerminate();
}
bool Visualizer::createVisualizerWindow(const std::string &window_name,
                                        const int width ,
                                        const int height,
                                        const int left,
                                        const int top,
                                        const bool visible)
{

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

    window_ = glfwCreateWindow(width,height,window_name_.c_str(),NULL,NULL);
    if (!window_) {
        std::cerr<<"Failed to create window\n"<<std::endl;
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window_);
    glfwSetWindowUserPointer(window_,this);

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
}

