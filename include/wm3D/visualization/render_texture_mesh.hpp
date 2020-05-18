#pragma once

#include <fstream>
#include <iostream>
#include <memory>
// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>

#include <wm3D/texture_mesh.h>
#include <wm3D/visualization/shader.h>
#include <eigen3/Eigen/Core>
#include <wm3D/utility/utils.hpp>
class RenderMesh
{
  public:
	virtual ~RenderMesh()
	{
	}
	RenderMesh(const std::string& name, const std::string& render_texture_mesh_vertex_shader_file, const std::string& render_texture_mesh_fragment_shader_file)
	  : render_texture_mesh_vertex_shader_file_(render_texture_mesh_vertex_shader_file), render_texture_mesh_fragment_shader_file_(render_texture_mesh_fragment_shader_file)
	{
	}
	bool loadShaders();

	/// Function to create a window and initialize GLFW
	/// This function MUST be called from the main thread.
	bool CreateVisualizerWindow(const std::string& window_name = "wm3D", const int width = 640, const int height = 480, const int left = 50, const int top = 50, const bool visible = true);
	void compileShaders()
	{
		loadShaders();
		Compile();
	}

  protected:
	bool Compile();
	void Release();
	bool BindGeometry(const TriangleMesh::Ptr& mesh);
	bool RenderGeometry(const TriangleMesh::Ptr& mesh);
	void UnbindGeometry();

  protected:
	bool prepareRendering(const TriangleMesh::Ptr& mesh);
	bool prepareBinding(const TriangleMesh::Ptr& mesh, std::vector<Eigen::Vector3f>& points, std::vector<Eigen::Vector2f>& uvs);

  protected:
	GLuint vertex_position_;  // name of vertex position in gsls shader
	GLuint vertex_uv_;		  // name of uv position in gsls fragment shader
	GLuint texture_;
	GLuint MVP_;					  // extrinsics in gsls vertex shader
	int num_materials_;				  // number of texture images
	std::vector<int> array_offsets_;  //??
	std::vector<GLsizei> draw_array_sizes_;

	GLuint vao_id_;

	std::vector<GLuint> vertex_position_buffers_;
	std::vector<GLuint> vertex_uv_buffers_;
	std::vector<GLuint> texture_buffers_;

	std::string render_texture_mesh_vertex_shader_file_;	// vertex shader file
	std::string render_texture_mesh_fragment_shader_file_;  // fragment shader file
	std::string vertex_shader_code_;						// code string in vertex shader file
	std::string fragment_shader_code_;						// code string in fragment shader file

	// window
	GLFWwindow* window_ = NULL;
	std::string window_name_ = "wm3D";
	bool bound_;
};

class RenderTextureMesh : public RenderMesh

{
  public:
	RenderTextureMesh(const std::string& name, const std::string& render_texture_mesh_vertex_shader_file, const std::string& render_texture_mesh_fragment_shader_file)
	  : RenderMesh(name, render_texture_mesh_vertex_shader_file, render_texture_mesh_fragment_shader_file)
	{
	}
	void readTextureMesh(const TriangleMesh::Ptr& texture_mesh);
	void rendering(const Eigen::Matrix3d& intrins, const Eigen::Matrix4d& extrinsics, cv::Mat& rendered_image);

  protected:
};
