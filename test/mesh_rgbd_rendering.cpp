#include <jsoncpp/json/json.h>
#include <iostream>
// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>
#include <eigen3/Eigen/Core>

#include <pcl/TextureMesh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/texture_mapping.h>

#include <pcl/io/vtk_lib_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include "wm3D/utility/vision_utils.hpp"
#include "wm3D/visualization/render_texture_mesh.hpp"

int main(int argc, char** argv)
{
	std::string data_path = "../data/";
	std::string texture_file = data_path + "texture_model.obj";
	std::string vert_shader = "../src/shaders/TextureSimpleVertexShader.glsl";
	std::string frag_shader = "../src/shaders/TextureSimpleFragmentShader.glsl";

	/*

		std::vector<Eigen::Matrix4d> extrinsics;
		Eigen::Matrix3d intrins;

		readDataFromJsonFile(data_path+"config.json",extrinsics,intrins);
		int width = (intrins(0,2)+0.5)*2;
		int height = (intrins(1,2)+0.5)*2;



		std::shared_ptr<open3d::geometry::TriangleMesh> mesh = std::make_shared<open3d::geometry::TriangleMesh>();
		open3d::io::ReadTriangleMesh(data_path+"texture_model.obj",*mesh);

		std::shared_ptr<RenderTextureMesh> render =
				std::make_shared<RenderTextureMesh>("render_texture_mesh",vert_shader,frag_shader);
		render->CreateVisualizerWindow("wm3D",width,height,50,50,true);
		render->compileShaders();


		render->readTextureMesh(mesh);
		for(int i=0; i< extrinsics.size(); i++)
		{
			std::cout<<extrinsics[i]<<std::endl;
			cv::Mat img;
			render->rendering(intrins,extrinsics[i],img);
			std::ostringstream curr_frame_prefix;
			curr_frame_prefix << std::setw(6) << std::setfill('0') << i;
			cv::imwrite("frame-"+curr_frame_prefix.str()+".rtexture.png",img);

		}
		*/
	return 0;
}
