#include <pcl/PolygonMesh.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/features/board.h>
#include <pcl/io/io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cuda/camera/camera_intrinsic_cuda.hpp>
#include <cuda/container/device_array.hpp>
#include <cuda/cuda_headers.hpp>
#include <cuda/geometry/triangle_mesh_cuda.hpp>
#include <cuda/integration/tsdf_volume_cuda.hpp>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <wm3D/integration/tsdf_volume.hpp>
#include <wm3D/utility/utils.hpp>
int main()
{
	std::string data_path =
		"/home/ipa-mah/catkin_ws/data/"
		"scanStation_rec_ipatest_2020-05-13-09-20-01/2020-05-16-22-11-12/";
	Eigen::Matrix3d cam_param;
	int image_width, image_height, num_views;
	float depth_scale;
	Utils::readIntrinsicsAndNumViews(data_path, cam_param, num_views, image_width, image_height, depth_scale);
	std::cout << "Read RGBD frames" << std::endl;
	std::vector<cv::Mat> color_images(num_views), depth_images(num_views);
	std::vector<Eigen::Matrix4d> cam2worlds(num_views);
	num_views -= 800;
	int resolution = 512;
	float voxel_length = 0.001;
	float sdf_trunc = voxel_length * 5;
	std::cout << "num views: " << num_views << std::endl;
	std::cout << "voxel_length: " << voxel_length << std::endl;
	std::cout << "sdf_trunc: " << sdf_trunc << std::endl;
	std::cout << "resolution: " << resolution << std::endl;

	cuda::CameraIntrinsicCuda intrins(cam_param.cast<float>(), image_width, image_height);
	DeviceArray2D<uchar3> color_image_cuda;
	DeviceArray2D<ushort> depth_image_cuda;
	DeviceArray2D<float3> model_normal,model_vertex;
	DeviceArray2D<uchar3> render_normal;
	color_image_cuda.create(image_height, image_width);
	depth_image_cuda.create(image_height, image_width);	
	render_normal.create(image_height,image_width);
	cuda::TSDFVolumeCuda::Ptr volume = std::make_shared<cuda::TSDFVolumeCuda>(resolution, voxel_length, sdf_trunc);

	for (int frame_idx = 0; frame_idx < num_views; frame_idx++)
	{
		std::ostringstream curr_frame_prefix;
		curr_frame_prefix << std::setw(6) << std::setfill('0') << frame_idx;
		// // Read current frame depth
		std::string depth_im_file = data_path + "/frame-" + curr_frame_prefix.str() + ".depth.png";
		std::string rgb_im_file = data_path + "/frame-" + curr_frame_prefix.str() + ".color.png";
		std::string cam2world_file = data_path + "/frame-" + curr_frame_prefix.str() + ".pose.txt";
		cv::Mat color = cv::imread(rgb_im_file);
		if (color.empty()) std::cout << "rgb error" << std::endl;
		color_images[frame_idx] = color;
		cv::Mat depth = cv::imread(depth_im_file, 2);
		if (depth.empty()) std::cout << "depth error" << std::endl;
		depth_images[frame_idx] = depth;

		std::ifstream pose_f;
		pose_f.open(cam2world_file.c_str());
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				pose_f >> cam2worlds[frame_idx](i, j);
			}
		}
		pose_f.close();
	}
	std::cout<<"ok"<<std::endl;
	cv::Mat ray_cast(image_height,image_width,CV_8UC3);
	for (size_t frame_idx = 0;frame_idx < num_views; frame_idx++)
	{
		
		color_image_cuda.upload(color_images[frame_idx].data, color_images[frame_idx].step,
								 color_images[frame_idx].rows, color_images[frame_idx].cols);
		depth_image_cuda.upload(depth_images[frame_idx].data, depth_images[frame_idx].step,
								 depth_images[frame_idx].rows, depth_images[frame_idx].cols);
		volume->integrateTsdfVolume(depth_image_cuda, intrins, cam2worlds[frame_idx].cast<float>().inverse(), 0.001);
		volume->rayCasting(model_vertex,model_normal,intrins,cam2worlds[frame_idx].cast<float>(),0.7);
		cuda::createRenderMap(model_normal,render_normal);
		render_normal.download(ray_cast.ptr<void>(),ray_cast.step);
		cv::namedWindow("image",CV_WINDOW_NORMAL);
		cv::imshow("image",ray_cast);
		cv::waitKey(12);
	}
	
	cv::destroyAllWindows();
	std::cout << "tsdf" << std::endl;
	
	DeviceArray2D<Eigen::Vector3i> vertex_indices;
	DeviceArray2D<int> table_indices;
	vertex_indices.create(resolution * resolution, resolution);
	table_indices.create(resolution * resolution, resolution);
	// cuda::allocateVertexHost(volume->tsdf_volume_,volume->weight_volume_,vertex_indices,table_indices,
	// 								dims);
	/*
	Eigen::Vector3d crop_min(0.03, 0.03, 0.009);
	Eigen::Vector3d crop_max(0.399, 0.285, 0.4);
	TSDFVolume::Ptr surface = std::make_shared<TSDFVolume>(dims, 0.001);
	surface->downloadTsdfAndWeights(volume->tsdf_volume_, volume->weight_volume_);
	std::cout << "extract mesh" << std::endl;
	pcl::PolygonMesh mesh = surface->extractMesh(crop_min, crop_max);
	pcl::io::savePLYFile("mesh0.ply", mesh);
	*/
	return 0;
}
