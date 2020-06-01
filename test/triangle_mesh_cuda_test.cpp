
#include <Open3D/Open3D.h>
#include <cuda/camera/camera_intrinsic_cuda.hpp>
#include <cuda/container/device_array.hpp>
#include <cuda/cuda_headers.hpp>
#include <cuda/geometry/triangle_mesh_cuda.hpp>
#include <cuda/integration/tsdf_volume_cuda.hpp>
#include <iostream>
#include <memory>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <wm3D/integration/tsdf_volume.hpp>
#include <wm3D/utility/utils.hpp>

int main()
{
	std::string mesh_file = "/home/ipa-mah/1_projects/wm3D/data/texture_model.obj";
	open3d::geometry::TriangleMesh mesh;
	open3d::io::ReadTriangleMeshFromOBJ(mesh_file, mesh,true);
	cuda::TriangleMeshCuda cuda_mesh(100000, 100000);
	cuda_mesh.upload(mesh);
	DeviceArray<Eigen::Vector3f> devices;
	std::vector<Eigen::Vector3f> ver;
	for (size_t i = 0; i < mesh.vertices_.size(); i++)
	{
		/* code */
		Eigen::Vector3f v = mesh.vertices_[i].cast<float>();
		ver.push_back(v);
	}
	
	devices.upload(ver);
	cuda::testTriangleMeshCuda(devices);
	return 0;
}
