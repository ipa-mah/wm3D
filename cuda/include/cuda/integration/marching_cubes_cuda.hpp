#pragma once
#include <cuda/common/common.hpp>
#include <cuda/container/device_array.hpp>
#include <cuda/geometry/triangle_mesh_cuda.hpp>
#include <eigen3/Eigen/Core>
 
class MarchingCubesCuda
{
  private:
	uchar* table_indices_;
	Eigen::Vector3i* vertex_indices_;
	//TriangleMes
  public:
	MarchingCubesCuda(/* args */);
	~MarchingCubesCuda();

	Eigen::Vector3i dims_;
	
};

MarchingCubesCuda::MarchingCubesCuda(/* args */)
{
}

MarchingCubesCuda::~MarchingCubesCuda()
{
}
