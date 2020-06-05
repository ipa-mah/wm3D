#pragma once
#include <Open3D/Open3D.h>
#include <cuda/common/common.hpp>
#include <cuda/container/device_array.hpp>
#include <cuda/integration/marching_cubes_table_cuda.hpp>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
namespace cuda
{
class TriangleMeshCudaDevice
{
  public:
	PtrStepSz<Eigen::Vector3f> vertices_;
	PtrStepSz<Eigen::Vector3f> normals_;
	PtrStepSz<Eigen::Vector3f> vertex_colors_;
	PtrStepSz<Eigen::Vector3i> triangles_;

  public:
	int max_vertices_;
	int max_triangles_;

  public:
	TriangleMeshCudaDevice(/* args */);
	using Ptr = std::shared_ptr<TriangleMeshCudaDevice>;
	using ConstPtr = std::shared_ptr<const TriangleMeshCudaDevice>;
};

class TriangleMeshCuda
{
  public:
	TriangleMeshCudaDevice::Ptr device_ = nullptr;
	DeviceArray<Eigen::Vector3f> vertices_;
	DeviceArray<Eigen::Vector3f> normals_;
	DeviceArray<Eigen::Vector3f> vertex_colors_;
	DeviceArray<Eigen::Vector3i> triangles_;

  public:
	int max_vertices_;
	int max_triangles_;

  public:
	using Ptr = std::shared_ptr<TriangleMeshCuda>;
	using ConstPtr = std::shared_ptr<const TriangleMeshCuda>;
	TriangleMeshCuda();
	TriangleMeshCuda(int max_vertices, int max_triangles);
	TriangleMeshCuda(const TriangleMeshCuda& other);
	TriangleMeshCuda& operator=(const TriangleMeshCuda& other);
	~TriangleMeshCuda();

	void reset();
	void updateDevice();
	void create(int max_vertices, int max_triangles);
	void release();

	bool hasVertices() const;
	bool hasTriangles() const;
	bool hasVertexNormals() const;
	bool hasVertexColors() const;

	void upload(const open3d::geometry::TriangleMesh& mesh);
	open3d::geometry::TriangleMesh download();

	TriangleMeshCuda& clear();
	bool isEmpty();
};

}  // namespace cuda
