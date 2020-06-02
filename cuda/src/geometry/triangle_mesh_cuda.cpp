#include <cuda/geometry/triangle_mesh_cuda.hpp>
namespace cuda
{
TriangleMeshCuda::TriangleMeshCuda()
{
	max_vertices_ = -1;
	max_triangles_ = -1;
}
TriangleMeshCuda::~TriangleMeshCuda()
{
}
void TriangleMeshCuda::create(int max_vertices, int max_triangles)
{
	max_vertices_ = max_vertices;
	max_triangles_ = max_triangles;
}
TriangleMeshCuda::TriangleMeshCuda(int max_vertices, int max_triangles)
{
	create(max_vertices, max_triangles);
}

void TriangleMeshCuda::upload(const open3d::geometry::TriangleMesh& mesh)
{
	std::vector<Eigen::Vector3f> vertices, vertex_normals;
	std::vector<Eigen::Vector3i> triangles;
	if (!mesh.HasVertices() || !mesh.HasTriangles())
	{
		printf("Empty mesh!\n");
		return;
	}
	vertices.resize(mesh.vertices_.size());
	triangles.resize(mesh.triangles_.size());
	// upload vertices to gpu
	for (size_t i = 0; i < mesh.vertices_.size(); ++i)
	{
		vertices[i] = Eigen::Vector3f(mesh.vertices_[i](0), mesh.vertices_[i](1), mesh.vertices_[i](2));
	}
	gpu_vertices_.upload(vertices);
	// upload triangles to gpu
	for (size_t i = 0; i < mesh.triangles_.size(); ++i)
	{
		triangles[i] = Eigen::Vector3i(mesh.triangles_[i](0), mesh.triangles_[i](1), mesh.triangles_[i](2));
	}
	gpu_triangles_.upload(triangles);

	if (mesh.HasVertexNormals())
	{
		vertex_normals.resize(mesh.vertices_.size());
		for (size_t i = 0; i < mesh.vertices_.size(); ++i)
		{
			vertex_normals[i] = Eigen::Vector3f(mesh.vertex_normals_[i](0), mesh.vertex_normals_[i](1), mesh.vertex_normals_[i](2));
		}
		gpu_normals_.upload(vertex_normals);
	}
}
}  // namespace cuda
