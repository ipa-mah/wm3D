#include <cuda/integration/tsdf_volume_cuda.hpp>
namespace cuda
{
TSDFVolumeCuda::TSDFVolumeCuda()
{
	dims_ = Eigen::Vector3i(-1, -1, -1);
}

TSDFVolumeCuda::TSDFVolumeCuda(Eigen::Vector3i dims, float voxel_length, float sdf_trunc)
{
	create(dims, voxel_length, sdf_trunc);
}

TSDFVolumeCuda::~TSDFVolumeCuda()
{
	release();
}

void TSDFVolumeCuda::create(const Eigen::Vector3i& dims, const float voxel_length, const float sdf_trunc)
{
	dims_ = dims;
	voxel_length_ = voxel_length;
	inv_voxel_length_ = 1 / voxel_length;
	sdf_trunc_ = sdf_trunc;

	volume_to_world_.setIdentity();
	world_to_volume_ = volume_to_world_.inverse();
	tsdf_volume_.create(dims(2) * dims(1), dims(0));
	weight_volume_.create(dims(2) * dims(1), dims(0));
	initializeVolume();
}

void TSDFVolumeCuda::release()
{
	weight_volume_.release();
	tsdf_volume_.release();
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<Eigen::Vector3i>> TSDFVolumeCuda::downloadVolume()
{
	std::vector<float> tsdf, weights;
	std::vector<Eigen::Vector3i> color;


	tsdf.resize(dims_(0) * dims_(1) * dims_(2));
	weights.resize(dims_(0) * dims_(1) * dims_(2));
	color.resize(dims_(0) * dims_(1) * dims_(2));

	const size_t NNN = dims_(0) * dims_(1) * dims_(2);
	CheckCuda(cudaMemcpy(tsdf.data(), tsdf_volume_, sizeof(float) * NNN,
                         cudaMemcpyDeviceToHost));
    CheckCuda(cudaMemcpy(weights.data(), weight_volume_, sizeof(float) * NNN,
                         cudaMemcpyDeviceToHost));


	return std::make_tuple(tsdf, weights, color);
}

}  // namespace cuda