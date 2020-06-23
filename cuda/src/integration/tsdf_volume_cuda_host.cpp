#include <cuda/integration/tsdf_volume_cuda.hpp>
namespace cuda
{
TSDFVolumeCuda::TSDFVolumeCuda()
{
	res_ = -1;
}

TSDFVolumeCuda::TSDFVolumeCuda(int res, float voxel_length, float sdf_trunc)
{
	create(res, voxel_length, sdf_trunc);
}

TSDFVolumeCuda::~TSDFVolumeCuda()
{
	release();
}

void TSDFVolumeCuda::create(const int res, const float voxel_length, const float sdf_trunc)
{
	res_ = res;
	voxel_length_ = voxel_length;
	inv_voxel_length_ = 1 / voxel_length;
	sdf_trunc_ = sdf_trunc;

	volume_to_world_.setIdentity();
	world_to_volume_ = volume_to_world_.inverse();
	tsdf_volume_.create(res * res, res);
	weight_volume_.create(res * res, res);
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
	
	const size_t NNN = res_ * res_ * res_;

	tsdf.resize(NNN);
	weights.resize(NNN);
	color.resize(NNN);

	CheckCuda(cudaMemcpy(tsdf.data(), tsdf_volume_, sizeof(float) * NNN,
                         cudaMemcpyDeviceToHost));
    CheckCuda(cudaMemcpy(weights.data(), weight_volume_, sizeof(float) * NNN,
                         cudaMemcpyDeviceToHost));


	return std::make_tuple(tsdf, weights, color);
}

}  // namespace cuda