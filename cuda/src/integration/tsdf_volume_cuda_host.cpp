#include <cuda/integration/tsdf_volume_cuda.hpp>
namespace cuda
{
TSDFVolumeCuda::TSDFVolumeCuda()
{
	dims_ = Eigen::Vector3i(-1, -1, -1);

}

TSDFVolumeCuda::TSDFVolumeCuda(Eigen::Vector3i dims, float voxel_length, float sdf_trunc)
{
	create(dims,voxel_length,sdf_trunc);
}

TSDFVolumeCuda::~TSDFVolumeCuda()
{
}

void TSDFVolumeCuda::create(const Eigen::Vector3i& dims, const float voxel_length, const float sdf_trunc)
{
	dims_ = dims;
	voxel_length_ = voxel_length;
	inv_voxel_length_ = 1/voxel_length;
	sdf_trunc_ = sdf_trunc;

	volume_to_world_.setIdentity();
	world_to_volume_ = volume_to_world_.inverse();
	tsdf_volume_.create(dims(2) * dims(1), dims(0));
	weight_volume_.create(dims(2) * dims(1), dims(0));
	initializeVolume();


}


}  // namespace cuda