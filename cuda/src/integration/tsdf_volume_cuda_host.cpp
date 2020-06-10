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

	CheckCuda(cudaMemcpy(tsdf.data(), tsdf_volume_, sizeof(float) * NNN, cudaMemcpyDeviceToHost));
	CheckCuda(cudaMemcpy(weights.data(), weight_volume_, sizeof(float) * NNN, cudaMemcpyDeviceToHost));
	return std::make_tuple(tsdf, weights, color);
}
///////////////////////////////////////////////////////////

TestTSDFVolume::TestTSDFVolume()
{
	res_ = -1;
}
TestTSDFVolume::TestTSDFVolume(int res, float voxel_length, float sdf_trunc, Eigen::Matrix4d& volume_to_world)
{
	voxel_length_ = voxel_length;
	sdf_trunc_ = sdf_trunc;
	volume_to_world_ = volume_to_world;
	create(res);
}
void TestTSDFVolume::create(int res)
{
	 if (device_ != nullptr) {
        printf("[UniformTSDFVolumeCuda] Already created!\n");
        return;
    }
	res_ = res;
	device_ = std::make_shared<TSDFVolumeDevice>();
	const size_t NNN = res * res * res;
	tsdf_volume_.create(res*res,res);
	weight_volume_.create(res*res,res);
	
	device_->tsdf_volume_ = tsdf_volume_;
	device_->weight_volume_ = weight_volume_;
	updateDevice();
	initializeVolume();
}
void TestTSDFVolume::updateDevice()
{
	if (device_ != nullptr)
	{
		device_->res_ = res_;
		device_->voxel_length_ = voxel_length_;
		device_->inv_voxel_lenth_ = 1.0 / voxel_length_;
		device_->sdf_trunc_ = sdf_trunc_;
		device_->volume_to_world_ = volume_to_world_.cast<float>();
		device_->world_to_volume_ = volume_to_world_.cast<float>().inverse();
	}
}

}  // namespace cuda