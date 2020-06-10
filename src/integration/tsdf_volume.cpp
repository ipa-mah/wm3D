#include <cmath>
#include <utility>
#include <wm3D/integration/tsdf_volume.hpp>

TSDFVolume::TSDFVolume(const Eigen::Vector3i& dims, const float voxel_length) : dims_(dims), voxel_length_(voxel_length)
{
}
void TSDFVolume::downloadTsdfAndWeights(const DeviceArray2D<float>& device_tsdf, const DeviceArray2D<float>& device_weights)
{
	tsdf_.resize(dims_(0) * dims_(1) * dims_(2));
	weights_.resize(dims_(0) * dims_(1) * dims_(2));
	const size_t NNN = dims_(0) * dims_(1) * dims_(2);
	//CheckCuda(cudaMemcpy(tsdf_.data(), device_tsdf, sizeof(float) * NNN, cudaMemcpyDeviceToHost));
	//CheckCuda(cudaMemcpy(weights_.data(), device_weights, sizeof(float) * NNN, cudaMemcpyDeviceToHost));

	/*
	tsdf_.assign(dims_(0) * dims_(1) * dims_(2), 0.0f);
	weights_.assign(dims_(0) * dims_(1) * dims_(2), 0.0f);
	cv::Mat tsdf_mat, weight_mat;
	tsdf_mat.create(device_tsdf.rows(), device_tsdf.cols(), CV_32FC1);
	weight_mat.create(device_weights.rows(), device_weights.cols(), CV_32FC1);

	device_tsdf.download(tsdf_mat.ptr<void>(), tsdf_mat.step);
	device_weights.download(weight_mat.ptr<void>(), weight_mat.step);

#pragma omp parallel for schedule(dynamic)
	for (int x = 0; x < dims_(0); x++)
	{
		for (int y = 0; y < dims_(1); y++)
		{
			for (int z = 0; z < dims_(2); z++)
			{
				int idx = z * dims_(0) * dims_(1) + y * dims_(0) + x;
				float tsdf_value = tsdf_mat.at<float>(z * dims_(1) + y, x);
				float weight = weight_mat.at<float>(z * dims_(1) + y, x);
				tsdf_[idx] = tsdf_value;
				weights_[idx] = weight;
			}
		}
	}
	*/
}
pcl::PolygonMesh TSDFVolume::extractMesh(const Eigen::Vector3d& crop_min, const Eigen::Vector3d& crop_max)
{
	pcl::PolygonMesh polygon_mesh;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	std::unordered_map<Eigen::Vector4i, int, Utils::hash_eigen::hash<Eigen::Vector4i>> edgeindex_to_vertexindex;
	int edge_to_index[12];
	for (int x = 0; x < dims_(0) - 1; x++)
	{
		for (int y = 0; y < dims_(1) - 1; y++)
		{
			for (int z = 0; z < dims_(2) - 1; z++)
			{
				Eigen::Vector3d pt_world(static_cast<double>(voxel_length_) * (0.5 + x), static_cast<double>(voxel_length_) * (0.5 + y), static_cast<double>(voxel_length_) * (0.5 + z));
				if (pt_world(0) > crop_max(0) || pt_world(0) < crop_min(0) || pt_world(1) > crop_max(1) || pt_world(1) < crop_min(1) || pt_world(2) > crop_max(2) || pt_world(2) < crop_min(2))
					continue;
				int cube_index = 0;
				float f[8];
				// check a cube with 8 voxels, break
				for (int i = 0; i < 8; i++)
				{
					Eigen::Vector3i idx = Eigen::Vector3i(x, y, z) + shift[i];
					// if weight of 1 voxel =0
					if (weights_[IndexOf(idx)] == 0.0f)
					{
						cube_index = 0;
						break;
					}
					// remember cube index and get tsdf
					else
					{
						f[i] = tsdf_[IndexOf(idx)];
						if (f[i] < 0.0f)
						{
							cube_index |= (1 << i);
						}
					}
				}
				if (cube_index == 0 || cube_index == 255) continue;
				for (int i = 0; i < 12; i++)
				{
					if (edge_table[cube_index] & (1 << i))
					{
						Eigen::Vector4i edge_index = Eigen::Vector4i(x, y, z, 0) + edge_shift[i];
						if (edgeindex_to_vertexindex.find(edge_index) == edgeindex_to_vertexindex.end())
						{
							edge_to_index[i] = static_cast<int>(cloud->points.size());
							edgeindex_to_vertexindex[edge_index] = static_cast<int>(cloud->points.size());
							// Coordinate(x,y,z) of edge index in volume
							Eigen::Vector3d pt(static_cast<double>(voxel_length_) * (0.5 + edge_index(0)), static_cast<double>(voxel_length_) * (0.5 + edge_index(1)),
											   static_cast<double>(voxel_length_) * (0.5 + edge_index(2)));

							float f0 = std::fabs(static_cast<float>(f[edge_to_vert[i][0]]));
							float f1 = std::fabs(static_cast<float>(f[edge_to_vert[i][1]]));
							pt(edge_index(3)) += static_cast<double>(f0) * static_cast<double>(voxel_length_) / static_cast<double>((f0 + f1));
							pcl::PointXYZ p;
							p.x = static_cast<float>(pt(0));
							p.y = static_cast<float>(pt(1));
							p.z = static_cast<float>(pt(2));
							cloud->points.push_back(p);
						}
						else
						{
							edge_to_index[i] = edgeindex_to_vertexindex.find(edge_index)->second;
						}
					}
				}

				for (int i = 0; tri_table[cube_index][i] != -1; i += 3)
				{
					pcl::Vertices v;
					v.vertices.push_back(edge_to_index[tri_table[cube_index][i]]);
					v.vertices.push_back(edge_to_index[tri_table[cube_index][i + 2]]);
					v.vertices.push_back(edge_to_index[tri_table[cube_index][i + 1]]);
					polygon_mesh.polygons.push_back(v);
				}
			}
		}
	}
	pcl::toPCLPointCloud2(*cloud, polygon_mesh.cloud);
	return std::move(polygon_mesh);
}

pcl::PointCloud<pcl::PointNormal>::Ptr TSDFVolume::extractPointCloud(const Eigen::Vector3d& crop_min, const Eigen::Vector3d& crop_max)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normal(new pcl::PointCloud<pcl::Normal>);

	for (int x = 0; x < dims_(0) - 1; x++)
	{
		for (int y = 0; y < dims_(1) - 1; y++)
		{
			for (int z = 0; z < dims_(2) - 1; z++)
			{
				// Index of voxel
				Eigen::Vector3i idx0(x, y, z);
				float w0 = weights_[IndexOf(idx0)];
				float f0 = tsdf_[IndexOf(idx0)];
				if (w0 != 0.0f && f0 < 0.98f && f0 >= -0.98f)
				{
					// get voxel coordinate
					Eigen::Vector3d p0(static_cast<double>(voxel_length_) * (0.5 + x), static_cast<double>(voxel_length_) * (0.5 + y), static_cast<double>(voxel_length_) * (0.5 + z));
					if (p0(0) > crop_max(0) || p0(0) < crop_min(0) || p0(1) > crop_max(1) || p0(1) < crop_min(1) || p0(2) > crop_max(2) || p0(2) < crop_min(2)) continue;
					for (int i = 0; i < 3; i++)
					{
						// increment in x,y,z direction
						Eigen::Vector3d p1 = p0;
						p1(i) += static_cast<double>(voxel_length_);
						Eigen::Vector3i idx1 = idx0;
						idx1(i) += 1;
						// If inside the cube
						if (idx1(i) < dims_(0) - 1)
						{
							float w1 = weights_[IndexOf(idx1)];
							float f1 = tsdf_[IndexOf(idx1)];
							// If 2 voxel idx0,idx1 are valid
							if (w1 != 0.0f && f1 < 0.98f && f1 >= -0.98f && f0 * f1 < 0)
							{
								// absolute value
								float r0 = std::fabs(f0);
								float r1 = std::fabs(f1);
								Eigen::Vector3d p = p0;
								pcl::PointXYZ pt(p(0), p(1), p(2));
								// vertex is the interpolation between 2 voxels
								p(i) = (p0(i) * static_cast<double>(r1) + p1(i) * static_cast<double>(r0)) / static_cast<double>((r0 + r1));
								Eigen::Vector3d normal = getNormalAt(p);
								pcl::Normal pt_normal(normal(0), normal(1), normal(2));

								cloud->points.push_back(pt);
								cloud_normal->points.push_back(pt_normal);
							}
						}
					}
				}
			}
		}
	}
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normal(new pcl::PointCloud<pcl::PointNormal>);
	pcl::concatenateFields(*cloud, *cloud_normal, *cloud_with_normal);
	return std::move(cloud_with_normal);
}

Eigen::Vector3d TSDFVolume::getNormalAt(const Eigen::Vector3d& p)
{
	Eigen::Vector3d n;
	const float half_gap = 0.99 * voxel_length_;
	for (int i = 0; i < 3; i++)
	{
		Eigen::Vector3d p0 = p;
		p0(i) -= half_gap;
		Eigen::Vector3d p1 = p;
		p1(i) += half_gap;
		n(i) = getTSDFAt(p1) - getTSDFAt(p0);
	}
	return n.normalized();
}
float TSDFVolume::getTSDFAt(const Eigen::Vector3d& p)
{
	Eigen::Vector3i idx;
	Eigen::Vector3d p_grid = p / voxel_length_ - Eigen::Vector3d(0.5, 0.5, 0.5);
	for (int i = 0; i < 3; i++)
	{
		idx(i) = (int)std::floor(p_grid(i));
	}
	Eigen::Vector3d r = p_grid - idx.cast<double>();

	return (1 - r(0)) * ((1 - r(1)) * ((1 - r(2)) * tsdf_[IndexOf(idx + Eigen::Vector3i(0, 0, 0))] + r(2) * tsdf_[IndexOf(idx + Eigen::Vector3i(0, 0, 1))]) +
						 r(1) * ((1 - r(2)) * tsdf_[IndexOf(idx + Eigen::Vector3i(0, 1, 0))] + r(2) * tsdf_[IndexOf(idx + Eigen::Vector3i(0, 1, 1))])) +
		   r(0) * ((1 - r(1)) * ((1 - r(2)) * tsdf_[IndexOf(idx + Eigen::Vector3i(1, 0, 0))] + r(2) * tsdf_[IndexOf(idx + Eigen::Vector3i(1, 0, 1))]) +
				   r(1) * ((1 - r(2)) * tsdf_[IndexOf(idx + Eigen::Vector3i(1, 1, 0))] + r(2) * tsdf_[IndexOf(idx + Eigen::Vector3i(1, 1, 1))]));
}
