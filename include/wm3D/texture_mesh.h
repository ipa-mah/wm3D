#pragma once
#include <unordered_set>
#include <memory>
#include <iostream>
#include <eigen3/Eigen/Core>
#include <pcl/common/common.h>
#include <opencv2/opencv.hpp>


class TextureMesh
{

public:
    std::vector<Eigen::Vector3d> vertices_;
    pcl::PointCloud<pcl::PointXYZ> point_cloud_;
    std::vector<Eigen::Vector3d> vertex_normals_;
    std::vector<Eigen::Vector3d> vertex_colors_;
    std::vector<Eigen::Vector3i> triangles_;
    std::vector<Eigen::Vector3d> triangle_normals_;
    std::vector<std::unordered_set<int>> adjacency_list_;
    std::vector<Eigen::Vector2d> triangle_uvs_;
    std::vector<int> triangle_material_ids_;
    std::vector<cv::Mat> textures_;

public:
    void getMinMax3D(Eigen::Vector4f& min_pt, Eigen::Vector4f& max_pt)
    {
        pcl::getMinMax3D(point_cloud_,min_pt,max_pt);
    }
public:
    using Ptr = std::shared_ptr<TextureMesh>;
    using ConstPtr = std::shared_ptr<const TextureMesh>;
}; // struct TextureMesh

using TextureMeshPtr = TextureMesh::Ptr;
using TextureMeshConstPtr = TextureMesh::ConstPtr;


