#include "perception_utils/utility/vision_utils.hpp"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyobjloader/tiny_obj_loader.h"
namespace  VisionUtils {

bool readOBJFromFile(const std::string& filename,TextureMeshPtr& mesh)
{
    mesh = TextureMeshPtr(new TextureMesh);
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn;
    std::string err;

    std::string mtl_base_path = Utils::getFileParentDirectory(filename);
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                                filename.c_str(), mtl_base_path.c_str());
    if (!warn.empty()) {
        PRINT_RED("Read OBJ failed: %s ", warn.c_str());
    }
    if (!err.empty()) {
        PRINT_RED("Read OBJ failed: %s", err.c_str());
    }

    if (!ret) {
        return false;
    }
    // copy vertex and data
    for (size_t vidx = 0; vidx < attrib.vertices.size(); vidx += 3) {
        tinyobj::real_t vx = attrib.vertices[vidx + 0];
        tinyobj::real_t vy = attrib.vertices[vidx + 1];
        tinyobj::real_t vz = attrib.vertices[vidx + 2];
        mesh->vertices_.push_back(Eigen::Vector3d(vx, vy, vz));
        mesh->point_cloud_.push_back(pcl::PointXYZ(vx, vy, vz));
    }

    for (size_t vidx = 0; vidx < attrib.colors.size(); vidx += 3) {
        tinyobj::real_t r = attrib.colors[vidx + 0];
        tinyobj::real_t g = attrib.colors[vidx + 1];
        tinyobj::real_t b = attrib.colors[vidx + 2];
        mesh->vertex_colors_.push_back(Eigen::Vector3d(r, g, b));
    }
    // resize normal data and create bool indicator vector
    mesh->vertex_normals_.resize(mesh->vertices_.size());
    std::vector<bool> normals_indicator(mesh->vertices_.size(), false);
    // copy face data and copy normals data
    // append face-wise uv data
    for (size_t s = 0; s < shapes.size(); s++) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            int fv = shapes[s].mesh.num_face_vertices[f];
            if (fv != 3) {
                PRINT_RED("Read OBJ failed: facet with number of vertices not "
                          "equal to 3");
                return false;
            }

            Eigen::Vector3i facet;
            for (int v = 0; v < fv; v++) {
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                int vidx = idx.vertex_index;
                facet(v) = vidx;

                if (!attrib.normals.empty() && !normals_indicator[vidx] &&
                        (3 * idx.normal_index + 2) < int(attrib.normals.size())) {
                    tinyobj::real_t nx =
                            attrib.normals[3 * idx.normal_index + 0];
                    tinyobj::real_t ny =
                            attrib.normals[3 * idx.normal_index + 1];
                    tinyobj::real_t nz =
                            attrib.normals[3 * idx.normal_index + 2];
                    mesh->vertex_normals_[vidx](0) = nx;
                    mesh->vertex_normals_[vidx](1) = ny;
                    mesh->vertex_normals_[vidx](2) = nz;
                    normals_indicator[vidx] = true;
                }

                if (!attrib.texcoords.empty() &&
                        2 * idx.texcoord_index + 1 < int(attrib.texcoords.size())) {
                    tinyobj::real_t tx =
                            attrib.texcoords[2 * idx.texcoord_index + 0];
                    tinyobj::real_t ty =
                            attrib.texcoords[2 * idx.texcoord_index + 1];
                    mesh->triangle_uvs_.push_back(Eigen::Vector2d(tx, ty));
                }
            }
            mesh->triangles_.push_back(facet);
            mesh->triangle_material_ids_.push_back(
                        shapes[s].mesh.material_ids[f]);
            index_offset += fv;
        }
    }

    // if not all normals have been set, then remove the vertex normals
    bool all_normals_set =
            std::accumulate(normals_indicator.begin(), normals_indicator.end(),
                            true, [](bool a, bool b) { return a && b; });
    if (!all_normals_set) {
        mesh->vertex_normals_.clear();
    }

    // if not all triangles have corresponding uvs, then remove uvs
    if (3 * mesh->triangles_.size() != mesh->triangle_uvs_.size()) {
        mesh->triangle_uvs_.clear();
    }

    // Now we assert only one shape is stored, we only select the first
    // diffuse material
    for (auto& material : materials) {
        if (!material.diffuse_texname.empty()) {
            mesh->textures_.push_back(cv::imread(mtl_base_path+material.diffuse_texname));
        }
    }

    return true;
}

}
