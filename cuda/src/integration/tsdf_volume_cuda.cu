#include <cuda/integration/tsdf_volume_cuda.hpp>
namespace cuda {
__device__ inline bool TSDFVolumeCudaDevice::inVolume(const Eigen::Vector3i& x)
{
    return 0 <= x(0) && x(0) < (N_ - 1) && 0 <= x(1) && x(1) < (N_ - 1) &&
    0 <= x(2) && x(2) < (N_ - 1);
}

__device__ inline bool TSDFVolumeCudaDevice::inVolumef(const Eigen::Vector3f& x)
{
    return 0 <= x(0) && x(0) < (N_ - 1) && 0 <= x(1) && x(1) < (N_ - 1) &&
           0 <= x(2) && x(2) < (N_ - 1);
} 



__device__ inline Eigen::Vector3f TSDFVolumeCudaDevice::worldToVoxelf(
        const Eigen::Vector3f &x_w) {

                Eigen::Vector4f x4_w = Eigen::Vector4f(x_w(0),x_w(1),x_w(2),1);
                //transform to volume
                Eigen::Vector3f x_v = (world_to_volume_.template cast<float>() * x4_w).head<3>();
    return volumeToVoxelf(x_v);
}

__device__ inline Eigen::Vector3f TSDFVolumeCudaDevice::voxelfToWorld(
        const Eigen::Vector3f &x) {
                Eigen::Vector3f v = voxelfToVolume(x);
    return (volume_to_world_.template cast<float>() * Eigen::Vector4f(v(0),v(1),v(2),1)).head<3>();
}

__device__ inline Eigen::Vector3f TSDFVolumeCudaDevice::voxelfToVolume(
        const Eigen::Vector3f &x) {
    return Eigen::Vector3f((x(0) + 0.5f) * voxel_length_,
                    (x(1) + 0.5f) * voxel_length_,
                    (x(2) + 0.5f) * voxel_length_);
}

__device__ inline Eigen::Vector3f TSDFVolumeCudaDevice::volumeToVoxelf(
        const Eigen::Vector3f &x_v) {
    return Eigen::Vector3f(x_v(0) * inv_voxel_length_ - 0.5f,
                    x_v(1) * inv_voxel_length_ - 0.5f,
                    x_v(2) * inv_voxel_length_ - 0.5f);
}

/** trilinear interpolations. **/
/** Ensure it is called within [0, N - 1)^3 **/

__device__ float TSDFVolumeCudaDevice::tsdfAt(const Eigen::Vector3f& x)
{
    Eigen::Vector3i xi = x.template cast<int>();
    Eigen::Vector3f r = x - xi.template cast<float>();

    return (1 - r(0)) *
                   ((1 - r(1)) *
                            ((1 - r(2)) *
                                     tsdf_[indexOf(xi + Eigen::Vector3i(0, 0, 0))] +
                             r(2) * tsdf_[indexOf(xi + Eigen::Vector3i(0, 0, 1))]) +
                    r(1) * ((1 - r(2)) *
                                    tsdf_[indexOf(xi + Eigen::Vector3i(0, 1, 0))] +
                            r(2) * tsdf_[indexOf(xi + Eigen::Vector3i(0, 1, 1))])) +
           r(0) * ((1 - r(1)) *
                           ((1 - r(2)) *
                                    tsdf_[indexOf(xi + Eigen::Vector3i(1, 0, 0))] +
                            r(2) * tsdf_[indexOf(xi + Eigen::Vector3i(1, 0, 1))]) +
                   r(1) * ((1 - r(2)) * tsdf_[indexOf(xi + Eigen::Vector3i(1, 1, 0))] +
                           r(2) * tsdf_[indexOf(xi + Eigen::Vector3i(1, 1, 1))]));
}


__device__ uchar TSDFVolumeCudaDevice::weightAt(const Eigen::Vector3f &x) {
    Eigen::Vector3i xi = x.template cast<int>();
    Eigen::Vector3f r = x - xi.template cast<float>();

    return uchar(
            (1 - r(0)) *
                    ((1 - r(1)) *
                             ((1 - r(2)) *
                                      weight_[indexOf(xi + Eigen::Vector3i(0, 0, 0))] +
                              r(2) * weight_[indexOf(xi + Eigen::Vector3i(0, 0, 1))]) +
                     r(1) * ((1 - r(2)) *
                                     weight_[indexOf(xi + Eigen::Vector3i(0, 1, 0))] +
                             r(2) * weight_[indexOf(xi + Eigen::Vector3i(0, 1, 1))])) +
            r(0) * ((1 - r(1)) *
                            ((1 - r(2)) *
                                     weight_[indexOf(xi + Eigen::Vector3i(1, 0, 0))] +
                             r(2) * weight_[indexOf(xi + Eigen::Vector3i(1, 0, 1))]) +
                    r(1) * ((1 - r(2)) *
                                    weight_[indexOf(xi + Eigen::Vector3i(1, 1, 0))] +
                            r(2) * weight_[indexOf(xi + Eigen::Vector3i(1, 1, 1))])));
}


__device__ Eigen::Vector3f TSDFVolumeCudaDevice::colorAt(const Eigen::Vector3f &x) {
    Eigen::Vector3i xi = x.template cast<int>();
    Eigen::Vector3f r = x - xi.template cast<float>();

    Eigen::Vector3f colorf =
            (1 - r(0)) *
                    ((1 - r(1)) *
                             ((1 - r(2)) *
                                      color_[indexOf(xi + Eigen::Vector3i(0, 0, 0))]
                                              .template cast<float>() +
                              r(2) * color_[indexOf(xi + Eigen::Vector3i(0, 0, 1))]
                                              .template cast<float>()) +
                     r(1) * ((1 - r(2)) *
                                     color_[indexOf(xi + Eigen::Vector3i(0, 1, 0))]
                                             .template cast<float>() +
                             r(2) * color_[indexOf(xi + Eigen::Vector3i(0, 1, 1))]
                                             .template cast<float>())) +
            r(0) * ((1 - r(1)) *
                            ((1 - r(2)) *
                                     color_[indexOf(xi + Eigen::Vector3i(1, 0, 0))]
                                             .template cast<float>() +
                             r(2) * color_[indexOf(xi + Eigen::Vector3i(1, 0, 1))]
                                             .template cast<float>()) +
                    r(1) * ((1 - r(2)) * color_[indexOf(xi + Eigen::Vector3i(1, 1, 0))]
                                                 .template cast<float>() +
                            r(2) * color_[indexOf(xi + Eigen::Vector3i(1, 1, 1))]
                                            .template cast<float>()));
    return colorf;
    
}

__device__ void TSDFVolumeCudaDevice::integrate(const Eigen::Vector3i& x,
        const PtrStepSz<uchar3>& color_image,
        const PtrStepSz<ushort>& depth_image,
        const Eigen::Matrix3f& intrins,
        const Eigen::Matrix4f& cam_to_world,
        int image_width,
        int image_height,
        float depth_scale)
{
        //Transform voxel from volume to world
        Eigen::Vector3f x_w = voxelfToWorld(x.template cast<float>());
        // transform voxel from world to camera
        Eigen::Vector3f x_c = (cam_to_world* Eigen::Vector4f(x_w(0),x_w(1),x_w(1),1)).head<3>();
        int2 pixel = make_int2(
                __float2int_rn(intrins(0,0) * x_c(0) / x_c (2) + intrins(0,2)),
                __float2int_rn(intrins(1,1) * x_c(1) / x_c (2) + intrins(1,2)));
        if (pixel.x < 0 || pixel.x >= image_width || pixel.y< 0 || pixel.y>= image_height)
                return;
        float d = depth_image.ptr(pixel.y)[pixel.x] * depth_scale;
        if (d <= 0.0001 || d>5.0)
                return;
        float tsdf = d - x_c(2);
        if(tsdf <= - sdf_trunc_) return;
        tsdf = fminf(tsdf/sdf_trunc_, 1.0f);
        uchar3 color = color_image.ptr(pixel.y)[pixel.x];

        float &tsdf_sum = this->tsdf(x);
        uchar &weight_sum = this->weight(x);
        Eigen::Vector3i &color_sum = this->color(x);

        float w0 = 1 / (weight_sum + 1.0f);
        float w1 = 1 - w0;

        tsdf_sum = tsdf * w0 + tsdf_sum * w1;
        color_sum = Eigen::Vector3i(color.x * w0 + color_sum(0) * w1,
                         color.y * w0 + color_sum(1) * w1,
                         color.z * w0 + color_sum(2) * w1);
        weight_sum = uchar(fminf(weight_sum + 1.0f, 255));

}

// __device__ Eigen::Vector3f TSDFVolumeCudaDevice::rayCasting(const Eigen::Vector2i& p, const Eigen::Matrix3d& intrins,
//         const Eigen::Matrix4f& cam_to_world)
// {



// }

} 
