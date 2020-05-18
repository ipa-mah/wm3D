#include <wm3D/cuda/tsdf_volume_cuda.h>
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

} 