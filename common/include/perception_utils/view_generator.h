#include <iostream>
#include <json/json.h>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

class ViewGenerator
{
public:
    virtual ~ViewGenerator();
    explicit ViewGenerator(const Eigen::Matrix3d& intrins, const int vertical_views, const int horizontal_views);
    void savePoses2Json(const std::string pose_file);
private:
    Json::Value root_;
};
