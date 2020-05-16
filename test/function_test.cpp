#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <wm3D/utility/utils.hpp>
int main()
{

  std::string data_path = "/home/ipa-mah/catkin_ws/data/"
                          "scanStation_rec_ipatest_2020-05-13-09-20-01/2020-05-16-22-11-12/";
  Eigen::Matrix3d cam_param;
  int image_width, image_height, num_views;
  float depth_scale;
  Utils::readIntrinsicsAndNumViews(data_path,cam_param,num_views,image_width,image_height,depth_scale);
  std::cout<<"Read RGBD frames"<<std::endl;
  std::vector<cv::Mat> color_images(num_views), depth_images(num_views);
  std::vector<Eigen::Matrix4d> cam2worlds(num_views);
  for (int frame_idx = 0 ;frame_idx < num_views-800 ; frame_idx++) {
    std::ostringstream curr_frame_prefix;
    curr_frame_prefix << std::setw(6) << std::setfill('0') << frame_idx;
    // // Read current frame depth
    std::string depth_im_file = data_path + "/frame-" + curr_frame_prefix.str() + ".depth.png";
    std::string rgb_im_file = data_path + "/frame-" + curr_frame_prefix.str() + ".color.png";
    std::string cam2world_file = data_path + "/frame-" + curr_frame_prefix.str() + ".pose.txt";

    cv::Mat color = cv::imread(rgb_im_file);
    if(color.empty()) std::cout<<"rgb error"<<std::endl;
    color_images[frame_idx] = color;
    cv::Mat depth = cv::imread(depth_im_file,2);
    if(depth.empty()) std::cout<<"depth error"<<std::endl;
    depth_images[frame_idx] = depth;

    std::ifstream pose_f;
    pose_f.open(cam2world_file.c_str());
    for (int i=0;i<4;i++){
      for (int j=0;j<4;j++) {
        pose_f>>cam2worlds[frame_idx](i,j);
      }
    }
    pose_f.close();
  }
  return 0;
}
