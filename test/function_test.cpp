#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
// Include GLEW
#include <wm3D/texture_mesh.h>
#include <wm3D/view_control.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/texture_mapping.h>
#include <pcl/TextureMesh.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/ply_io.h>

#include <pcl/io/vtk_lib_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <json/json.h>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>

Eigen::Matrix3d intrins;;
std::vector<Eigen::Matrix4d> extrinsics;
pcl::PointCloud<pcl::PointXYZ> cloud;
void visual()
{

    int image_width = (intrins(0,2)+0.5) * 2;
    int image_height = (intrins(1,2)+0.5) * 2;
    // visualization object
    pcl::visualization::PCLVisualizer visu ("cameras");
    // read current camera
    double focal_x = intrins(0,0);
    double focal_y = intrins(1,1);
    double height = intrins(1,2)*2;
    double width = intrins(0,2)*2;
    for (int view = 0;view < extrinsics.size();view++) {
        // create a 5-point visual for each camera
        Eigen::Vector3d p1, p2, p3, p4, p5;
        p1 = Eigen::Vector3d(0,0,0);
        //double angleX = RAD2DEG (2.0 * atan (width / (2.0*focal)));
        //double angleY = RAD2DEG (2.0 * atan (height / (2.0*focal)));
        double dist = 0.05;
        double minX, minY, maxX, maxY;
        maxX = dist*tan (atan (width / (2.0*focal_x)));
        minX = -maxX;
        maxY = dist*tan (atan (height / (2.0*focal_y)));
        minY = -maxY;
        p2 = Eigen::Vector3d(minX,minY,dist);
        p3 = Eigen::Vector3d(maxX,minY,dist);
        p4 = Eigen::Vector3d(maxX,maxY,dist);
        p5 = Eigen::Vector3d(minX,maxY,dist);

        //Transform points from camera coordinate to world coordinate
        p1=(extrinsics[view].inverse() * Eigen::Vector4d(p1[0], p1[1], p1[2], 1.0)).head<3>();
        p2=(extrinsics[view].inverse() * Eigen::Vector4d(p2[0], p2[1], p2[2], 1.0)).head<3>();
        p3=(extrinsics[view].inverse() * Eigen::Vector4d(p3[0], p3[1], p3[2], 1.0)).head<3>();
        p4=(extrinsics[view].inverse() * Eigen::Vector4d(p4[0], p4[1], p4[2], 1.0)).head<3>();
        p5=(extrinsics[view].inverse() * Eigen::Vector4d(p5[0], p5[1], p5[2], 1.0)).head<3>();
        pcl::PointXYZ pt1,pt2,pt3,pt4,pt5;
        pt1 = pcl::PointXYZ(p1.cast<float>()[0], p1.cast<float>()[1], p1.cast<float>()[2]);
        pt2 = pcl::PointXYZ(p2.cast<float>()[0], p2.cast<float>()[1], p2.cast<float>()[2]);
        pt3 = pcl::PointXYZ(p3.cast<float>()[0], p3.cast<float>()[1], p3.cast<float>()[2]);
        pt4 = pcl::PointXYZ(p4.cast<float>()[0], p4.cast<float>()[1], p4.cast<float>()[2]);
        pt5 = pcl::PointXYZ(p5.cast<float>()[0], p5.cast<float>()[1], p5.cast<float>()[2]);

        std::stringstream ss;
        //ss << "cam " << view;
        visu.addText3D(ss.str (), pt1, 0.1, 1.0, 1.0, 1.0, ss.str ());
        ss.str ("");
        ss << "camera_" << view << "line1";
        visu.addLine (pt1, pt2,ss.str ());
        ss.str ("");
        ss << "camera_" << view << "line2";
        visu.addLine (pt1, pt3,ss.str ());
        ss.str ("");
        ss << "camera_" << view << "line3";
        visu.addLine (pt1, pt4,ss.str ());
        ss.str ("");
        ss << "camera_" << view << "line4";
        visu.addLine (pt1, pt5,ss.str ());
        ss.str ("");
        ss << "camera_" << view << "line5";
        visu.addLine (pt2, pt5,ss.str ());
        ss.str ("");
        ss << "camera_" << view << "line6";
        visu.addLine (pt5, pt4,ss.str ());
        ss.str ("");
        ss << "camera_" << view << "line7";
        visu.addLine (pt4, pt3,ss.str ());
        ss.str ("");
        ss << "camera_" << view << "line8";
        visu.addLine (pt3, pt2,ss.str ());
    }
    // add a coordinate system
    visu.addCoordinateSystem (1.0, "global");
    // add the mesh's cloud (colored on Z axis)
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> color_handler (cloud.makeShared(), "z");
    visu.addPointCloud (cloud.makeShared(), color_handler, "cloud");

    // reset camera
    visu.resetCamera ();

    // wait for user input
    visu.spin ();
}
int main( void )
{


    std::string data_path = "../sample_data/";
    std::string texture_file = data_path+"texture_model.obj";
    float distance = 0.5;
    int vertical_views = 10;
    int horizontal_views = 10;

    intrins.setIdentity();
    intrins(0,0) = 500;
    intrins(1,1) = 500;
    intrins(0,2) = 320;
    intrins(1,2) = 240;
    std::cout<<intrins<<std::endl;
    TriangleMesh::Ptr mesh = std::make_shared<TriangleMesh>();
    readTextureMeshfromOBJFile(data_path+texture_file,mesh);
    for(const auto& v : mesh->vertices_)
        cloud.points.push_back(pcl::PointXYZ(v[0],v[1],v[2]));

    ViewControl::Ptr view_control = std::make_shared<ViewControl>(intrins,vertical_views,horizontal_views,distance);
    extrinsics = view_control->getExtrinsics();
    visual();




    return 0;
}

