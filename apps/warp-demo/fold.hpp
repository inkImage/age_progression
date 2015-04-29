#ifndef FOLD_HPP
#define FOLD_HPP

#include <iostream>
#include <opencv2/opencv.hpp>
#include "warp.hpp"

struct FaceData
{
    //loaded data
    std::string userId,
                originalImage;
    int         faceId;
    int         ageLow;  //age is written as
    int         ageHigh; //"%d" or "(%d, %d)" or "None"
    int         gender;    //"m"=0, "f"=1, else=-1
    cv::Rect    faceRect;
    int         tiltAng;
    int         fiducialYawAngle;
    int         fiducialScore;
    int         score2;
    int         yaw2;

    //calculated data
    std::string imgFname;
    std::vector<cv::Point2d> landmarks;
};

typedef std::vector<FaceData> FoldData;

cv::Ptr<FoldData> readFold(std::string fname);
vector<cv::Point2d> readFidu(std::string fname, int &score, int &yaw);
#endif //FOLD_HPP
