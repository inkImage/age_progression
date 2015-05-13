#ifndef WARP_HPP
#define WARP_HPP

#include <opencv2/opencv.hpp>
#include "asmmodel.h"
#include "modelfile.h"
#include "tracker/FaceTracker.hpp"
#include "tracker/myFaceTracker.hpp"
#include "tracker/ShapeModel.hpp"
#include "util.hpp"

cv::Mat drawPtsOnImg(cv::Mat img, std::vector<cv::Point2d> pts);
cv::Mat drawPtsOnImg(cv::Mat img, std::vector<cv::Point> pts);
cv::Mat showASMpts(cv::Mat img);
cv::Rect findFace(cv::Mat img);
cv::Rect runCascade(cv::Mat img, cv::CascadeClassifier cc);
cv::Rect boundingRect(std::vector<cv::Point2d> pts);

cv::Mat findOpticalFlow(cv::Mat a, cv::Mat b);
cv::Mat warpImgWithFlow(cv::Mat img, cv::Mat flow);

void findLandmarksFaceSDK(cv::Mat img, vector<cv::Point2d> &shape2d,
                          vector<cv::Point3d> &shape3d, FACETRACKER::Pose &pose,
                          cv::Rect rect);

cv::Mat alignToLandmarks(cv::Mat img, std::vector<cv::Point2d> landmarks);

bool loadObj(const std::string fname, Vertices& model);


#endif //WARP_HPP
