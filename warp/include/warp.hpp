#ifndef WARP_HPP
#define WARP_HPP

#include <opencv2/opencv.hpp>
#include "asmmodel.h"
#include "modelfile.h"

cv::Mat drawPtsOnImg(cv::Mat img, std::vector<cv::Point2d> pts);
cv::Mat drawPtsOnImg(cv::Mat img, std::vector<cv::Point> pts);
cv::Mat showASMpts(cv::Mat img);
cv::Rect findFace(cv::Mat img);
cv::Rect runCascade(cv::Mat img, cv::CascadeClassifier cc);
cv::Rect boundingRect(std::vector<cv::Point2d> pts);

cv::Mat findOpticalFlow(cv::Mat a, cv::Mat b);
cv::Mat warpImgWithFlow(cv::Mat img, cv::Mat flow);

std::vector<cv::Point2d> findLandmarksFaceSDK(cv::Mat img);

cv::Mat alignToLandmarks(cv::Mat img, std::vector<cv::Point2d> landmarks);
#endif //WARP_HPP
