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

#endif //WARP_HPP
