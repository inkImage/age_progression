#ifndef WARP_HPP
#define WARP_HPP

#include <opencv2/opencv.hpp>
#include "asmmodel.h"
#include "modelfile.h"

cv::Mat warpToFrontal(cv::Mat img);
cv::Rect findFace(cv::Mat img);
cv::Rect runCascade(cv::Mat img, cv::CascadeClassifier cc);

#endif //WARP_HPP
