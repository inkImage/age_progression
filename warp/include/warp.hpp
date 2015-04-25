#ifndef WARP_HPP
#define WARP_HPP

#include <opencv2/opencv.hpp>
#include "asmmodel.h"

cv::Mat warpToFrontal(cv::Mat img);
cv::Rect findFace(cv::Mat img);

#endif //WARP_HPP
