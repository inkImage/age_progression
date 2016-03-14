#ifndef CALIB_H
#define CALIB_H

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <opencv/cv.h>
#include "util_calib.h"
#include <vector>
#include <cmath>

const bool DEBUG_CALIB = false;

void startCalibration(cv::Size picSize,
                      std::vector<cv::Point2d> pts2d, std::vector<cv::Point3d> pts3d,
                      cv::Mat camMat, cv::Mat rmat, cv::Mat tvec,
                      bool onlyExtrinsic, int useExtrinsicGuess);
                      
#endif //CALIB_H
