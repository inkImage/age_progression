#ifndef FOLD_HPP
#define FOLD_HPP

#include <iostream>
#include <opencv2/opencv.hpp>
#include "warp.hpp"

struct FaceData
{

};

typedef std::vector<FaceData> FoldData;

cv::Ptr<FoldData> readFold(std::string fname);

#endif //FOLD_HPP
