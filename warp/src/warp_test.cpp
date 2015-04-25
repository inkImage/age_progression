#include <iostream>
#include <opencv2/opencv.hpp>
#include "warp.hpp"

using namespace cv;

int main()
{
    Mat image;

    string fname;
    image = imread(fname);

    Mat warped = warpToFrontal(image);

    imshow("warped", warped);
    waitKey(0);

    return 0;
}
