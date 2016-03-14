#include <iostream>
#include <opencv2/opencv.hpp>
#include "warp.hpp"

#include "fold.hpp"
#include "util.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    if(argc > 1)
    {
        Mat image = imread(argv[1]);
        if(image.empty())
        {
            cout << "Failed to load image!" << endl;
            return -1;
        }
        else
        {
            vector<Mat> chans;
            cv::split(image, chans);
            double val = 0;
            for(int i = 0; i < 3; i++)
            {
                Mat sx, sy, mag, angles;
                Sobel(chans[i], sx, CV_32F, 1, 0);
                Sobel(chans[i], sy, CV_32F, 0, 1);
                cartToPolar(sx, sy, mag, angles);
                val += mean(mag)[0]/3;
            }
            cout << "value is " << val << endl;
        }
    }
    else
    {
        cout << "Not enough args!" << endl;
        return -1;
    }

    return 0;
}


