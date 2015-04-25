#include "warp.hpp"

using namespace cv;


Mat warpToFrontal(Mat img)
{
    Rect faceRect = findFace(img);

    if(faceRect != Rect())
    {
        Mat cropped = img(faceRect);

    }

}


Rect findFace(Mat img)
{

    string cascadeFname = "~/opencv/OpenCV-2.4.3/data/haarcascades/haarcascade_frontalface_default.xml";
    CascadeClassifier cc(cascadeFname);

    vector<Rect> objects;
    vector<int> rejectLevels;
    vector<double> levelWeights;
    double scaleFactor=1.1;
    int minNeighbours=3, flags=0;
    Size minSize=Size(), maxSize=Size();
    bool outputRejectLevels=true;
    cc.detectMultiScale(img, objects, rejectLevels, levelWeights, scaleFactor,
                        minNeighbours, flags, minSize, maxSize, outputRejectLevels);

    if(objects.size() == 0)
    {
        return Rect();
    }
    else
    {
        return objects[0];
    }
}
