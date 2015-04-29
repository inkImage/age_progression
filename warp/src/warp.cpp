#include "warp.hpp"

#include "asmmodel.h"

using namespace cv;
using namespace std;
using namespace StatModel;

Mat showASMpts(Mat img)
{
    imshow("img", img);
    waitKey(0);

    Rect faceRect = findFace(img);

    ASMModel asmDetector;
    string asmFname = "./3rdparty/asmlib-opencv-read-only/data/muct76.model";
    asmDetector.loadFromFile(asmFname);
    vector<Rect> faces; faces.push_back(faceRect);
    int verboseL = 0;
    vector < ASMFitResult > fitResult = asmDetector.fitAll(img, faces, verboseL);
    vector<Point> points;
    fitResult[0].toPointList(points);

    Mat toDraw = img.clone();
    for(size_t i = 0; i < points.size(); i++)
    {
        circle(toDraw, points[i], 2, Scalar::all(255));
    }

    return toDraw;
}


Mat drawPtsOnImg(Mat img, vector<Point2d> pts)
{
    Mat buf = img.clone();
    for(size_t i = 0; i < pts.size(); i++)
    {
        circle(buf, pts[i], 2, Scalar(255, 0, 0));
    }
    return buf;
}


Mat drawPtsOnImg(Mat img, vector<Point> pts)
{
    Mat buf = img.clone();
    for(size_t i = 0; i < pts.size(); i++)
    {
        circle(buf, pts[i], 2, Scalar(255, 0, 0));
    }
    return buf;
}


Rect boundingRect(vector<Point2d> pts)
{
    int minx = 10000, miny = 10000, maxx = 0, maxy = 0;
    for(size_t j = 0; j < pts.size(); j++)
    {
        int x = pts[j].x, y = pts[j].y;
        minx = min(minx, x); miny = min(miny, y);
        maxx = max(maxx, x); maxy = max(maxy, y);
    }
    Rect r(minx, miny, maxx-minx, maxy-miny);
    return r;
}


Rect findFace(Mat img)
{
    string profileCascadeFname =
            "./3rdparty/asmlib-opencv-read-only/data/haarcascade_profileface.xml";
    string frontalCascadeFname =
            "./3rdparty/asmlib-opencv-read-only/data/haarcascade_frontalface_alt.xml";
    CascadeClassifier frontal(frontalCascadeFname);
    CascadeClassifier profile(profileCascadeFname);

    Rect frontR = runCascade(img, frontal);

    if(frontR == Rect())
    {
        Rect profileR = runCascade(img, profile);
        if(profileR == Rect())
        {
            return Rect(0, 0, img.cols, img.rows);
        }
        else
        {
            return profileR;
        }
    }
    else
    {
        return frontR;
    }
}


Rect runCascade(Mat img, CascadeClassifier cc)
{
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

