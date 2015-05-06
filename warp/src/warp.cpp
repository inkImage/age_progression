#include "warp.hpp"

#include "asmmodel.h"
#include "tracker/FaceTracker.hpp"

using namespace cv;
using namespace std;
using namespace StatModel;
using namespace FACETRACKER;

//range is [0.0 .. 1.0]
void HSVtoRGB( float& r, float &g, float &b, float h, float s, float v )
{
    int i;
    float f, p, q, t;

    if( s == 0 ) {
        // achromatic (grey)
        r = g = b = v;
        return;
    }

    h *= 6.0;            // sector 0 to 5
    i = floor( h );
    f = h - i;          // factorial part of h
    p = v * ( 1 - s );
    q = v * ( 1 - s * f );
    t = v * ( 1 - s * ( 1 - f ) );

    switch( i ) {
        case 0:
            r = v; g = t; b = p;
            break;
        case 1:
            r = q; g = v; b = p;
            break;
        case 2:
            r = p; g = v; b = t;
            break;
        case 3:
            r = p; g = q; b = v;
            break;
        case 4:
            r = t; g = p; b = v;
            break;
        default:        // case 5:
            r = v; g = p;b = q;
            break;
    }
}


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


//pair<cv::Mat, cv::Mat> findOpticalFlow(Mat a, Mat b)
cv::Mat findOpticalFlow(Mat a, Mat b)
{
    Mat flowAb, flowBa;
    //use SimpleFlow method
    //calcOpticalFlowSF(a, b, flowAb,
    //                  3, 2, 4, 4.1, 25.5, 18, 55.0, 25.5, 0.35, 18, 55.0, 25.5, 10);
    Mat aGray, bGray;
    cvtColor(a, aGray, CV_BGR2GRAY);
    cvtColor(b, bGray, CV_BGR2GRAY);
    //calcOpticalFlowFarneback(aGray, bGray, flowAb, 0.5, 3, 15, 3, 5, 1.2, 0);
    double pyr_scale = 0.5;
    int levels = 4;
    int winSize = 32;
    int iterations = 1;
    int poly_n = 7;
    double poly_sigma = 1.5;
    int flags = OPTFLOW_FARNEBACK_GAUSSIAN;
    calcOpticalFlowFarneback(aGray, bGray, flowAb, pyr_scale, levels, winSize, iterations,
                             poly_n, poly_sigma, flags);

    //visualize
    Mat flowViz(flowAb.rows, flowAb.cols, CV_8UC3);
    for(int x = 0; x < flowAb.cols; x++)
    {
        for(int y = 0; y < flowAb.rows; y++)
        {
            Vec3b& v = flowViz.at<Vec3b>(y, x);
            Point2f& f = flowAb.at<Point2f>(y, x);
            double angle = atan2(f.y, f.x)/(2*CV_PI)+0.5;
            double rad = sqrt(f.x*f.x+f.y*f.y);
            float r, g, b;
            HSVtoRGB(r, g, b, angle, rad/8.0, 1.0);
            v[0] = saturate_cast<uchar>(b*255.0);
            v[1] = saturate_cast<uchar>(g*255.0);
            v[2] = saturate_cast<uchar>(r*255.0);
        }
    }

    imshow("viz flow", flowViz);
    waitKey(0);


    //and backwards
    //calcOpticalFlowSF(b, a, flowBa,
    //                  3, 2, 4, 4.1, 25.5, 18, 55.0, 25.5, 0.35, 18, 55.0, 25.5, 10);
    //return pair<Mat, Mat> (flowAb, flowBa);
    return flowAb;
}


cv::Mat warpImgWithFlow(cv::Mat img, cv::Mat flow)
{
    Mat flowMap(flow.size(), CV_32FC2);
    for (int y = 0; y < flow.rows; ++y)
    {
        for (int x = 0; x < flow.cols; ++x)
        {
            Point2f f = flow.at<Point2f>(y, x);
            flowMap.at<Point2f>(y, x) = Point2f(x + f.x, y + f.y);
        }
    }

    Mat newFrame;
    remap(img, newFrame, flowMap, Mat(), INTER_LANCZOS4);
    return newFrame;
}


cv::Mat compute_pose_image(const Pose &pose, int height, int width)
{
  cv::Mat_<cv::Vec<uint8_t,3> > rv = cv::Mat_<cv::Vec<uint8_t,3> >::zeros(height,width);
  cv::Mat_<double> axes = pose_axes(pose);
  cv::Mat_<double> scaling = cv::Mat_<double>::eye(3,3);

  for (int i = 0; i < axes.cols; i++) {
    axes(0,i) = -0.5*double(width)*(axes(0,i) - 1);
    axes(1,i) = -0.5*double(height)*(axes(1,i) - 1);
  }

  cv::Point centre(width/2, height/2);
  // pitch
  cv::line(rv, centre, cv::Point(axes(0,0), axes(1,0)), cv::Scalar(255,0,0));
  // yaw
  cv::line(rv, centre, cv::Point(axes(0,1), axes(1,1)), cv::Scalar(0,255,0));
  // roll
  cv::line(rv, centre, cv::Point(axes(0,2), axes(1,2)), cv::Scalar(0,0,255));

  return rv;
}


vector<Point2d> findLandmarksFaceSDK(Mat img)
{
    string model_pathname = DefaultFaceTrackerModelPathname();
    string params_pathname = DefaultFaceTrackerParamsPathname();

    Ptr<FaceTracker> ft = LoadFaceTracker();
    if(ft.empty())
    {
        cout << "Failed to load Face SDK but it was expected" <<endl;
    }
    else
    {
        cout << "It's pretty strange but we've loaded Face SDK" << endl;
    }

    Ptr<FaceTrackerParams> params = LoadFaceTrackerParams();

    Mat gray;
    cvtColor(img, gray, CV_BGR2GRAY);
    int result = ft->NewFrame(gray, params);

    std::vector<cv::Point_<double> > shape;
    std::vector<cv::Point3_<double> > shape3;
    Pose pose;

    int tracking_threshold = 5;

    if (result >= tracking_threshold)
    {
      shape = ft->getShape();
      shape3 = ft->get3DShape();
      pose = ft->getPose();
    }
    else
    {
        cout << "Something's bad with tracking" << endl;
    }

    Mat displayedImage = img.clone();
    for (size_t i = 0; i < shape.size(); i++)
    {
        cv::circle(displayedImage, shape[i], 2, Scalar(0, 0, 255), 1, 8, 0);
    }
    imshow("image with points", displayedImage);

    int pose_image_height = 100;
    int pose_image_width = 100;
    cv::Mat poseImage = compute_pose_image(pose, pose_image_height, pose_image_width);
    imshow("pose", poseImage);
    waitKey(0);

    return shape;
}


Mat alignToLandmarks(Mat img, std::vector<Point2d> landmarks)
{
    Rect r = boundingRect(landmarks);
    Mat aligned = img(r);


    vector<Point2d> ftLandmarks = findLandmarksFaceSDK(img);




    Mat buf = drawPtsOnImg(img, landmarks);
    imshow("base landmarks", buf);
    waitKey(0);

    return aligned;
}

