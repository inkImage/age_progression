#include <iostream>
#include <opencv2/opencv.hpp>
#include "warp.hpp"

#include "fold.hpp"

using namespace cv;
using namespace std;

struct Settings
{
    vector<string> foldsFnames;
    string outPcaFname;
};

vector<string> loadSimpleList(string fname)
{
    vector<string> fileList;
    string dirName = fname.substr(0, fname.find_last_of("\\/"));
    ifstream file(fname.c_str());
    if(file.is_open())
    {
        string s;
        do
        {
            s.clear();
            file >> s;
            fileList.push_back(dirName +'/' + s);
        }
        while(!s.empty());
    }
    else
    {
        throw std::runtime_error("no such file: "+fname);
    }
    return fileList;
}

Settings loadSettings(string fname)
{
    ifstream file(fname);
    Settings settings;
    if(file.is_open())
    {
        while(true)
        {
            string s;
            file >> s;
            if(s.empty()) break;
            if(s == "fold")
            {
                string fname;
                file >> fname;
                settings.foldsFnames.push_back(fname);
            }
            else if(s == "outpca")
            {
                string fname;
                file >> fname;
                settings.outPcaFname = fname;
            }
        }
    }
    else
    {
        throw std::runtime_error("can't open settings file: "+fname);
    }
    return settings;
}

int main(int argc, char** argv)
{
    string settingsFname = argc > 1 ? argv[1] : string();
    Settings settings = loadSettings(settingsFname);
    Ptr<FoldData> data;
    for(size_t i = 0; i < settings.foldsFnames.size(); i++)
    {
        Ptr<FoldData> fold = readFold(listFname);
        if(data.empty())
        {
            data = fold;
        }
        else
        {
            data->insert(data->end(), fold->begin(), fold->end());
        }
    }

    size_t numImages = data->size();
    Size picSize(100, 100);
    int numPixels = picSize.width*picSize.height;
    Mat pcaMat(numPixels, numImages, CV_32F);

    for(size_t i = 0; i < numImages; i++)
    {
        string fname = data->at(i).imgFname;
        Mat image = imread(fname);
        Mat gray, cropped, floatImg, small, column;
        cvtColor(image, gray, CV_BGR2GRAY, 1);
        vector<Point2d> landmarks = data->at(i).landmarks;
        Rect r = boundingRect(landmarks);
        cropped = gray(r);
        resize(cropped, small, picSize);
        small.convertTo(floatImg, CV_32F);
        column = floatImg.reshape(1, numPixels);

        column.copyTo(pcaMat(Rect(i, 0, 1, numPixels)));

        cout << "Gathering images... ";
        cout << i << "/" << numImages << endl;
    }

    cout << "Computing PCA..." << endl;
    int numVectors = 12;
    PCA pca(pcaMat, Mat(), CV_PCA_DATA_AS_COL, numVectors);
    Mat eigenValues  = pca.eigenvalues;
    Mat eigenVectors = pca.eigenvectors.t();

    FileStorage fs(settings.outPcaFname, FileStorage::WRITE);
    if(fs.isOpened())
    {
        fs << "eigenValues"  << eigenValues;
        fs << "eigenVectors" << eigenVectors;
        fs << "mean" << mean;
    }
    else
    {
        throw std::runtime_error("can't save file: "+settings.outPcaFname);
    }

    Mat meanToShow, floatMean;
    floatMean = pca.mean.reshape(1, picSize.height);
    convertScaleAbs(floatMean, meanToShow);
    imshow("mean", meanToShow);
    waitKey(0);

    //draw svd
    for(int i = 0; i < eigenValues.rows; i++)
    {
        cout << "Drawing u and vt... " << i << " ";
        cout << eigenValues.at<float>(i, 0) << endl;
        Mat floatU, imgU;
        eigenVectors(Rect(i, 0, 1, numPixels)).copyTo(floatU);
        floatU = floatMean + floatU*255.0;
        convertScaleAbs(floatU.reshape(1, picSize.height), imgU);

        imshow("u", imgU); //imshow("vt", imgV);
        waitKey(0);
    }

    return 0;
}
