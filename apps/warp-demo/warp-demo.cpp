#include <iostream>
#include <opencv2/opencv.hpp>
#include "warp.hpp"

#include "fold.hpp"
#include "util.hpp"

using namespace cv;
using namespace std;

const bool DRAW_PCA = true;


struct Settings
{
    vector<string> foldsFnames;
    string outPcaFname;
};


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
            else
            {
                file.ignore(400, '\n');
            }
        }
    }
    else
    {
        throw std::runtime_error("can't open settings file: "+fname);
    }
    return settings;
}


Mat fromColumnToImage(Mat column, int rows)
{
    Mat img;
    vector<Mat> bgr(3);
    for(int i = 0; i < 3; i++)
    {
        Mat col = column(Rect(0, i*column.rows/3, 1, column.rows/3));
        Mat ucharCol;
        convertScaleAbs(col, ucharCol);
        bgr[i] = ucharCol.reshape(1, rows);
    }
    merge(bgr, img);
    return img;
}


int main(int argc, char** argv)
{
    cout << "Loading settings..." << endl;
    string settingsFname = argc > 1 ? argv[1] : string();
    Settings settings = loadSettings(settingsFname);
    Ptr<FoldData> data;
    for(size_t i = 0; i < settings.foldsFnames.size(); i++)
    {
        cout << "Reading folds... ";
        cout  << i+1 << '/' << settings.foldsFnames.size() << endl;
        Ptr<FoldData> fold = readFold(settings.foldsFnames[i]);
        if(data.empty())
        {
            data = fold;
        }
        else
        {
            data->insert(data->end(), fold->begin(), fold->end());
        }
    }

    const bool exportTxt = false;
    if(exportTxt)
    {
        ofstream tempFile("/tmp/flist.txt");
        if(tempFile.is_open())
        {
            for(size_t i = 0; i < data->size(); i++)
            {
                tempFile << data->at(i).imgFname << " ";
                tempFile << data->at(i).ageLow << " ";
                tempFile << data->at(i).ageHigh << " ";
                tempFile << (data->at(i).gender ? "f" : "m") << endl;
            }
        }
        tempFile.close();
    }

    size_t numImages = data->size();
    Size picSize(100, 100);
    int numPixels = picSize.width*picSize.height;

    Mat eigenValues;
    Mat eigenVectors;
    Mat meanVector;

    FileStorage fs(settings.outPcaFname, FileStorage::READ);
    if(fs.isOpened())
    {
        fs["eigenValues"] >> eigenValues;
        fs["eigenVectors"] >> eigenVectors;
        fs["mean"] >> meanVector;
    }
    else
    {
        Mat pcaMat(3*numPixels, numImages, CV_32F);
        for(size_t i = 0; i < numImages; i++)
        {
            string fname = data->at(i).imgFname;

            //frontalized
            replace(fname, "AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification",
                    "frontalized_Adience3D.0.1.1");

            Mat image = imread(fname);
            Mat aligned, column;
            vector<Point2d> landmarks = data->at(i).landmarks;

            //aligned = alignToLandmarks(image, landmarks);
            Rect boundR(14, 14, 114, 114);
            aligned = image(boundR);

            //imshow("aligned", aligned); waitKey(0);

            vector<Mat> bgr;
            split(aligned, bgr);
            for(size_t i = 0; i < bgr.size(); i++)
            {
                Mat small, floatImg, col;
                resize(bgr[i], small, picSize);
                small.convertTo(floatImg, CV_32F);
                col = floatImg.reshape(1, numPixels);
                column.push_back(col);
            }

            column.copyTo(pcaMat(Rect(i, 0, 1, 3*numPixels)));

            cout << "Gathering images... ";
            cout << i << "/" << numImages << endl;
        }

        cout << "Computing PCA..." << endl;
        int numVectors = 20;
        PCA pca(pcaMat, Mat(), CV_PCA_DATA_AS_COL, numVectors);
        eigenValues  = pca.eigenvalues;
        eigenVectors = pca.eigenvectors.t();
        meanVector = pca.mean;

        FileStorage fsWrite(settings.outPcaFname, FileStorage::WRITE);
        if(fsWrite.isOpened())
        {
            fsWrite << "eigenValues"  << eigenValues;
            fsWrite << "eigenVectors" << eigenVectors;
            fsWrite << "mean" << meanVector;
        }
        else
        {
            throw std::runtime_error("can't save file: "+settings.outPcaFname);
        }
    }

    PCA pca;
    pca.eigenvalues = eigenValues;
    pca.eigenvectors = eigenVectors.t();
    pca.mean = meanVector;

    //draw pca
    if(DRAW_PCA)
    {
        Mat meanToShow;
        meanToShow = fromColumnToImage(pca.mean, picSize.height);
        imshow("mean", meanToShow);
        waitKey(0);

        for(int i = 0; i < 20; i++)
        {
            double eig = eigenValues.at<float>(i, 0);
            cout << "Drawing u and vt... " << i << " " << eig << endl;
            Mat floatU, imgU;
            eigenVectors(Rect(i, 0, 1, 3*numPixels)).copyTo(floatU);
            double minv, maxv;
            cv::minMaxIdx(floatU, &minv, &maxv);
            floatU = 128.0 + floatU*(128.0/maxv);
            imgU = fromColumnToImage(floatU, picSize.height);

            imshow("u"+to_string(i), imgU); //imshow("vt", imgV);
            waitKey(0);
        }
    }

    Mat meanImg(picSize, CV_32FC3, Scalar::all(0));

    for(size_t i = 0; i < numImages; i++)
    {
        cout << "Projecting images... ";
        cout << i+1 << "/" << numImages << endl;

        string fname = data->at(i).imgFname;

        //frontalized
        //replace(fname, "AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification",
        //        "frontalized_Adience3D.0.1.1");

        Mat image = imread(fname);

        Mat resized, column;
        vector<Point2d> landmarks = data->at(i).landmarks;

        Mat aligned = alignToLandmarks(image, landmarks);
        //Rect boundR = boundingRect(landmarks);
//        Rect boundR(14, 14, 114, 114);
//        Mat aligned = image(boundR);

        resize(aligned, resized, picSize);

        //imshow("image", resized);

        vector<Mat> bgr;
        split(resized, bgr);
        for(size_t i = 0; i < bgr.size(); i++)
        {
            Mat floatImg, col;
            bgr[i].convertTo(floatImg, CV_32F);
            col = floatImg.reshape(1, numPixels);
            column.push_back(col);
        }

        Mat projection = pca.project(column);
        Mat restored = pca.backProject(projection);

        Mat toShow = fromColumnToImage(restored, picSize.height);

        //imshow("projection", toShow);

        cout << "Searching for optical flow..." << endl;
        Mat flow = findOpticalFlow(resized, toShow);
        cout << "Warping..." << endl;
        Mat warped = warpImgWithFlow(resized, flow);

        Mat converted;
        warped.convertTo(converted, CV_32FC3);
        meanImg += converted/numImages;

        //imshow("warped", warped);
        //waitKey(0);
    }

    Mat meanImgConverted;
    meanImg.convertTo(meanImgConverted, CV_8UC3);

    imshow("meanImg", meanImgConverted);
    waitKey(0);

    return 0;
}
