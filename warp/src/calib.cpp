#include <stdio.h>
#include <iostream>
#include <sstream>
#include <opencv/cv.h>
#include "util_calib.h"
#include "calib.h"
#include <vector>
#include <cmath>

using namespace std;

enum MatlabInEnum {
    MATLAB_IN_WIDTH = 0,
    MATLAB_IN_HEIGHT,
    MATLAB_IN_2D_POINTS,
    MATLAB_IN_3D_POINTS,
    MATLAB_IN_A,
    MATLAB_IN_USEPOSIT,
    MATLAB_IN_ONLYEXTRINSIC,
    MATLAB_IN_R,
    MATLAB_IN_T,
    MATLAB_IN_MAXMSE
};

int cvStdErrReport2(int code, const char *func_name, const char *err_msg, const char *file,
                    int line, void*)
{
    //	ostringstream s;
    //	s << "OpenCV ERROR: " << cvErrorStr(code) << " (" << (err_msg ? err_msg : "no description")
    //			<< ")" << endl;
    //	throw s.str().c_str();
    cout << "OpenCV ERROR: " << cvErrorStr(code) << " (" << (err_msg ? err_msg : "no description")
         << ")" << endl;
    cout << (func_name ? func_name : "<unknown>") << " " << (file != NULL ? file : "") << " "
         << line << endl;
    printf("OpenCV ERROR: %s (%s)\n\tin function ", cvErrorStr(code), err_msg ? err_msg
                                                                              : "no description");
    printf("%s, %s(%d)\n", func_name ? func_name : "<unknown>", file != NULL ? file : "", line);

    return 0;
}

double distanceMSE(int & numPoints, CvPoint2D64d *& imagePoints, double *& imgPointsIn)
{
    double dist = 0;
    for (int i = 0; i < numPoints; i++)
    {
        double dist_x = (imagePoints + i)->x - imgPointsIn[i * TWO + 0];
        double dist_y = (imagePoints + i)->y - imgPointsIn[i * TWO + 1];
        dist += dist_x * dist_x + dist_y * dist_y;
    }
    return dist / numPoints;
}

// taken from OpenCV 1.1
void _cvRQDecomp3x3(const CvMat *matrixM, CvMat *matrixR, CvMat *matrixQ, CvMat *matrixQx,
                    CvMat *matrixQy, CvMat *matrixQz, CvPoint3D64f *eulerAngles)
{

    double _M[3][3], _R[3][3], _Q[3][3];
    CvMat M = cvMat(3, 3, CV_64F, _M);
    CvMat R = cvMat(3, 3, CV_64F, _R);
    CvMat Q = cvMat(3, 3, CV_64F, _Q);
    double z, c, s;

    /* Validate parameters. */
    /*    CV_ASSERT( CV_IS_MAT(matrixM) && CV_IS_MAT(matrixR) && CV_IS_MAT(matrixQ) &&
     matrixM->cols == 3 && matrixM->rows == 3 &&
     CV_ARE_SIZES_EQ(matrixM, matrixR) && CV_ARE_SIZES_EQ(matrixM, matrixQ));
     */
    cvConvert(matrixM, &M);

    {
        /* Find Givens rotation Q_x for x axis (left multiplication). */
        /*
         ( 1  0  0 )
         Qx = ( 0  c  s ), c = m33/sqrt(m32^2 + m33^2), s = m32/sqrt(m32^2 + m33^2)
         ( 0 -s  c )
         */
        s = _M[2][1];
        c = _M[2][2];
        z = 1. / sqrt(c * c + s * s + DBL_EPSILON);
        c *= z;
        s *= z;

        double _Qx[3][3] = { { 1, 0, 0 }, { 0, c, s }, { 0, -s, c } };
        CvMat Qx = cvMat(3, 3, CV_64F, _Qx);

        cvMatMul(&M, &Qx, &R);
        assert(fabs(_R[2][1]) < FLT_EPSILON);
        _R[2][1] = 0;

        /* Find Givens rotation for y axis. */
        /*
         ( c  0  s )
         Qy = ( 0  1  0 ), c = m33/sqrt(m31^2 + m33^2), s = m31/sqrt(m31^2 + m33^2)
         (-s  0  c )
         */
        s = _R[2][0];
        c = _R[2][2];
        z = 1. / sqrt(c * c + s * s + DBL_EPSILON);
        c *= z;
        s *= z;

        double _Qy[3][3] = { { c, 0, s }, { 0, 1, 0 }, { -s, 0, c } };
        CvMat Qy = cvMat(3, 3, CV_64F, _Qy);
        cvMatMul(&R, &Qy, &M);

        assert(fabs(_M[2][0]) < FLT_EPSILON);
        _M[2][0] = 0;

        /* Find Givens rotation for z axis. */
        /*
         ( c  s  0 )
         Qz = (-s  c  0 ), c = m22/sqrt(m21^2 + m22^2), s = m21/sqrt(m21^2 + m22^2)
         ( 0  0  1 )
         */

        s = _M[1][0];
        c = _M[1][1];
        z = 1. / sqrt(c * c + s * s + DBL_EPSILON);
        c *= z;
        s *= z;

        double _Qz[3][3] = { { c, s, 0 }, { -s, c, 0 }, { 0, 0, 1 } };
        CvMat Qz = cvMat(3, 3, CV_64F, _Qz);

        cvMatMul(&M, &Qz, &R);
        assert(fabs(_R[1][0]) < FLT_EPSILON);
        _R[1][0] = 0;

        // Solve the decomposition ambiguity.
        // Diagonal entries of R, except the last one, shall be positive.
        // Further rotate R by 180 degree if necessary
        if (_R[0][0] < 0)
        {
            if (_R[1][1] < 0)
            {
                // rotate around z for 180 degree, i.e. a rotation matrix of
                // [-1,  0,  0],
                // [ 0, -1,  0],
                // [ 0,  0,  1]
                _R[0][0] *= -1;
                _R[0][1] *= -1;
                _R[1][1] *= -1;

                _Qz[0][0] *= -1;
                _Qz[0][1] *= -1;
                _Qz[1][0] *= -1;
                _Qz[1][1] *= -1;
            }
            else
            {
                // rotate around y for 180 degree, i.e. a rotation matrix of
                // [-1,  0,  0],
                // [ 0,  1,  0],
                // [ 0,  0, -1]
                _R[0][0] *= -1;
                _R[0][2] *= -1;
                _R[1][2] *= -1;
                _R[2][2] *= -1;

                cvTranspose(&Qz, &Qz);

                _Qy[0][0] *= -1;
                _Qy[0][2] *= -1;
                _Qy[2][0] *= -1;
                _Qy[2][2] *= -1;
            }
        }
        else if (_R[1][1] < 0)
        {
            // ??? for some reason, we never get here ???

            // rotate around x for 180 degree, i.e. a rotation matrix of
            // [ 1,  0,  0],
            // [ 0, -1,  0],
            // [ 0,  0, -1]
            _R[0][1] *= -1;
            _R[0][2] *= -1;
            _R[1][1] *= -1;
            _R[1][2] *= -1;
            _R[2][2] *= -1;

            cvTranspose(&Qz, &Qz);
            cvTranspose(&Qy, &Qy);

            _Qx[1][1] *= -1;
            _Qx[1][2] *= -1;
            _Qx[2][1] *= -1;
            _Qx[2][2] *= -1;
        }

        // calculate the euler angle
        if (eulerAngles)
        {
            eulerAngles->x = acos(_Qx[1][1]) * (_Qx[1][2] >= 0 ? 1 : -1) * (180.0 / CV_PI);
            eulerAngles->y = acos(_Qy[0][0]) * (_Qy[0][2] >= 0 ? 1 : -1) * (180.0 / CV_PI);
            eulerAngles->z = acos(_Qz[0][0]) * (_Qz[0][1] >= 0 ? 1 : -1) * (180.0 / CV_PI);
        }

        /* Calulate orthogonal matrix. */
        /*
         Q = QzT * QyT * QxT
         */
        cvGEMM(&Qz, &Qy, 1, 0, 0, &M, CV_GEMM_A_T + CV_GEMM_B_T);
        cvGEMM(&M, &Qx, 1, 0, 0, &Q, CV_GEMM_B_T);

        /* Save R and Q matrices. */
        cvConvert( &R, matrixR );
        cvConvert( &Q, matrixQ );

        if (matrixQx)
            cvConvert(&Qx, matrixQx);
        if (matrixQy)
            cvConvert(&Qy, matrixQy);
        if (matrixQz)
            cvConvert(&Qz, matrixQz);
    }

}


void run(int width, int height, int numPoints, double *imgPointsIn, double *objPointsIn,
         double *AOutput, double *ROutput, double *TOutput, double *Ain, double max_mse,
         bool onlyExtrinsic, int useExtrinsicGuess, double *Rin, double *Tin)
{
    cvRedirectError(cvStdErrReport2);
    //	cvRedirectError(cvNulDevReport);
    cvSetErrMode(CV_ErrModeParent);
    if (imgPointsIn == NULL || objPointsIn == NULL || numPoints == 0)
    {
        return;
    }
    CvSize image_size;
    image_size.width = width;
    image_size.height = height;
    CvPoint2D64d* imagePoints;
    CvPoint3D64d* objectPoints;
    //CvPoint2D64d* reprojectPoints;

    double transVects[3];
    double rotMatrs[3 * 3];
    double cameraMatrix[3 * 3];
    for (unsigned int i = 0; i < TRIPLET * TRIPLET; i++)
    {
        cameraMatrix[i] = Ain[i];
    }
    //		memset(cameraMatrix, 0, 9 * sizeof(cameraMatrix[0]));
    if (useExtrinsicGuess == 1)
    {
        for (unsigned int i = 0; i < TRIPLET * TRIPLET; i++)
        {
            rotMatrs[i] = Rin[i];
        }
        for (unsigned int i = 0; i < TRIPLET; i++)
        {
            transVects[i] = Tin[i];
        }
    }
    double distortion[4];

    int calibFlags;

    imagePoints = 0;
    objectPoints = 0;
    //reprojectPoints = 0;

    /* Need to allocate memory */
    imagePoints = (CvPoint2D64d*) cvAlloc(numPoints * sizeof(CvPoint2D64d));

    objectPoints = (CvPoint3D64d*) cvAlloc(numPoints * sizeof(CvPoint3D64d));

    /* Alloc memory for numbers */
    int *numbers = (int*) cvAlloc(sizeof(int));
    numbers[0] = numPoints;

    //cout << "input points (x,y) and (X,Y,Z):" << endl;
    for (int i = 0; i < numPoints; i++)
    {
        (imagePoints + i)->x = imgPointsIn[i * TWO];
        (imagePoints + i)->y = imgPointsIn[i * TWO + 1];
        (objectPoints + i)->x = objPointsIn[i * TRIPLET];
        (objectPoints + i)->y = objPointsIn[i * TRIPLET + 1];
        (objectPoints + i)->z = objPointsIn[i * TRIPLET + 2];
        if (DEBUG_CALIB)
        {
            cout << (imagePoints + i)->x << tab << (imagePoints + i)->y << tab
                 << (objectPoints + i)->x << tab << (objectPoints + i)->y << tab
                 << (objectPoints + i)->z << endl;

        }
    }
    calibFlags = CV_CALIB_FIX_PRINCIPAL_POINT + CV_CALIB_ZERO_TANGENT_DIST
            + CV_CALIB_FIX_ASPECT_RATIO + CV_CALIB_USE_INTRINSIC_GUESS;

    CvMat point_counts = cvMat(1, 1, CV_32SC1, numbers);
    CvMat image_points, object_points;
    CvMat dist_coeffs = cvMat(4, 1, CV_64FC1, distortion);
    CvMat camera_matrix = cvMat(3, 3, CV_64FC1, cameraMatrix);
    CvMat rotation_matrices = cvMat(1, 9, CV_64FC1, rotMatrs);
    //	double rotMatrsc[3];
    //	for (int i = 0; i < 3; i++) {
    //		rotMatrsc[i] = 0;
    //	}
    //	CvMat rotation_matrices = cvMat(1, 3, CV_64FC1, rotMatrsc);
    CvMat translation_vectors = cvMat(1, 3, CV_64FC1, transVects);

    image_points = cvMat(numPoints, 1, CV_64FC2, imagePoints);
    object_points = cvMat(numPoints, 1, CV_64FC3, objectPoints);

    if (onlyExtrinsic)
    {
        CvMat rotation_matrices2 = cvMat(3, 3, CV_64FC1, rotMatrs);

        double tmp[3];
        CvMat rotation_matrices_tmp = cvMat(1, 3, CV_64FC1, tmp);
        cvRodrigues2(&rotation_matrices2, &rotation_matrices_tmp);
        //cout << "using cvFindExtrinsic." << endl;
        cvFindExtrinsicCameraParams2(&object_points, &image_points, &camera_matrix, &dist_coeffs,
                                     &rotation_matrices_tmp, &translation_vectors, useExtrinsicGuess);
        //convert the result and store in rotMatrs
        cvRodrigues2(&rotation_matrices_tmp, &rotation_matrices2);
    }
    else
    {
        //cout << "using cvCalibrate." << endl;
        cvCalibrateCamera2(&object_points, &image_points, &point_counts, image_size,
                           &camera_matrix, &dist_coeffs, &rotation_matrices, &translation_vectors, calibFlags);
    }

    bool err = false;
    try
    {
        if (cvGetErrStatus() != CV_StsOk)
        {
            throw "Error";
        }

        if (AOutput != NULL && ROutput != NULL && TOutput != NULL)
        {
            transpose(cameraMatrix, 3, 3, AOutput);
            transpose(rotMatrs, 3, 3, ROutput);
            transpose(transVects, 1, 3, TOutput);
        }
        if (DEBUG_CALIB)
        {

            //cout << "cameraMatrix:" << endl;
            printMatrix(cameraMatrix, 3, 3);
            //cout << "distortion:" << endl;
            printMatrix(distortion, 1, 4);
            //cout << "rotation:" << endl;
            printMatrix(rotMatrs, 3, 3);
            //cout << "translation:" << endl;
            printMatrix(transVects, 1, 3);

            rotation_matrices = cvMat(3, 3, CV_64FC1, rotMatrs);
            for (int i = 0; i < numPoints; i++)
            {
                (imagePoints + i)->x = 0;
                (imagePoints + i)->y = 0;
            }
            image_points = cvMat(numPoints, 1, CV_64FC2, imagePoints);
            cvProjectPoints2(&object_points, &rotation_matrices, &translation_vectors,
                             &camera_matrix, &dist_coeffs, &image_points);
            if (cvGetErrStatus() != CV_StsOk)
            {
                throw "Error";
            }
            //cout << "reprojection (with distortion):" << endl;
            for (int i = 0; i < numPoints; i++)
            {
                //cout << (imagePoints + i)->x << tab << (imagePoints + i)->y << endl;
            }

            for (int i = 0; i < 4; i++)
            {
                distortion[i] = 0;
            }
            dist_coeffs = cvMat(4, 1, CV_64FC1, distortion);

            for (int i = 0; i < numPoints; i++)
            {
                (imagePoints + i)->x = 0;
                (imagePoints + i)->y = 0;
            }
            image_points = cvMat(numPoints, 1, CV_64FC2, imagePoints);
            cvProjectPoints2(&object_points, &rotation_matrices, &translation_vectors,
                             &camera_matrix, &dist_coeffs, &image_points);
            if (cvGetErrStatus() != CV_StsOk)
            {
                throw "Error";
            }
            //cout << "reprojection (without distortion):" << endl;
            for (int i = 0; i < numPoints; i++)
            {
                //cout << (imagePoints + i)->x << tab << (imagePoints + i)->y << endl;
            }

            double mse = distanceMSE(numPoints, imagePoints, imgPointsIn);
            //cout << "mean square error: " << mse << endl;
            if (max_mse > 0 && mse > max_mse)
            {
                throw string("mean square error is bigger than max_mse: " + toString(max_mse)).c_str();
            }

            double matrixR_[9];
            CvMat matrixR = cvMat(3, 3, CV_64FC1, matrixR_);
            double matrixQ_[9];
            CvMat matrixQ = cvMat(3, 3, CV_64FC1, matrixQ_);
            double matrixQx_[9];
            CvMat matrixQx = cvMat(3, 3, CV_64FC1, matrixQx_);
            double matrixQy_[9];
            CvMat matrixQy = cvMat(3, 3, CV_64FC1, matrixQy_);
            double matrixQz_[9];
            CvMat matrixQz = cvMat(3, 3, CV_64FC1, matrixQz_);
            CvPoint3D64f eulerAngles;
            _cvRQDecomp3x3(&rotation_matrices, &matrixR, &matrixQ, &matrixQx, &matrixQy, &matrixQz,
                           &eulerAngles);
            if (cvGetErrStatus() != CV_StsOk)
            {
                throw "Error";
            }
            //cout << "R" << endl;
            printMatrix(matrixR_, 3, 3);
            //cout << "Q" << endl;
            printMatrix(matrixQ_, 3, 3);
            //cout << "Qx" << endl;
            printMatrix(matrixQx_, 3, 3);
            //cout << "Qy" << endl;
            printMatrix(matrixQy_, 3, 3);
            //cout << "Qz" << endl;
            printMatrix(matrixQz_, 3, 3);
            //cout << "euler angles:" << endl;
            //cout << eulerAngles.x << tab << eulerAngles.y << tab << eulerAngles.z << endl;
        }
    }
    catch (const char *e)
    {
        puts(e);
        printf("\n");
        //cout << e << endl;
        err = true;
        //		sout<<cvErrorStr(cvGetErrStatus())<<endl;

    }
    cvFree(&imagePoints);
    cvFree(&objectPoints);
    cvFree(&numbers);
    cvRedirectError(cvNulDevReport);
    cvSetErrStatus(CV_StsOk);
    if (err)
    {
        throw 1;
    }
}


int exampleMain()
{
    int width = 500;
    int height = 500;
    const int numPoints = 6;

    //test
    double imgPoints[numPoints * TWO];
    double objPoints[numPoints * TRIPLET];

    int i = 0;
    imgPoints[i * TWO] = 148;
    imgPoints[i * TWO + 1] = 124;
    objPoints[i * TRIPLET] = -3.0111;
    objPoints[i * TRIPLET + 1] = -2.0266;
    objPoints[i * TRIPLET + 2] = 0.9845;
    i++;
    imgPoints[i * TWO] = 51;
    imgPoints[i * TWO + 1] = 373;
    objPoints[i * TRIPLET] = -3.9661;
    objPoints[i * TRIPLET + 1] = -2.0266;
    objPoints[i * TRIPLET + 2] = -0.9766;
    i++;
    imgPoints[i * TWO] = 451;
    imgPoints[i * TWO + 1] = 125;
    objPoints[i * TRIPLET] = -0.0280;
    objPoints[i * TRIPLET + 1] = -2.0266;
    objPoints[i * TRIPLET + 2] = 0.9766;
    i++;
    imgPoints[i * TWO] = 449;
    imgPoints[i * TWO + 1] = 373;
    objPoints[i * TRIPLET] = -0.0477;
    objPoints[i * TRIPLET + 1] = -2.0266;
    objPoints[i * TRIPLET + 2] = -0.9766;
    i++;
    imgPoints[i * TWO] = 107;
    imgPoints[i * TWO + 1] = 231;
    objPoints[i * TRIPLET] = -3.4147;
    objPoints[i * TRIPLET + 1] = -2.0266;
    objPoints[i * TRIPLET + 2] = 0.1418;
    i++;
    imgPoints[i * TWO] = 81;
    imgPoints[i * TWO + 1] = 300;
    objPoints[i * TRIPLET] = -3.6707;
    objPoints[i * TRIPLET + 1] = -2.0266;
    objPoints[i * TRIPLET + 2] = -0.4017;
    i++;

    //bool usePOSIT = false;
    bool onlyExtrinsic = false;
    double Ain[9] = { 770.455, 0, 250, 0, 963.068, 250, 0, 0, 1 };
    try {
        run(width, height, numPoints, imgPoints, objPoints, NULL, NULL, NULL, Ain, 0,
            onlyExtrinsic, 0, NULL, NULL);
        printf("ok\n");
    } catch (int err) {
    }
    //printf("%s", sout.str().c_str());
    return 0;
}


void shutdown(double *imgPoints, double *objPoints)
{
    if (NULL != imgPoints)
    {
        delete imgPoints;
    }
    if (NULL != objPoints)
    {
        delete objPoints;
    }
}


void startCalibration(cv::Size picSize,
                      std::vector<cv::Point2d> pts2d, std::vector<cv::Point3d> pts3d,
                      cv::Mat camMat, cv::Mat rmat, cv::Mat tvec,
                      bool onlyExtrinsic, int useExtrinsicGuess)
{
    double* imgPoints = NULL;
    double* objPoints = NULL;
    try
    {
        // input
        unsigned int width = picSize.width;
        unsigned int height = picSize.height;

        imgPoints = new double[pts2d.size()*2];
        for(size_t i = 0; i < pts2d.size(); i++)
        {
            imgPoints[i*2  ] = pts2d[i].x;
            imgPoints[i*2+1] = pts2d[i].y;
        }

        objPoints = new double[pts3d.size()*3];
        for(size_t i = 0; i < pts3d.size(); i++)
        {
            objPoints[i*3  ] = pts3d[i].x;
            objPoints[i*3+1] = pts3d[i].y;
            objPoints[i*3+2] = pts3d[i].z;
        }
        int numPoints = pts3d.size();
        assert(pts2d.size() == pts3d.size());

        cv::Mat cammClone = camMat.clone();
        cv::Mat rmatClone = rmat.clone();
        cv::Mat tvecClone = tvec.clone();

        double* Ain = (double*)(cammClone.data);
        double* Rin = (double*)(rmatClone.data);
        double* Tin = (double*)(tvecClone.data);

        double max_mse = 0;

        // output
        double *AOutput = (double*)camMat.data;
        double *ROutput = (double*)rmat.data;
        double *TOutput = (double*)tvec.data;

        //cout << cv::Mat(pts2d) << endl;

        run(width, height, numPoints, imgPoints, objPoints,
            AOutput, ROutput, TOutput, Ain,
            max_mse, onlyExtrinsic, useExtrinsicGuess, Rin, Tin);

        rmat = rmat.t();

        shutdown(imgPoints, objPoints);
    }
    catch (int err)
    {
        //printf("%s", sout.str().c_str());
        shutdown(imgPoints, objPoints);
    }
}
