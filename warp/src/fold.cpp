#include "fold.hpp"
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>

using namespace std;
using namespace cv;


vector<Point2d> readFidu(string fname, int& score, int& yaw)
{
    ifstream file(fname.c_str());
    if(file.is_open())
    {
        vector<Point2d> pts;
        string s0;
        //score,yaw_angle
        getline(file, s0);
        vector<string> tok = split(s0, ',');
        score = stoi(tok[0]);
        yaw   = stoi(tok[1]);
        //"x,y,dx,dy,x_c,y_c"
        getline(file, s0);
        while(true)
        {
            string s;
            getline(file, s);
            if(s.empty()) break;
            vector<string> tokens = split(s, ',');
            //stuff,stuff,stuff,stuff,x,y
            Point2d p(std::stod(tokens[4]), std::stod(tokens[5]));
            pts.push_back(p);
        }

        return pts;
    }
    else
    {
        throw std::runtime_error("no such file: "+fname);
    }

    return vector<Point2d>();
}


Ptr<FoldData> readFold(string fname)
{
    //user_id  |original_image     |face_id        |age |gender
    //x        |y                  |dx             |dy
    //tilt_ang |fiducial_yaw_angle |fiducial_score

    string dirName = fname.substr(0, fname.find_last_of("\\/"));
    ifstream file(fname.c_str());
    if(file.is_open())
    {
        Ptr<FoldData> fold = new FoldData;
        //skip caption
        string dummy;
        getline(file, dummy);

        string s;
        while(true)
        {
            getline(file, s);
            if(s.empty()) break;
            vector<string> tokens = split(s, '\t');

            FaceData face;
            face.userId = tokens[0];
            face.originalImage = tokens[1];
            face.faceId = stoi(tokens[2]);
            string& age = tokens[3];
            if(age == "None")
            {
                face.ageHigh = face.ageLow = -1;
            }
            else if(age[0] == '(')
            {
                int comma = age.find_first_of(",");
                string ageLow  = age.substr(1, comma-1);
                string ageHigh = age.substr(comma+2, age.size()-1-(comma+2));
                face.ageLow = stoi(ageLow);
                face.ageHigh = stoi(ageHigh);
            }
            else
            {
                //one number
                face.ageLow = face.ageHigh = stoi(age);
            }
            string& gnd = tokens[4];
            if(gnd.empty())
            {
                face.gender = -1;
            }
            else if(gnd[0] == 'm')
            {
                face.gender = 0;
            }
            else if(gnd[0] == 'f')
            {
                face.gender = 1;
            }
            else
            {
                face.gender = -1;
            }
            face.faceRect = Rect(stoi(tokens[5]), stoi(tokens[6]),
                                 stoi(tokens[7]), stoi(tokens[8]));
            face.tiltAng  = stoi(tokens[9]);
            face.fiducialYawAngle = stoi(tokens[10]);
            face.fiducialScore = stoi(tokens[11]);

            face.imgFname = dirName + "/faces/" +
                            face.userId + '/' +
                            "coarse_tilt_aligned_face." +
                            to_string(face.faceId) + "." +
                            face.originalImage;
            string noExtFname = face.originalImage.substr(0, face.originalImage.size()-4);
            string ptsFname = dirName + "/faces/" +
                              face.userId + '/' +
                              "landmarks." +
                              to_string(face.faceId) + "." +
                              noExtFname + ".txt";
            int score, yaw;
            face.landmarks = readFidu(ptsFname, score, yaw);
            face.score2 = score;
            face.yaw2   = yaw;

            fold->push_back(face);
        }
        return fold;
    }
    else
    {
        throw std::runtime_error("no such file: "+fname);
    }
}
