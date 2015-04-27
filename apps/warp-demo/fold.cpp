#include "fold.hpp"

using namespace std;
using namespace cv;


Ptr<FoldData> readFold(string fname)
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
}
