#include "util.hpp"

#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>

using namespace std;
using namespace cv;

std::vector<std::string> &split(const std::string &s,
                                char delim,
                                std::vector<std::string> &elems)
{
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


std::vector<std::string> split(const std::string &s, char delim)
{
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}


bool loadObj(const string fname, Vertices& model)
{
    ifstream obj(fname.c_str());
    if(obj.is_open())
    {
        model.clear();
        while(!obj.eof())
        {
            string s0;
            getline(obj, s0);
            vector<string> tok = split(s0, ' ');
            if(tok.empty()) continue;
            if(tok[0][0] == '#')
            {
                //obj.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            }
            else if(tok[0][0] == 'v')
            {
                Point3d p(std::stod(tok[1]), std::stod(tok[2]), std::stod(tok[3]));
                model.push_back(p);
            }
            else
            {
                //obj.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            }
        }
        return true;
    }
    else
    {
        return false;
    }
}


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


