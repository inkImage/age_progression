#ifndef UTIL_HPP
#define UTIL_HPP

#include <iostream>
#include <opencv2/opencv.hpp>

typedef std::vector<cv::Point3d> Vertices;

bool loadObj(const std::string fname, Vertices& model);
std::vector<std::string> loadSimpleList(std::string fname);
std::vector<std::string> split(const std::string &s, char delim);
std::vector<std::string> &split(const std::string &s,
                                char delim,
                                std::vector<std::string> &elems);
bool replace(std::string& str, const std::string& from, const std::string& to);


#endif //UTIL_HPP
