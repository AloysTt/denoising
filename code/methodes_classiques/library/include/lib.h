#ifndef LIB_H
#define LIB_H

#include <opencv2/opencv.hpp>

void bilateral(const cv::Mat & image, cv::Mat & out, int d, double color, double space);
float gaussian(float x, float sigma);

#endif // LIB_H
