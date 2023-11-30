#ifndef LIB_H
#define LIB_H

#include <opencv2/opencv.hpp>

void bilateral(const cv::Mat & image, cv::Mat & out, int d, double color, double space);
float gaussian(float x, float sigma);
void ajouterBruitSelEtPoivre(cv::Mat& imageRef, cv::Mat& image, double pourcentageBruit);
void ajouterBruitGaussian(cv::Mat& imageRef, cv::Mat& image, double valeurMoyenne);
void ajouterBruitPoisson(cv::Mat& imageRef, cv::Mat& image, double valeurMoyenne);
double calculatePSNR(const cv::Mat& img1, const cv::Mat& img2);
double calculateMean(const cv::Mat& image, const cv::Rect& region); 
double calculateVariance(const cv::Mat& image, const cv::Rect& region, double mean);
double calculateCovariance(const cv::Mat& image1, const cv::Rect& region1, const cv::Mat& image2, const cv::Rect& region2, double mean1, double mean2);
double calculateSSIM(const cv::Mat& img1, const cv::Mat& img2, int windowSize, double C1, double C2);
#endif // LIB_H

