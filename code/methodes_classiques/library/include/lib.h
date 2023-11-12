#ifndef LIB_H
#define LIB_H

#include <opencv2/opencv.hpp>

void bilateral(const cv::Mat & image, cv::Mat & out, int d, double color, double space);
float gaussian(float x, float sigma);
void ajouterBruitSelEtPoivre(cv::Mat& imageRef, cv::Mat& image, double pourcentageBruit);
void ajouterBruitGaussian(cv::Mat& imageRef, cv::Mat& image, double valeurMoyenne);
void ajouterBruitPoisson(cv::Mat& imageRef, cv::Mat& image, double valeurMoyenne);
#endif // LIB_H
